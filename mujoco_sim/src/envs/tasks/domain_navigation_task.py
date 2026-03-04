import numpy as np
from .base_task import BaseTask
from src.utils.astar_planner import AStarPlanner

class DomainNavigationTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = self.config.reward_weights
        self.goals = self.config.goals
        
        self.safety = getattr(self.config, 'safety', None)
        if self.safety is None:
            class DummySafety:
                warning_distance = 4.0
                critical_distance = 0.4
            self.safety = DummySafety()

        # [修复 Bug 2] 观测维度更新为 39 (因为增加了长度为 3 的 CTE 向量)
        self.obs_dim = 39 
        self.planner = AStarPlanner(resolution=0.05, safe_margin=1.7)
        self.waypoints =[]
        
        self.last_action = None
        self.last_s_current = 0.0
        self.max_s_reached = 0.0  # [新增] 记录历史最高进度，确保不倒退
        self.smoothed_lookahead_pt = None
        
        # 纯追踪参数 (可在这里微调)
        self.lookahead_distance = 0.7     
        self.search_window = 3             
        
    def get_obs_dim(self):
        return self.obs_dim

    def reset(self, env):
        self.last_action = np.zeros(env.action_space.shape)
        self.current_is_collision = False
        self.max_s_reached = 0.0  # [新增] 每回合重置最高进度
        
        env.target_pos = getattr(env, 'fixed_target_pos', np.array([18.0, 0.0, 10.0]))
        active_obstacles = env.scene_builder.get_active_obstacles() 
        start_pos = env.data.xpos[env.robot.body_id].copy()
        
        # 1. 规划 A* 路径
        self.waypoints = self.planner.plan(start_pos, env.target_pos, active_obstacles)
        
        # 2. 预计算线段累加距离 (1D S坐标系统)
        self.wp_cum_dists = [0.0]
        if len(self.waypoints) > 1:
            for i in range(1, len(self.waypoints)):
                dist = np.linalg.norm(self.waypoints[i] - self.waypoints[i-1])
                self.wp_cum_dists.append(self.wp_cum_dists[-1] + dist)
        self.total_path_length = self.wp_cum_dists[-1] if len(self.waypoints) > 1 else 0.0

        # 3. 初始化状态
        self.current_seg_idx = 0
        s_curr, _ = self._get_projection_status(start_pos)
        self.last_s_current = s_curr
        self.last_s_current = self.max_s_reached  # [修改] 对齐最大进度
        
        # 强制初始化一个平滑点 (基于 max_s_reached)
        self.smoothed_lookahead_pt = self._map_1d_s_to_3d_pos(self.max_s_reached + self.lookahead_distance)
        

    def _get_projection_status(self, pos):
        """计算 AUV 在路径上的 1D 进度与横向误差"""
        if getattr(self, 'waypoints', None) is None or len(self.waypoints) < 2:
            return 0.0, np.linalg.norm(pos - getattr(self, 'env', None).target_pos)

        end_idx = min(self.current_seg_idx + self.search_window, len(self.waypoints) - 1)
        min_dist_to_path = float('inf')
        best_i = self.current_seg_idx
        best_t = 0.0

        for i in range(self.current_seg_idx, end_idx):
            A = self.waypoints[i]
            B = self.waypoints[i+1]
            AB = B - A
            len_sq = np.dot(AB, AB)
            if len_sq < 1e-6:
                continue

            AP = pos - A
            t = np.clip(np.dot(AP, AB) / len_sq, 0.0, 1.0)
            proj = A + t * AB

            dist_to_path = np.linalg.norm(pos - proj)
            if dist_to_path < min_dist_to_path:
                min_dist_to_path = dist_to_path
                best_i = i
                best_t = t

        if best_i > self.current_seg_idx:
            self.current_seg_idx = best_i

        segment_len = np.linalg.norm(self.waypoints[best_i+1] - self.waypoints[best_i])
        S_current = self.wp_cum_dists[best_i] + best_t * segment_len

        # [核心新增] 确保记录下历史走过的最远 S 距离，维持单调性
        if not hasattr(self, 'max_s_reached'):
            self.max_s_reached = 0.0
        self.max_s_reached = max(self.max_s_reached, S_current)

        return S_current, min_dist_to_path

    def _map_1d_s_to_3d_pos(self, S_target):
        """1D 映射回 3D"""
        if S_target <= 0.0: return self.waypoints[0]
        if S_target >= self.total_path_length: return self.waypoints[-1]

        for i in range(len(self.waypoints) - 1):
            if self.wp_cum_dists[i] <= S_target <= self.wp_cum_dists[i+1] + 1e-5:
                seg_start_s = self.wp_cum_dists[i]
                seg_len = self.wp_cum_dists[i+1] - seg_start_s
                if seg_len < 1e-6:
                    return self.waypoints[i]
                t = (S_target - seg_start_s) / seg_len
                A = self.waypoints[i]
                B = self.waypoints[i+1]
                return A + t * (B - A)
        return self.waypoints[-1]

    def compute_reward(self, env, action, obs):
        raw = env.sensors.get_raw_data()
        body_id = env.model.body('yuyuan').id 
        rot_mat = env.data.xmat[body_id].reshape(3, 3)
        pos = env.data.xpos[body_id].copy()

        # 1. 获取进度
        s_current, cross_track_error = self._get_projection_status(pos)
        
        # [修复逻辑] A. 推进奖励使用 max_s_reached 结算，防止倒退再前进时反复刷分
        delta_s = self.max_s_reached - self.last_s_current
        # [修改点] CTE 容忍度，假设偏离 0.5 米，收益减半
        cte_tolerance = 0.35 
        progress_multiplier = np.exp(- (cross_track_error / cte_tolerance) ** 2)
        reward_progress = delta_s * getattr(self.cfg, 'w_progress', 40.0) * progress_multiplier

        # [修改建议] B. 补充一个小额的绝对值贴轨惩罚 (加上这行)
        reward_cte = np.abs(cross_track_error) * getattr(self.cfg, 'w_cte', 5.0) # 权重可以调小一点

        # [修复 Bug 3] C. 姿态防飞天惩罚 (利用旋转矩阵Z轴分量，完全无Bug)
        # 如果 AUV 水平，Z轴朝向正上方 [0, 0, 1]。
        # 如果抬头或者翻滚，z_x 和 z_y 会显著增大。用这两个平方和能完美衡量俯仰与横滚惩罚。
        z_x, z_y = rot_mat[0, 2], rot_mat[1, 2]
        cost_pitch_roll = (z_x**2 + z_y**2) * getattr(self.cfg, 'w_pitch_roll', 20.0)

        # D. 朝向对齐奖励
        body_x = rot_mat[:, 0]
        # 【重要】这里直接使用 get_obs 更新好的平滑点，防止状态冲突
        vec_target = self.smoothed_lookahead_pt - pos
        dist_lookahead = np.linalg.norm(vec_target)
        desired_x = vec_target / (dist_lookahead + 1e-6)
        align_cos = np.dot(body_x, desired_x)
        reward_align = align_cos * getattr(self.cfg, 'w_align', 2.0)

        # 2. 动力学限制
        local_vel = raw.get('dvl', np.zeros(3))
        v_surge, v_sway, v_heave = local_vel[0], local_vel[1], local_vel[2]
        cost_sway_heave = getattr(self.cfg, 'w_sway_vel', 2.0) * (v_sway**2) + \
                          getattr(self.cfg, 'w_heave_vel', 2.0) * (v_heave**2)
        cost_action = getattr(self.cfg, 'w_accel', 0.15) * np.sum(np.square(action))
        cost_smooth = getattr(self.cfg, 'w_delta_accel', 0.5) * np.sum(np.square(action - self.last_action))

        # 3. 避障惩罚
        sonar_dists = raw.get('sonar', np.ones(15) * 12.0)
        min_sonar_dist = np.min(sonar_dists)
        reward_obstacle_penalty = 0.0
        self.current_is_collision = False

        if min_sonar_dist < self.safety.critical_distance:
            self.current_is_collision = True
            penalty_factor = (self.safety.critical_distance - min_sonar_dist) / self.safety.critical_distance
            reward_obstacle_penalty = penalty_factor * getattr(self.cfg, 'w_collision_step', 5.0)

        # 4. 成功判定
        reward_success = 0.0
        is_success = False
        dist_to_final = np.linalg.norm(pos - env.target_pos)

        if dist_to_final < self.goals.success_dist: 
            is_success = True
            reward_success = self.cfg.success + getattr(self.cfg, 'w_final_bonus', 200.0)
            time_penalty_applied = 0.0
        else:
            time_penalty_applied = self.cfg.time_penalty

        # 5. 总分
        total_reward = (
            reward_progress         
            + reward_align          
            + reward_success 
            - reward_cte              
            - cost_pitch_roll       
            - reward_obstacle_penalty
            - cost_sway_heave       
            - cost_action           
            - cost_smooth           
            - time_penalty_applied  
        )

        self.last_s_current = self.max_s_reached
        self.last_action = action.copy()

        info = {
            "rew/progress": reward_progress,
            "rew/cte_penalty": -reward_cte,
            "rew/pitch_roll_cost": -cost_pitch_roll,
            "state/s_current": s_current,
            "state/cross_track_err": cross_track_error,
            "is_success": float(is_success),
            "is_collision": float(self.current_is_collision)     
        }
        
        return total_reward, is_success, info
    
    def is_done(self, env, current_step, max_steps):
        body_id = env.model.body('yuyuan').id 
        pos = env.data.xpos[body_id].copy()
        target = env.target_pos
        
        if np.linalg.norm(pos - target) < self.goals.success_dist:
            return True, "success"
        if current_step >= max_steps:
            return True, "timeout"
        return False, None

    def get_obs(self, env):
        raw = env.sensors.get_raw_data()
        pos_world = env.data.xpos[env.robot.body_id]
        rot_mat = env.data.xmat[env.robot.body_id].reshape(3, 3)

        #[修复 Bug 1] 彻底移除被删除的方法调用，使用全新的纯追踪计算
        s_curr, cross_err = self._get_projection_status(pos_world)
        target_s = min(self.max_s_reached + self.lookahead_distance, self.total_path_length)
        raw_lookahead_pt = self._map_1d_s_to_3d_pos(target_s)

        # [修复 Bug 4] 仅在此处做平滑滤波，供全局(含 compute_reward)使用
        alpha = 0.5
        if self.smoothed_lookahead_pt is None:
            self.smoothed_lookahead_pt = raw_lookahead_pt.copy()
        self.smoothed_lookahead_pt = (1.0 - alpha) * self.smoothed_lookahead_pt + alpha * raw_lookahead_pt
        
        target_vec_world = self.smoothed_lookahead_pt - pos_world
        target_vec_body = rot_mat.T @ target_vec_world
        gravity_body = rot_mat.T @ np.array([0., 0., -1.]) 
        
        obs_pos = np.clip(target_vec_body / self.goals.max_dist, -1.0, 1.0)
        obs_vel = np.clip(raw['dvl'] / 2.0, -1.0, 1.0)
        obs_gyro = np.clip(raw['gyro'] / 6.0, -1.0, 1.0)
        obs_quat = raw['quat']
        depth = env.WATER_SURFACE_Z - pos_world[2]
        obs_depth = np.array([np.clip(depth / 50.0, 0.0, 1.0)])
        
        obs_sonar = np.clip(raw.get('sonar', np.zeros(15)) / 12.0, 0.0, 1.0)
        obs_alt = np.array([np.clip(raw.get('altitude', 0) / 50.0, 0.0, 1.0)])
        obs_accel = np.clip(raw['accel'] / 9.81, -3.0, 3.0)

        # 核心：计算 CTE 并转到局部坐标系加入网络观测
        proj_pt_world = self._map_1d_s_to_3d_pos(s_curr)
        cte_vec_world = proj_pt_world - pos_world
        cte_vec_body = rot_mat.T @ cte_vec_world
        obs_cte = np.clip(cte_vec_body / 5.0, -1.0, 1.0) 

        obs = np.concatenate([
            obs_pos, obs_vel, obs_gyro, obs_quat, gravity_body,
            obs_depth, obs_sonar, obs_alt, obs_accel, obs_cte
        ]).astype(np.float32) 
        
        return obs