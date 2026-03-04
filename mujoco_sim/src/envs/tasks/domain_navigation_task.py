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

        self.obs_dim = 36 
        self.planner = AStarPlanner(resolution=0.05, safe_margin=1.7)
        self.waypoints =[]
        self.current_lookahead_pt = None
        self.last_action = None
        self.last_path_potential = None 
        self.just_reached_waypoint = False # [修复] 初始化标志位
        
    def get_obs_dim(self):
        return self.obs_dim

    def reset(self, env):
        self.last_action = None
        self.current_is_collision = False
        self.just_reached_waypoint = False
        
        env.target_pos = getattr(env, 'fixed_target_pos', np.array([18.0, 0.0, 10.0]))
        active_obstacles = env.scene_builder.get_active_obstacles() 
        start_pos = env.data.xpos[env.robot.body_id].copy()
        
        # 1. 仅在回合开始时规划一次 A* 路径
        self.waypoints = self.planner.plan(start_pos, env.target_pos, active_obstacles)
        
        # =======================================================
        # [终极设计] 预计算沿着折线路径的累计弧长 (1D坐标系统 S)
        # =======================================================
        self.wp_cum_dists =[0.0]
        if len(self.waypoints) > 1:
            for i in range(1, len(self.waypoints)):
                dist = np.linalg.norm(self.waypoints[i] - self.waypoints[i-1])
                self.wp_cum_dists.append(self.wp_cum_dists[-1] + dist)
        self.total_path_length = self.wp_cum_dists[-1] if len(self.waypoints) > 1 else 0.0

        # 初始化状态记忆 (防倒退与保底的核心)
        self.current_seg_idx = 0        # 当前所在的线段段号
        self.rabbit_s = 0.0             # [幽灵兔] 在 1D 路径上的进度(米)，只增不减！
        self.max_auv_s = 0.0            # [AUV真实进度] 最高水位线，用于计算势能奖励，防止原地刷分
        
        # 3. 初始化前视点与沿路径势能
        self.current_lookahead_pt = self._get_lookahead_point(start_pos, env.target_pos)
        self.smoothed_lookahead_pt = self.current_lookahead_pt.copy()
        self.last_path_potential = self._calc_path_potential(start_pos, env.target_pos)

    def _get_lookahead_point(self, pos, target_pos):
        """
        [终极大一统版] 1D 弧长追踪 + 幽灵兔机制
        无论 AUV 怎么倒退，目标点绝对不后退，且按最低速度稳步前进，绝不瞬移！
        """
        if getattr(self, 'waypoints', None) is None or len(self.waypoints) < 2:
            return target_pos

        # 1. 寻找 AUV 在局部路径线段上的真实投影点，计算真实进度 (S_current)
        search_window = min(self.current_seg_idx + 5, len(self.waypoints) - 1)
        min_dist_to_path = float('inf')
        best_i = self.current_seg_idx
        best_t = 0.0

        for i in range(self.current_seg_idx, search_window):
            A = self.waypoints[i]
            B = self.waypoints[i+1]
            AB = B - A
            len_sq = np.dot(AB, AB)
            if len_sq < 1e-6:
                continue

            AP = pos - A
            # 计算垂足比例 t，锁定在 [0, 1] 也就是线段内部
            t = np.clip(np.dot(AP, AB) / len_sq, 0.0, 1.0)
            proj = A + t * AB

            dist_to_path = np.linalg.norm(pos - proj)
            if dist_to_path < min_dist_to_path:
                min_dist_to_path = dist_to_path
                best_i = i
                best_t = t

        # AUV 目前在 1D 路径上的真实总行驶米数 (S_current)
        segment_len = np.linalg.norm(self.waypoints[best_i+1] - self.waypoints[best_i])
        S_current = self.wp_cum_dists[best_i] + best_t * segment_len

        # 推进线段索引 (防倒车检查)
        if best_i > self.current_seg_idx:
            self.just_reached_waypoint = True
        self.current_seg_idx = best_i
        
        # 记录 AUV 自身的最高真实进度 (用于 RL 奖励，防止卡住时幽灵兔跑了白给分)
        self.max_auv_s = max(self.max_auv_s, S_current)

        # =======================================================
        # 2. [核心找回] 幽灵兔：最高水位线防倒车 + 保底推进
        # =======================================================
        # 假设控制频率 10Hz，每步 0.02米 = 每秒至少推进 0.2米 (可按需调整)
        rabbit_speed_m_per_step = 0.05
        
        # 幽灵兔进度 = Max(幽灵兔自己保底走, AUV推着它走)
        self.rabbit_s = max(self.rabbit_s + rabbit_speed_m_per_step, S_current)
        self.rabbit_s = min(self.rabbit_s, self.total_path_length) # 不能超出总长

        # =======================================================
        # 3. 计算前视点：在幽灵兔前方 1.5 米处挂胡萝卜！
        # =======================================================
        lookahead_distance = 1.5 
        S_target = min(self.rabbit_s + lookahead_distance, self.total_path_length)

        # 4. 把 1D 的米数重新映射回 3D 坐标！
        return self._map_1d_s_to_3d_pos(S_target)

    def _map_1d_s_to_3d_pos(self, S_target):
        """将 1D 路径的长度进度 S 映射回 3D 空间坐标"""
        if S_target <= 0.0: return self.waypoints[0]
        if S_target >= self.total_path_length: return self.waypoints[-1]

        # 找到目标点落在哪条线段 [i, i+1] 之间
        for i in range(len(self.waypoints) - 1):
            if self.wp_cum_dists[i] <= S_target <= self.wp_cum_dists[i+1] + 1e-5:
                seg_start_s = self.wp_cum_dists[i]
                seg_end_s = self.wp_cum_dists[i+1]
                seg_len = seg_end_s - seg_start_s
                
                if seg_len < 1e-6:
                    return self.waypoints[i]
                
                # 线性插值求 3D 点
                t = (S_target - seg_start_s) / seg_len
                A = self.waypoints[i]
                B = self.waypoints[i+1]
                return A + t * (B - A)

        return self.waypoints[-1] # 保底返回终点
    
    def _calc_path_potential(self, pos, target_pos):
        """ 
        沿规划路径计算剩余长度势能。完美匹配 1D 坐标系。
        基于 AUV 的最高物理进度计算，原地停滞不给分，倒退不扣分，只有创纪录才给分。
        """
        if getattr(self, 'waypoints', None) is None or len(self.waypoints) < 2:
            dist = np.linalg.norm(pos - target_pos)
        else:
            # 真实剩余距离 = 路径总长 - AUV 达到的最远进度
            dist = self.total_path_length - getattr(self, 'max_auv_s', 0.0)
            
        # 归一化为平滑势能
        phi_dist = - (dist / self.goals.max_dist) * self.cfg.phi_dist 
        return phi_dist
    
    def _get_desired_posture(self, pos, target, rot_mat):
        body_x = rot_mat[:, 0]
        body_y = rot_mat[:, 1]
        body_z = rot_mat[:, 2]
        world_up = np.array([0.0, 0.0, 1.0])
        
        vec_target = target - pos
        dist = np.linalg.norm(vec_target)
        desired_x = vec_target / (dist + 1e-6)
        
        if abs(desired_x[2]) > 0.99:
            desired_y = np.array([0.0, 1.0, 0.0])
        else:
            desired_y = np.cross(world_up, desired_x)
            desired_y = desired_y / np.linalg.norm(desired_y)
            
        desired_z = np.cross(desired_x, desired_y)
        
        align_cos = np.dot(body_x, desired_x)  
        up_cos = np.dot(body_z, desired_z)     
        error_y_roll = 1.0 - abs(body_y[2])
        
        return dist, align_cos, up_cos, error_y_roll

    def compute_reward(self, env, action, obs):
        raw = env.sensors.get_raw_data()
        body_id = env.model.body('yuyuan').id 
        rot_mat = env.data.xmat[body_id].reshape(3, 3)
        pos = env.data.xpos[body_id].copy()

        # 1. 姿态跟踪基于局部前视点 (引导拐弯)
        _, align_cos, up_cos, error_y_roll = self._get_desired_posture(pos, self.current_lookahead_pt, rot_mat)

        # 2. [核心修改] 接入“沿路径势能”，告别直线距离陷阱！
        current_potential = self._calc_path_potential(pos, env.target_pos)
        
        # 势能差：只要顺着 A* 路径走进度增加了，就会得正分
        reward_shaping = (current_potential - self.last_path_potential) * 10.0 
        # 吃到路点给个大奖
        if getattr(self, 'just_reached_waypoint', False):
            reward_shaping += 5.0  
            
        reward_align = 0.5 * (align_cos + 1.0) * self.cfg.w_align_err 
        reward_roll = 0.5 * (up_cos + 1.0) * self.cfg.w_roll_err

        # ----------------------------------------------------
        # 3. 弱化版声呐安全 (Soft Obstacle Penalty)
        # ----------------------------------------------------
        sonar_dists = raw.get('sonar', np.ones(15) * 12.0)
        min_sonar_dist = np.min(sonar_dists)
        
        reward_obstacle_penalty = 0.0
        self.current_is_collision = False

        # [核心调整] 因为碰撞不终止回合，这里改为持续性的步进轻微惩罚
        # 不再一刀切给 w_collision，而是越近惩罚稍大一点，但上限被锁死
        if min_sonar_dist < self.safety.critical_distance:
            self.current_is_collision = True
            # 建议将 config 中的 w_collision 改名为 w_collision_step，值设在 1.0 ~ 5.0 左右
            step_penalty = getattr(self.cfg, 'w_collision_step', 10.0) 
            reward_obstacle_penalty = step_penalty 

        # 4. 动力学约束
        local_vel = raw.get('dvl', np.zeros(3))
        v_sway, v_heave = local_vel[1], local_vel[2]
        cost_sway_heave = self.cfg.w_sway_vel * (v_sway**2 + v_heave**2)

        gyro = raw.get('gyro', np.zeros(3))
        cost_energy = 0.05 * self.cfg.w_energy * np.sum(np.square(gyro))
        cost_action = 0.05 * self.cfg.w_accel * np.sum(np.square(action))
        
        cost_smooth = 0.0
        if self.last_action is not None:
            cost_smooth = self.cfg.w_delta_accel * np.sum(np.square(action - self.last_action))

        # ----------------------------------------------------
        # 5. 成功判定 (只看距离，不管碰撞)
        # ----------------------------------------------------
        reward_success = 0.0
        reward_final_bonus = 0.0  
        is_success = False
        
        dist_to_final = np.linalg.norm(pos - env.target_pos)
        in_zone = dist_to_final < self.goals.success_dist

        # [核心修复] 只要进圈就是成功，哪怕是贴着墙进圈。这才能贴合“弱化避障”的要求
        if in_zone: 
            is_success = True
            reward_success = self.cfg.success 
            reward_final_bonus = getattr(self.cfg, 'w_final_bonus', 500.0)
            time_penalty_applied = 0.0
        else:
            time_penalty_applied = self.cfg.time_penalty

        bonus_y_roll = error_y_roll * self.cfg.bonus_roll 

        # 6. 总分结算
        total_reward = (
            reward_shaping +      
            reward_align +        
            reward_roll +         
            reward_success +      
            reward_final_bonus -
            reward_obstacle_penalty - 
            cost_sway_heave -     
            cost_energy -         
            cost_action -         
            cost_smooth +         
            bonus_y_roll -
            time_penalty_applied  
        )

        self.last_path_potential = current_potential #[注意这里变量名改了]
        self.last_action = action.copy()

        info = {
            "rew/shaping": reward_shaping,
            "rew/align": reward_align,
            "rew/obstacle_penalty": -reward_obstacle_penalty, 
            "rew/cost_sway": -cost_sway_heave,
            "state/dist_to_final": dist_to_final,
            "is_success": float(is_success),
            "is_collision": float(self.current_is_collision)     
        }
        
        return total_reward, is_success, info
    
    def is_done(self, env, current_step, max_steps):
        body_id = env.model.body('yuyuan').id 
        pos = env.data.xpos[body_id].copy()
        target = env.target_pos
        
        dist = np.linalg.norm(pos - target)
        if dist < self.goals.success_dist:
            return True, "success"
            
        if current_step >= max_steps:
            return True, "timeout"
        
        return False, None

    def get_obs(self, env):
        # 此处逻辑基本正确，无需大改
        # 你的观测网络能够看到 local lookahead pt，这是很好的设计
        raw = env.sensors.get_raw_data()
        pos_world = env.data.xpos[env.robot.body_id]
        rot_mat = env.data.xmat[env.robot.body_id].reshape(3, 3)

        # 1. 获取原始的跳跃前视点
        raw_lookahead_pt = self._get_lookahead_point(pos_world, env.target_pos)
        
        # 2. [核心新增] 指数移动平均 (EMA) 滤波
        # alpha 越小越丝滑，越大越敏感。0.1~0.2 是很好的平滑系数
        alpha = 0.15 
        self.smoothed_lookahead_pt = (1.0 - alpha) * self.smoothed_lookahead_pt + alpha * raw_lookahead_pt
        
        # 3. [重要] 网络观测和朝向计算，全部使用平滑后的点！
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

        obs = np.concatenate([
            obs_pos, obs_vel, obs_gyro, obs_quat, gravity_body,
            obs_depth, obs_sonar, obs_alt, obs_accel
        ]).astype(np.float32) 
        
        return obs