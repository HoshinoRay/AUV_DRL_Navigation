import numpy as np
from .base_task import BaseTask
from src.utils.astar_planner import AStarPlanner

class DomainNavigationTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = self.config.reward_weights
        self.goals = self.config.goals
        
        # 1. 严格解析并绑定所有配置（仅在此处使用一次 getattr 提取默认值）
        self.w_cte = getattr(self.cfg, 'w_cte', 15.0)
        self.w_collision_step = getattr(self.cfg, 'w_collision_step', 25.0)
        self.w_final_bonus = getattr(self.cfg, 'w_final_bonus', 500.0)
        
        # 安全模块初始化
        safety_cfg = getattr(self.config, 'safety', None)
        self.warning_distance = getattr(safety_cfg, 'warning_distance', 4.0) if safety_cfg else 4.0
        self.critical_distance = getattr(safety_cfg, 'critical_distance', 0.4) if safety_cfg else 0.4

        # 2. 显式声明所有内部状态变量（消除业务逻辑中的 Attribute Error 隐患）
        self.obs_dim = 36 
        self.planner = AStarPlanner(resolution=0.05, safe_margin=1.7)
        
        self.waypoints =[]           # 严格初始化为空列表
        self.path_lengths = np.array([])
        
        # 状态机变量
        self.current_wp_idx = 0 
        self.lookahead_wp_idx = 0.0  
        self.just_reached_waypoint = False 
        self.current_is_collision = False
        
        # 历史记录变量
        self.last_action = None
        self.last_path_potential = 0.0 
        self.smoothed_lookahead_pt = np.zeros(3)  # 初始化明确的 shape
        self.current_lookahead_pt = np.zeros(3)

    def update_navigation_state(self, env):
        """
        [Command] 状态更新机：每个物理 step 结束后严格调用且仅调用一次！
        负责计算路点进度、幽灵兔更新、平滑滤波。
        """
        if len(self.waypoints) == 0:
            return

        body_id = env.model.body('yuyuan').id 
        pos = env.data.xpos[body_id].copy()
        
        # 1. 重置单步事件标志位
        self.just_reached_waypoint = False
        
        # 2. 计算动态锚点跃迁
        search_window = min(self.current_wp_idx + 60, len(self.waypoints))
        min_dist = float('inf')
        closest_idx = self.current_wp_idx
        
        for i in range(self.current_wp_idx, search_window):
            d = np.linalg.norm(self.waypoints[i] - pos)
            if d < min_dist:
                min_dist = d
                closest_idx = i
                
        if closest_idx > self.current_wp_idx:
            self.current_wp_idx = closest_idx
            self.just_reached_waypoint = True
            
        if min_dist < 1.0 and self.current_wp_idx < len(self.waypoints) - 1:
            self.current_wp_idx += 1
            self.just_reached_waypoint = True

        # 3. 计算幽灵兔保底前视点
        if self.current_wp_idx >= len(self.waypoints) - 1:
            self.current_wp_idx = len(self.waypoints) - 1
            raw_lookahead_pt = env.target_pos
        else:
            lookahead_dist = 1.2
            found_idx = self.current_wp_idx
            for i in range(self.current_wp_idx, len(self.waypoints)):
                if np.linalg.norm(self.waypoints[i] - pos) >= lookahead_dist:
                    found_idx = i
                    break
                    
            rabbit_speed_idx = 0.3
            self.lookahead_wp_idx = max(self.lookahead_wp_idx + rabbit_speed_idx, float(found_idx))
            final_idx = min(int(self.lookahead_wp_idx), len(self.waypoints) - 1)
            raw_lookahead_pt = self.waypoints[final_idx]

        # 4. EMA 滤波更新 (禁止在 get_obs 中执行！)
        alpha = 0.15 
        self.smoothed_lookahead_pt = (1.0 - alpha) * self.smoothed_lookahead_pt + alpha * raw_lookahead_pt
        self.current_lookahead_pt = raw_lookahead_pt

    def get_obs_dim(self):
        return self.obs_dim

    def reset(self, env):
        self.last_action = None
        self.current_is_collision = False
        self.current_wp_idx = 0 
        self.lookahead_wp_idx = 0.0  
        self.just_reached_waypoint = False 
        
        env.target_pos = getattr(env, 'fixed_target_pos', np.array([18.0, 0.0, 10.0]))
        active_obstacles = env.scene_builder.get_active_obstacles() 
        start_pos = env.data.xpos[env.robot.body_id].copy()
        
        # 1. 规划路径及防崩兜底
        self.waypoints = self.planner.plan(start_pos, env.target_pos, active_obstacles)
        if not self.waypoints or len(self.waypoints) == 0:
            # 工业级兜底：A*失败时降级为全局直线
            self.waypoints = [start_pos, env.target_pos]
            
        # 2. 计算沿径势能基带
        self.path_lengths = np.zeros(len(self.waypoints))
        if len(self.waypoints) > 1:
            for i in range(len(self.waypoints)-2, -1, -1):
                self.path_lengths[i] = self.path_lengths[i+1] + np.linalg.norm(self.waypoints[i] - self.waypoints[i+1])
                
        # 3. 初始对齐：计算第一帧的前视点
        # 调用一次更新逻辑，但不触发 EMA 的历史滞后！
        self.update_navigation_state(env)
        
        # [核心修复] 强制首帧的 EMA 点等于当前计算点，彻底消除初值的漂移不同步
        self.smoothed_lookahead_pt = self.current_lookahead_pt.copy()
        
        # 4. 初始化第一帧势能
        self.last_path_potential = self._calc_path_potential(start_pos, env.target_pos)
        
    def _calc_cross_track_error(self, pos):
        """计算 AUV 到当前 A* 路径线段的垂直距离 (CTE)"""
        if len(self.waypoints) < 2:
            return 0.0
            
        # 找到当前锚点和下一个点构成的线段
        idx1 = min(self.current_wp_idx, len(self.waypoints) - 2)
        idx2 = idx1 + 1
        
        A = self.waypoints[idx1]
        B = self.waypoints[idx2]
        
        # 点到线段的最短距离向量计算
        AB = B - A
        AP = pos - A
        # 投影比例
        t = np.dot(AP, AB) / (np.dot(AB, AB) + 1e-6)
        t = np.clip(t, 0.0, 1.0) # 限制在线段端点内
        
        # 投影点
        projection = A + t * AB
        cte = np.linalg.norm(pos - projection)
        return cte

    def _calc_path_potential(self, pos, target_pos):
        """ 
        沿规划路径计算剩余长度势能。
        完美解决 RL 因为绕弯直线距离变远而自我惩罚打转的问题。
        """
        # 如果没有路径，或者点吃完了，退化为全局直线距离
        if self.current_wp_idx >= len(self.waypoints):
            dist = np.linalg.norm(pos - target_pos)
        else:
            # 真实剩余距离 = (机器人到当前前沿路点的距离) + (该路点到终点的累计规划路径长度)
            dist_to_frontier = np.linalg.norm(self.waypoints[self.current_wp_idx] - pos)
            dist = dist_to_frontier + self.path_lengths[self.current_wp_idx]
            
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

        # ----------------------------------------------------
        # 1. 姿态跟踪与路径势能 (彻底移除状态突变隐患)
        # ----------------------------------------------------
        # 直接读取已冻结的安全状态 self.smoothed_lookahead_pt
        _, align_cos, up_cos, error_y_roll = self._get_desired_posture(pos, self.smoothed_lookahead_pt, rot_mat)
        
        current_potential = self._calc_path_potential(pos, env.target_pos)
        reward_shaping = (current_potential - self.last_path_potential) * 10.0 
        
        # 直接读取已确定的布尔标志，杜绝 getattr 掩盖错误
        if self.just_reached_waypoint:
            reward_shaping += 5.0  

        # ----------------------------------------------------
        # 2. 横向误差 CTE 惩罚 (引入 Huber-like Loss 机制)
        # ----------------------------------------------------
        cte = self._calc_cross_track_error(pos)
        
        # [核心优化] 工业级防超调处理：
        # 对于欠驱动 AUV，拐弯时出现 >1m 的偏航是物理必然现象。
        # 纯二次方惩罚会导致大拐角处惩罚爆炸，使模型学到“原地停滞不前”的次优策略。
        # 解决方案：小误差二次方惩罚(紧贴轨道)，大误差线性惩罚(限制上限)。
        if cte < 1.0:
            cost_cte = self.w_cte * (cte ** 2)
        else:
            cost_cte = self.w_cte * (2.0 * cte - 1.0) # 保证在 cte=1.0 处函数连续且导数连续

        # ----------------------------------------------------
        # 3. 姿态朝向得分
        # ----------------------------------------------------
        reward_align = 0.5 * (align_cos + 1.0) * self.cfg.w_align_err 
        reward_roll = 0.5 * (up_cos + 1.0) * self.cfg.w_roll_err

        # ----------------------------------------------------
        # 4. 指数级声呐安全力场
        # ----------------------------------------------------
        sonar_dists = raw.get('sonar', np.ones(15) * 12.0)
        min_sonar_dist = np.min(sonar_dists)
        
        reward_obstacle_penalty = 0.0
        self.current_is_collision = False

        if min_sonar_dist < self.warning_distance:
            if min_sonar_dist < self.critical_distance:
                self.current_is_collision = True
                # 撞死区：给出硬惩罚（注：如果环境不终止，网络需容忍此处的连续扣分）
                reward_obstacle_penalty = self.w_collision_step 
            else:
                # 警告区：越靠近墙壁，惩罚呈抛物线激增
                scale = (self.warning_distance - min_sonar_dist) / (self.warning_distance - self.critical_distance)
                reward_obstacle_penalty = 5.0 * (scale ** 2)

        # ----------------------------------------------------
        # 5. 动力学约束 (保持原始逻辑，写法更加工整)
        # ----------------------------------------------------
        local_vel = raw.get('dvl', np.zeros(3))
        v_surge, v_sway, v_heave = local_vel[0], local_vel[1], local_vel[2]
        
        cost_sway_heave = self.cfg.w_sway_vel * (v_sway**2 + v_heave**2)

        gyro = raw.get('gyro', np.zeros(3))
        cost_energy = 0.05 * self.cfg.w_energy * np.sum(np.square(gyro))
        cost_action = 0.05 * self.cfg.w_accel * np.sum(np.square(action))

        # 速度限制约束 (防超速暴走)
        v_surge_excess = max(0.0, v_surge - 1.2)
        cost_overspeed = 10.0 * (v_surge_excess ** 2)
        
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
            reward_final_bonus = self.w_final_bonus
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
            cost_smooth -   
            cost_cte -           # [新增] 偏离轨道惩罚
            cost_overspeed +     #[新增] 超速惩罚     
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
            "is_collision": float(self.current_is_collision),    
            "rew/cost_cte": -cost_cte,               # [新增监控] 
            "rew/cost_overspeed": -cost_overspeed,   # [新增监控]
            "state/min_sonar_dist": min_sonar_dist,  # [新增监控] 看到墙壁的距离
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
# 直接使用已经计算好的、冻结的平滑前视点
        # 不再调 _get_lookahead_point()，也不执行 EMA！
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