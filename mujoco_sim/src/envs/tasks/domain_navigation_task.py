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
        
        # [核心设计] 严格单调递增的路径索引
        self.current_wp_idx = 0 
        self.lookahead_wp_idx = 0.0  # [核心新增] 前视点最高水位线，保证前视点永不后退
        self.just_reached_waypoint = False 
        
        env.target_pos = getattr(env, 'fixed_target_pos', np.array([18.0, 0.0, 10.0]))
        
        active_obstacles = env.scene_builder.get_active_obstacles() 
        start_pos = env.data.xpos[env.robot.body_id].copy()
        
        # 1. 仅在回合开始时规划一次 A* 路径
        self.waypoints = self.planner.plan(start_pos, env.target_pos, active_obstacles)
        
        # 2.[核心设计] 预计算：沿 A* 路径每个点到达终点的“真实剩余累计距离”
        self.path_lengths = np.zeros(len(self.waypoints))
        if len(self.waypoints) > 1:
            # 从后往前倒推，算出每个点沿着线走到终点的距离
            for i in range(len(self.waypoints)-2, -1, -1):
                self.path_lengths[i] = self.path_lengths[i+1] + np.linalg.norm(self.waypoints[i] - self.waypoints[i+1])
                
        # 3. 初始化前视点与沿路径势能
        self.current_lookahead_pt = self._get_lookahead_point(start_pos, env.target_pos)
        # [核心新增] 初始化平滑前视点
        self.smoothed_lookahead_pt = self.current_lookahead_pt.copy()
        self.last_path_potential = self._calc_path_potential(start_pos, env.target_pos)

    def _get_lookahead_point(self, pos, target_pos):
        """[终极优化版] 虚拟目标跟踪 (Virtual Target Tracking)
        包含动态锚点跃迁（防倒车）与幽灵兔保底牵引（防卡死）
        """
        if getattr(self, 'waypoints', None) is None or len(self.waypoints) == 0:
            return target_pos
            
        if getattr(self, 'current_wp_idx', 0) >= len(self.waypoints) - 1:
            return target_pos

        self.just_reached_waypoint = False
        
        # =======================================================
        # 机制 1: 动态锚点跃迁 (滑动投影，彻底杜绝冲过头倒车)
        # =======================================================
        # 在当前进度前方找一段窗口 (往后看 60 个点，A* 分辨率 0.05m 的话就是 3.0 米)
        search_window = min(self.current_wp_idx + 60, len(self.waypoints))
        
        min_dist = float('inf')
        closest_idx = self.current_wp_idx
        
        # 找到这 3 米路径里，离当前 AUV 真实位置最近的点
        for i in range(self.current_wp_idx, search_window):
            d = np.linalg.norm(self.waypoints[i] - pos)
            if d < min_dist:
                min_dist = d
                closest_idx = i
                
        # 如果最近点在前方，说明 AUV 已经“走到了那里”（可能是正道，也可能是冲过头偏离了）
        # 强制把当前锚点跃迁过去，绝不回头！
        if closest_idx > self.current_wp_idx:
            self.current_wp_idx = closest_idx
            self.just_reached_waypoint = True
            
        # 如果离锚点已经很近了（<1.0m），为了防卡死也强制 +1
        if min_dist < 1.0 and self.current_wp_idx < len(self.waypoints) - 1:
            self.current_wp_idx += 1
            self.just_reached_waypoint = True

        # 如果锚点到头了，直接返回终点
        if self.current_wp_idx >= len(self.waypoints) - 1:
            self.current_wp_idx = len(self.waypoints) - 1
            return target_pos

        # =======================================================
        # 机制 2: 幽灵兔保底牵引 (防原地卡死等待)
        # =======================================================
        lookahead_dist = 2.0  # 正常视距 2.0 米
        found_idx = self.current_wp_idx
        
        # 寻找距离大于 2.0 米的纯追踪目标点
        for i in range(self.current_wp_idx, len(self.waypoints)):
            if np.linalg.norm(self.waypoints[i] - pos) >= lookahead_dist:
                found_idx = i
                break
                
        #[核心修改] 幽灵兔保底速度！
        # 假设控制频率 10Hz，每步 +0.4 相当于每秒 +4个点(0.2米/秒 的保底逃逸速度)
        # 如果发现兔子走得太快脱离了视野，可以把 rabbit_speed_idx 调小一点 (比如 0.2)
        rabbit_speed_idx = 0.2
        
        # 更新最高水位线：
        # 它等于 (自身加上保底速度) 与 (纯追踪找出的前方锚点) 之间的最大值！
        self.lookahead_wp_idx = max(self.lookahead_wp_idx + rabbit_speed_idx, float(found_idx))
        
        # 限制不能超过路径数组的总长度
        final_idx = min(int(self.lookahead_wp_idx), len(self.waypoints) - 1)

        return self.waypoints[final_idx]
    def _calc_path_potential(self, pos, target_pos):
        """ 
        沿规划路径计算剩余长度势能。
        完美解决 RL 因为绕弯直线距离变远而自我惩罚打转的问题。
        """
        # 如果没有路径，或者点吃完了，退化为全局直线距离
        if getattr(self, 'current_wp_idx', 0) >= len(self.waypoints):
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
            step_penalty = getattr(self.cfg, 'w_collision_step', 2.0) 
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