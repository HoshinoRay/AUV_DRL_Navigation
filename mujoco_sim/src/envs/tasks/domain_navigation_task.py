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
        self.planner = AStarPlanner(resolution=0.5, safe_margin=1.0)
        self.waypoints =[]
        self.current_lookahead_pt = None
        self.last_action = None
        self.last_global_potential = None  # [修复] 改为全局势能
        self.just_reached_waypoint = False # [修复] 初始化标志位
        
    def get_obs_dim(self):
        return self.obs_dim

    def reset(self, env):
        self.last_action = None
        self.current_is_collision = False
        self.just_reached_waypoint = False
        # [核心新增] 重置路径跟踪索引，防止继承上一回合的进度
        self.closest_wp_idx = 0 
        self.last_reached_idx = -1
        env.target_pos = getattr(env, 'fixed_target_pos', np.array([18.0, 0.0, 10.0]))
        
        active_obstacles = env.scene_builder.get_active_obstacles() 
        start_pos = env.data.xpos[env.robot.body_id].copy()
        self.waypoints = self.planner.plan(start_pos, env.target_pos, active_obstacles)
        
        # 获取第一个前视点
        self.current_lookahead_pt = self._get_lookahead_point(start_pos, env.target_pos)
        
        # [修复] 势能初始化必须基于全局终点，防止路点切换带来的数值突变断层
        self.last_global_potential = self._calc_grand_potential(start_pos, env.target_pos)

    def _get_lookahead_point(self, pos, target_pos):
        """
        获取前视点：寻找路径上距离当前位置最近的点，并向前延伸一段距离
        pos: 机器人当前 3D 坐标
        target_pos: 最终目标点坐标 (用于路径为空时的备选)
        """
        # 1. 基础安全检查：如果没有路径，直接返回终点
        if self.waypoints is None or len(self.waypoints) == 0:
            return target_pos

        lookahead_dist = 3.0  # 前视距离（米）
        
        # 2. 核心逻辑：找到距离当前位置最近的路点的索引
        # 我们通过 hasattr 确保在 reset 时已经初始化了索引，如果没有则设为 0
        if not hasattr(self, 'closest_wp_idx'):
            self.closest_wp_idx = 0
            
        min_dist = float('inf')
        best_idx = self.closest_wp_idx
        
        # 只在当前点及其之后的点中寻找最近点，强制不走回头路
        # 搜索范围限制在当前索引往后 20 个点，提高效率并防止索引越界
        search_window = min(len(self.waypoints), self.closest_wp_idx + 20)
        
        for i in range(self.closest_wp_idx, search_window):
            dist = np.linalg.norm(self.waypoints[i] - pos)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        # 更新类属性，记录当前已经走到了哪一个点
        self.closest_wp_idx = best_idx 

        # 3. 判断是否“吞噬”了新的路点 (用于奖励机制)
        self.just_reached_waypoint = False
        if min_dist < 1.5: # 距离小于 1.5 米认为到达了该点
            # 只有当索引确实增加时，才触发奖励，防止原地打转刷分
            if not hasattr(self, 'last_reached_idx'):
                self.last_reached_idx = -1
            
            if best_idx > self.last_reached_idx:
                self.just_reached_waypoint = True
                self.last_reached_idx = best_idx

        # 4. 寻找前视点 (Pure Pursuit)
        # 从最近点开始往后找，直到找到一个距离机器人超过 lookahead_dist 的点
        lookahead_pt = self.waypoints[-1] # 默认是路径最后一个点
        for i in range(self.closest_wp_idx, len(self.waypoints)):
            dist_to_robot = np.linalg.norm(self.waypoints[i] - pos)
            if dist_to_robot >= lookahead_dist:
                lookahead_pt = self.waypoints[i]
                break
                
        return lookahead_pt

    def _calc_grand_potential(self, pos, target_pos):
        """ [核心修复] 势能必须基于静止的全局终点，而非移动的前视点 """
        dist = np.linalg.norm(pos - target_pos)
        # 归一化到最大距离，保持奖励平滑
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

        # 2. 距离势能基于全局终点 (提供恒定向前的动力，杜绝突变)
        current_potential = self._calc_grand_potential(pos, env.target_pos)
        reward_shaping = (current_potential - self.last_global_potential) * 10.0 
        
        # [修复] 吞噬路点的奖励现在可以正常触发了
        if self.just_reached_waypoint:
            reward_shaping += 33.0  
            
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

        self.last_global_potential = current_potential
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

        self.current_lookahead_pt = self._get_lookahead_point(pos_world, env.target_pos)
        target_vec_world = self.current_lookahead_pt - pos_world
        
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