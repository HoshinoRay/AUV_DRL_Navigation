import numpy as np
from .base_task import BaseTask

class AvoidanceTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = self.config.reward_weights
        self.goals = self.config.goals
        
        # 读取安全配置，如果没有配置则给出安全默认值
        self.safety = getattr(self.config, 'safety', None)
        if self.safety is None:
            # Fallback 默认值
            class DummySafety:
                warning_distance = 4.0
                critical_distance = 0.4
            self.safety = DummySafety()

        self.obs_dim = 36 
        
        # 状态缓存
        self.last_action = None
        self.last_potential = None  
        

    def get_obs_dim(self):
        return self.obs_dim

    def reset(self, env):
        self.last_action = None
        env.target_pos = getattr(env, 'fixed_target_pos', np.array([30, 0, 10]))
        self.last_potential = self._calc_grand_potential(env)

    def _calc_grand_potential(self, env):
        """ 距离势能函数 """
        pos = env.data.xpos[env.robot.body_id]
        target = env.target_pos
        dist = np.linalg.norm(pos - target)
        phi_dist = - (dist / self.goals.max_dist) * self.cfg.phi_dist
        return phi_dist
    
    def _get_desired_posture(self, pos, target, rot_mat):
        """ 获取姿态偏差 """
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
        target = env.target_pos

        dist, align_cos, up_cos, error_y_roll = self._get_desired_posture(pos, target, rot_mat)

        # 引导奖励 
        current_potential = self._calc_grand_potential(env)
        reward_shaping = (current_potential - self.last_potential) * 10.0 
        reward_align = 0.7 * (align_cos + 1.0) * self.cfg.w_align_err 
        reward_roll = 0.5 * (up_cos + 1.0) * self.cfg.w_roll_err

        # ----------------------------------------------------
        # 2. 避障斥力势场 (Repulsive Field) - [重构核心]
        # ----------------------------------------------------
        sonar_dists = raw.get('sonar', np.ones(15) * 12.0)
        min_sonar_dist = np.min(sonar_dists)
        
        reward_obstacle_penalty = 0.0
        self.current_is_collision = False

        if min_sonar_dist < self.safety.critical_distance:
            # 撞死
            self.current_is_collision = True
            reward_obstacle_penalty = self.cfg.w_collision 
        else:
            # 危险警告区 - 使用指数衰减函数，距离越近惩罚呈指数上升
            # 这样智能体会提前感受到平滑的“推力”使其转向，而不是突然撞墙
            for d in sonar_dists:
                if d < self.safety.warning_distance:
                    # 例如 d=4时不惩罚，d=0.5时极度惩罚
                    penalty_factor = np.exp(-1.5 * (d - self.safety.critical_distance))
                    reward_obstacle_penalty += self.cfg.w_danger_zone * penalty_factor
        # 终点区域逻辑
        reward_success = 0.0
        reward_final_bonus = 0.0  
        is_success = False
        
        in_zone = dist < self.goals.success_dist

        if in_zone and not self.current_is_collision: # 必须是活着进圈才算赢
            is_success = True
            reward_success = self.cfg.success 
            align_score = (align_cos + 1.0) / 2.0
            up_score = (up_cos + 1.0) / 2.0
            w_bonus = getattr(self.cfg, 'w_final_bonus', 500.0) 
            reward_final_bonus = 0.2 * w_bonus * (align_score + up_score)
            time_penalty_applied = 0.0
        else:
            time_penalty_applied = self.cfg.time_penalty
            
        bonus_y_roll = error_y_roll * self.cfg.bonus_roll  

        # ----------------------------------------------------
        # 4. 轻微的成本约束 
        # ----------------------------------------------------
        gyro = raw.get('gyro', np.zeros(3))
        cost_energy = 0.05 * self.cfg.w_energy * np.sum(np.square(gyro)) 
        cost_action = 0.05 * self.cfg.w_accel * np.sum(np.square(action))
        
        cost_smooth = 0.0
        if self.last_action is not None:
            cost_smooth = self.cfg.w_delta_accel * np.sum(np.square(action - self.last_action))

        total_reward = (
            reward_shaping +      
            reward_align +        
            reward_roll +         
            reward_success +      
            reward_final_bonus -
            reward_obstacle_penalty - 
            cost_energy -         
            cost_action -         
            cost_smooth +         
            bonus_y_roll -
            time_penalty_applied  
        )

        self.last_potential = current_potential
        self.last_action = action.copy()

        info = {
            "rew/shaping": reward_shaping,
            "rew/align": reward_align,
            "rew/obstacle_penalty": -reward_obstacle_penalty, 
            "state/real_align_cos": align_cos, 
            "state/dist": dist,
            "state/min_sonar_dist": min_sonar_dist, 
            "is_success": float(is_success),
            "is_collision": float(self.current_is_collision)     
        }
        
        return total_reward, is_success, info
    
    def is_done(self, env, current_step, max_steps):
        """ 终止条件 """
        # --- 获取全局状态数据 ---
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
        """ 与 NavigationTask 一模一样，你的 obs_sonar 已经在里面了！ """
        raw = env.sensors.get_raw_data()
        pos_world = env.data.xpos[env.robot.body_id]
        target_vec_world = env.target_pos - pos_world
        rot_mat = env.data.xmat[env.robot.body_id].reshape(3, 3)
        target_vec_body = rot_mat.T @ target_vec_world
        gravity_body = rot_mat.T @ np.array([0., 0., -1.]) 
        
        obs_pos = np.clip(target_vec_body / self.goals.max_dist, -1.0, 1.0)
        obs_vel = np.clip(raw['dvl'] / 2.0, -1.0, 1.0)
        obs_gyro = np.clip(raw['gyro'] / 6.0, -1.0, 1.0)
        obs_quat = raw['quat']
        depth = env.WATER_SURFACE_Z - pos_world[2]
        obs_depth = np.array([np.clip(depth / 50.0, 0.0, 1.0)])
        
        # 你的 15根声呐数据在这里已经被归一化到了 [0, 1] 并且塞进神经网络了
        # Agent 会自动将这些维度与 compute_reward 里的惩罚关联起来
        obs_sonar = np.clip(raw.get('sonar', np.zeros(15)) / 12.0, 0.0, 1.0)
        
        obs_alt = np.array([np.clip(raw.get('altitude', 0) / 50.0, 0.0, 1.0)])
        obs_accel = np.clip(raw['accel'] / 9.81, -3.0, 3.0)

        obs = np.concatenate([
            obs_pos, obs_vel, obs_gyro, obs_quat, gravity_body,
            obs_depth, obs_sonar, obs_alt, obs_accel
        ]).astype(np.float32) 
        
        return obs