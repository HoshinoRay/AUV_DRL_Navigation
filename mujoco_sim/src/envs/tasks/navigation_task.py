import numpy as np
from .base_task import BaseTask

class NavigationTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = self.config.reward_weights
        self.goals = self.config.goals
        self.obs_dim = 36
        # 状态缓存
        self.last_action = None
        self.last_potential = None  # 记录上一帧的势能

    def get_obs_dim(self):
        return self.obs_dim

    def reset(self, env):
        self.last_action = None
        env.target_pos = getattr(env, 'fixed_target_pos', np.array([15, 0, 10])) #调用+保护性赋值
        
        # 重置时，计算初始势能，避免第一帧产生巨大的差分奖励
        self.last_potential = self._calc_grand_potential(env)

    def _quat_to_euler(self, quat):
        """四元数转欧拉角 (Roll, Pitch, Yaw)"""
        w, x, y, z = quat
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        # 使用 clip 截断可能的计算误差，保证 arcsin 输入合法，避免 NaN
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def _calc_grand_potential(self, env):
        """
        [修改后] PBRS 势能函数
        只保留距离势能。完全剥离姿态约束，防止积分抵消。
        """
        pos = env.data.xpos[env.robot.body_id]
        target = env.target_pos
        dist = np.linalg.norm(pos - target)
        
        # 只保留距离势能！
        # 形式：-Distance。越近，势能越大(越接近0)。
        phi_dist = - (dist / self.goals.max_dist) * self.cfg.phi_dist
        
        return phi_dist
    
    def _get_desired_posture(self, pos, target, rot_mat):
        """ 动态计算 3D 空间下“包含 Pitch 且零 Roll”的完美姿态 """
        
        # 修复 3：正确提取旋转矩阵中的局部 X, Y, Z 轴向量（按列切片）
        body_x = rot_mat[:, 0]
        body_y = rot_mat[:, 1]
        body_z = rot_mat[:, 2]
        world_up = np.array([0.0, 0.0, 1.0])
        
        # 1. 期望的 X 轴 (精准指向目标)
        vec_target = target - pos
        dist = np.linalg.norm(vec_target)
        desired_x = vec_target / (dist + 1e-6)
        
        # 2. 期望的 Y 轴 (强制水平，即零 Roll 的核心)
        # 修复 1：只判断 Z 轴分量 desired_x[2] 是否接近 1 或 -1
        if abs(desired_x[2]) > 0.99:
            # 修复 2：给一个备用的 Y 轴方向，避免空数组报错
            desired_y = np.array([0.0, 1.0, 0.0])
        else:
            desired_y = np.cross(world_up, desired_x)
            desired_y = desired_y / np.linalg.norm(desired_y)
            
        # 3. 期望的 Z 轴 (垂直于期望的 X 和 Y)
        desired_z = np.cross(desired_x, desired_y)
        
        # 4. 计算当前的对齐度 (-1 到 1，1 为最完美)
        # 经过修复后，body_x 和 desired_x 都是一维向量，点乘将正确返回标量
        align_cos = np.dot(body_x, desired_x)  
        up_cos = np.dot(body_z, desired_z)     
    
        error_y_roll = 1.0 - abs(body_y[2])
        
        # 注意：这里多返回了一个 error_y_roll
        return dist, align_cos, up_cos, error_y_roll


    def compute_reward(self, env, action, obs):
        raw = env.sensors.get_raw_data()
        body_id = env.model.body('yuyuan').id 
        rot_mat = env.data.xmat[body_id].reshape(3, 3)
        pos = env.data.xpos[body_id].copy()
        target = env.target_pos

        # 获取完美姿态的对齐指标
        dist, align_cos, up_cos, error_y_roll= self._get_desired_posture(pos, target, rot_mat)

        # ----------------------------------------------------
        # 1. 引导奖励 (The Carrots) - 鼓励探索
        # ----------------------------------------------------
        # A. 距离势能差分 (靠近目标的动力)
        current_potential = self._calc_grand_potential(env)
        reward_shaping = current_potential - self.last_potential
        # 放大距离奖励，让 Agent 尝到甜头
        reward_shaping *= 2.0 

        # B. 姿态追踪奖励 (只要你在看目标，你就得分！)
        # 将区间从 映射到 的平滑曲线
        # Agent 会拼命为了吃这口分而主动对齐目标
        reward_align = 0.5 * (align_cos + 1.0) * self.cfg.w_align_err 
        reward_roll = 0.5 * (up_cos + 1.0) * self.cfg.w_roll_err

        # ----------------------------------------------------
        # 2. 终点区域逻辑 (解决“冲刺翻船”问题)
        # ----------------------------------------------------
        reward_success = 0.0
        reward_final_bonus = 0.0  # 新增：结算时的姿态额外奖金
        is_success = False
        
        in_zone = dist < self.goals.success_dist

        if in_zone:
            # === 核心修改：只要进圈，立刻算赢 ===
            is_success = True
            
            # 1. 基础大奖 (拿到低保)
            reward_success = self.cfg.success 
            
            # 2. 姿态 Bonus (线性奖励，只加分不扣分)
            # 逻辑：将 cos 从 [-1, 1] 映射到 [0, 1]
            # 最完美(1.0)时拿满分，完全反向(-1.0)时拿0分，绝不倒扣
            align_score = (align_cos + 1.0) / 2.0
            up_score = (up_cos + 1.0) / 2.0
            
            # 读取配置中的 bonus 权重，如果没有配置默认给 500 分
            w_bonus = getattr(self.cfg, 'w_final_bonus', 500.0) 
            
            # 计算最终姿态奖金
            reward_final_bonus = w_bonus * (align_score + up_score)
            
            # 3. 成功时不扣时间分
            time_penalty_applied = 0.0
            
            # (清理掉旧的 hovering 逻辑，防止它产生任何负分)
            reward_hovering = 0.0 
        else:
            # 还在路上，正常扣时间
            time_penalty_applied = self.cfg.time_penalty
            reward_hovering = 0.0 # 保持为0，不在路上搞复杂操作
        # Roll 水平奖励 
        bonus_y_roll = error_y_roll * self.cfg.bonus_roll  
        # ----------------------------------------------------
        # 3. 轻微的成本约束 (保证动作合理性)
        # ----------------------------------------------------
        gyro = raw.get('gyro', np.zeros(3))
        # 降低约束权重，别把 Agent 吓得不敢动
        cost_energy = 0.05 * self.cfg.w_energy * np.sum(np.square(gyro)) 
        cost_action = 0.05 * self.cfg.w_accel * np.sum(np.square(action))
        
        # 修复 Bug：这里是惩罚，所以一会要减去
        cost_smooth = 0.0
        if self.last_action is not None:
            cost_smooth = self.cfg.w_delta_accel * np.sum(np.square(action - self.last_action))

        # ----------------------------------------------------
        # 4. 汇总总分
        # ----------------------------------------------------
        total_reward = (
            reward_shaping +      # 靠近目标的动力
            reward_align +        # 持续的瞄准加分
            reward_roll +         # 持续的平稳加分
            reward_hovering +     # 终点区域的“引力”和姿态矫正加分
            reward_success +      # 最终大奖
            reward_final_bonus -
            cost_energy -         # 极轻微的能量惩罚
            cost_action -         # 极轻微的动作惩罚
            cost_smooth +         # 极轻微的抖动惩罚
            bonus_y_roll -
            time_penalty_applied  # 每一秒的损耗 (进入终点区域后暂停)
        )

        # 状态更新
        self.last_potential = current_potential
        self.last_action = action.copy()

        # Tensorboard 日志 (修复了变量覆盖导致的假数据 Bug)
        info = {
            "rew/shaping": reward_shaping,
            "rew/align": reward_align,
            "rew/roll": reward_roll,
            "rew/hovering": reward_hovering,
            "state/real_align_cos": align_cos, 
            "state/real_up_cos": up_cos,       
            "state/dist": dist,
            "cost/y_roll_penalty": -bonus_y_roll, # 记录为负数方便在曲线图里看
            "state/error_y_roll": error_y_roll,  # 新增监控：越接近0说明身体越平
            "is_success": float(is_success)
        }
        
        return total_reward, is_success, info
    
    def _check_success_condition(self, env):
        """ 供 is_done 调用的纯逻辑判定 """
        # 修复 4：必须指定 body_id，否则取到的是全局所有刚体的矩阵数据
        body_id = env.model.body('yuyuan').id 
        rot_mat = env.data.xmat[body_id].reshape(3, 3)
        pos = env.data.xpos[body_id].copy()
        target = env.target_pos
        
        dist, align_cos, up_cos, _= self._get_desired_posture(pos, target, rot_mat)
        
        is_success = dist < self.goals.success_dist
        return is_success, dist, align_cos, up_cos

    def is_done(self, env, current_step, max_steps):
        is_success, dist, _, _ = self._check_success_condition(env)
        
        if is_success:
            return True, "success"
            
        if current_step >= max_steps:
            return True, "timeout"
        
        return False, None

    def get_obs(self, env):
        raw = env.sensors.get_raw_data()
        pos_world = env.data.xpos[env.robot.body_id]
        target_vec_world = env.target_pos - pos_world
        rot_mat = env.data.xmat[env.robot.body_id].reshape(3, 3)
        target_vec_body = rot_mat.T @ target_vec_world
        # 2. [新增] 重力在机体坐标系的向量 (姿态核心)
        # 这能让 Agent 直接知道"下"在哪里，不用猜四元数
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
        ]).astype(np.float32) #拼接+强制类型转换
        return obs

   