# src/envs/tasks/base_task.py

from abc import ABC, abstractmethod
import numpy as np

class BaseTask(ABC):
    """
    所有 AUV 任务的抽象基类。
    Env 会调用这些方法来驱动逻辑，而不是将逻辑写死在 Env 里。
    """
    def __init__(self, config):
        """
        config: 来自 Hydra 的 DictConfig (对应 yaml 中的 'task' 部分)
        """
        self.config = config
        self.last_action = None
        self.prev_dist = None

    @abstractmethod
    def reset(self, env):
        """
        每回合开始时调用。
        用于生成新的随机目标、重置内部计数器等。
        """
        pass

    @abstractmethod
    def get_obs(self, env):
        """
        计算并返回 RL 智能体所需的 Observation。
        不同任务需要的观测维度可能不同 (例如避障任务需要声呐)。
        Returns: np.ndarray
        """
        pass

    @abstractmethod
    def get_obs_dim(self) -> int:
        """
        返回 Observation 的维度大小，用于定义 Gym Space。
        """
        pass

    @abstractmethod
    def compute_reward(self, env, action, obs):
        """
        核心奖励函数。
        Returns:
            total_reward (float)
            is_success (bool)
            info (dict): 用于 Tensorboard 记录的详细信息
        """
        pass

    @abstractmethod
    def is_done(self, env, current_step, max_steps):
        """
        判断回合是否结束。
        Returns:
            terminated (bool): 任务结束 (成功或失败)
            reason (str): 结束原因 ("success", "flipped", "timeout" 等)
        """
        pass

    # --- 通用工具函数 ---
    def _get_distance(self, env):
        current_pos = env.data.xpos[env.robot.body_id]
        # 确保 env.target_pos 存在
        if not hasattr(env, 'target_pos'):
            return 0.0
        return np.linalg.norm(current_pos - env.target_pos)

    def _get_body_velocity(self, env):
        """通用：获取机器人坐标系下的速度 (Surge, Sway, Heave)"""
        R_body_to_world = env.data.xmat[env.robot.body_id].reshape(3, 3)
        vel_world = env.data.qvel[0:3]
        vel_body = R_body_to_world.T @ vel_world
        return vel_body