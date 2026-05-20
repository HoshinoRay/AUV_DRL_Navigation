from abc import ABC, abstractmethod

import numpy as np


class BaseTask(ABC):
    """
    Abstract base class for all AUV task definitions.

    The environment delegates observation construction, reward computation,
    and termination logic to a concrete subclass so that the physics
    container (AUVGymEnv) stays task-agnostic.
    """

    def __init__(self, config):
        self.config = config
        self.last_action = None
        self.prev_dist = None

    @abstractmethod
    def reset(self, env):
        """Called at the start of every episode to set up task-specific state."""

    @abstractmethod
    def get_obs(self, env) -> np.ndarray:
        """Construct and return the RL observation vector."""

    @abstractmethod
    def get_obs_dim(self) -> int:
        """Return the observation dimensionality (used to create the Gym space)."""

    @abstractmethod
    def compute_reward(self, env, action, obs):
        """
        Returns
        -------
        total_reward : float
        is_success   : bool
        info         : dict  — per-component breakdowns for logging
        """

    @abstractmethod
    def is_done(self, env, current_step: int, max_steps: int):
        """
        Returns
        -------
        terminated : bool
        reason     : str  — "success" | "timeout" | None
        """

    # ------------------------------------------------------------------
    # Shared utility methods
    # ------------------------------------------------------------------

    def _get_distance(self, env) -> float:
        current_pos = env.data.xpos[env.robot.body_id]
        if not hasattr(env, 'target_pos'):
            return 0.0
        return float(np.linalg.norm(current_pos - env.target_pos))

    def _get_body_velocity(self, env) -> np.ndarray:
        """Return body-frame linear velocity [u, v, w]."""
        R = env.data.xmat[env.robot.body_id].reshape(3, 3)
        return R.T @ env.data.qvel[0:3]
