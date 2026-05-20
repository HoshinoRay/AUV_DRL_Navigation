import numpy as np

from .base_task import BaseTask


class NavigationTask(BaseTask):
    """
    Stage 1 task: point-to-point navigation without obstacles.

    Reward structure
    ----------------
    - PBRS potential shaping on distance (motivates approach)
    - Alignment reward: heading toward the goal
    - Roll stability reward: penalises non-level attitude
    - Success bonus on reaching the goal sphere
    - Time penalty to encourage efficiency
    - Small energy and action-smoothness costs
    """

    OBS_DIM = 36  # [pos(3), vel(3), gyro(3), quat(4), gravity(3), depth(1), sonar(15), alt(1), accel(3)]

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.reward_weights
        self.goals = config.goals
        self.obs_dim = self.OBS_DIM
        self.last_action = None
        self.last_potential = None

    def get_obs_dim(self):
        return self.obs_dim

    def reset(self, env):
        self.last_action = None
        env.target_pos = getattr(env, 'fixed_target_pos', np.array([15.0, 0.0, 10.0]))
        self.last_potential = self._calc_potential(env)

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _calc_potential(self, env) -> float:
        """PBRS potential: negative normalised distance."""
        pos = env.data.xpos[env.robot.body_id]
        dist = np.linalg.norm(pos - env.target_pos)
        return -(dist / self.goals.max_dist) * self.cfg.phi_dist

    def _get_posture_metrics(self, pos, target, rot_mat):
        """
        Compute alignment and roll metrics relative to the target.

        Returns
        -------
        dist        : float — Euclidean distance to target
        align_cos   : float — cosine similarity of body-X with desired heading
        up_cos      : float — cosine similarity of body-Z with desired up
        error_y_roll: float — roll error (0 = perfectly level)
        """
        body_x = rot_mat[:, 0]
        body_z = rot_mat[:, 2]
        world_up = np.array([0.0, 0.0, 1.0])

        vec_target = target - pos
        dist = np.linalg.norm(vec_target)
        desired_x = vec_target / (dist + 1e-6)

        if abs(desired_x[2]) > 0.99:
            desired_y = np.array([0.0, 1.0, 0.0])
        else:
            desired_y = np.cross(world_up, desired_x)
            desired_y /= np.linalg.norm(desired_y)

        desired_z = np.cross(desired_x, desired_y)

        align_cos    = float(np.dot(body_x, desired_x))
        up_cos       = float(np.dot(body_z, desired_z))
        error_y_roll = 1.0 - abs(rot_mat[:, 1][2])  # body-Y should be horizontal

        return dist, align_cos, up_cos, error_y_roll

    # ------------------------------------------------------------------
    # Task interface
    # ------------------------------------------------------------------

    def compute_reward(self, env, action, obs):
        raw = env.sensors.get_raw_data()
        body_id = env.model.body('yuyuan').id
        rot_mat = env.data.xmat[body_id].reshape(3, 3)
        pos = env.data.xpos[body_id].copy()
        target = env.target_pos

        dist, align_cos, up_cos, error_y_roll = self._get_posture_metrics(pos, target, rot_mat)

        # Potential-based shaping
        current_potential = self._calc_potential(env)
        reward_shaping = (current_potential - self.last_potential) * 2.0

        # Continuous posture rewards (map [-1,1] cos to [0, weight])
        reward_align = 0.5 * (align_cos + 1.0) * self.cfg.w_align_err
        reward_roll  = 0.5 * (up_cos  + 1.0) * self.cfg.w_roll_err

        # Success
        reward_success = 0.0
        reward_final_bonus = 0.0
        is_success = dist < self.goals.success_dist

        if is_success:
            reward_success = self.cfg.success
            align_score = (align_cos + 1.0) / 2.0
            up_score    = (up_cos    + 1.0) / 2.0
            w_bonus = getattr(self.cfg, 'w_final_bonus', 500.0)
            reward_final_bonus = w_bonus * (align_score + up_score)
            time_penalty = 0.0
        else:
            time_penalty = self.cfg.time_penalty

        # Roll level bonus
        bonus_roll = error_y_roll * self.cfg.bonus_roll

        # Costs
        gyro = raw.get('gyro', np.zeros(3))
        cost_energy = 0.05 * self.cfg.w_energy * np.sum(np.square(gyro))
        cost_action = 0.05 * self.cfg.w_accel  * np.sum(np.square(action))
        cost_smooth = 0.0
        if self.last_action is not None:
            cost_smooth = self.cfg.w_delta_accel * np.sum(np.square(action - self.last_action))

        total_reward = (
            reward_shaping
            + reward_align
            + reward_roll
            + reward_success
            + reward_final_bonus
            - cost_energy
            - cost_action
            - cost_smooth
            + bonus_roll
            - time_penalty
        )

        self.last_potential = current_potential
        self.last_action = action.copy()

        info = {
            "rew/shaping":         reward_shaping,
            "rew/align":           reward_align,
            "rew/roll":            reward_roll,
            "state/align_cos":     align_cos,
            "state/up_cos":        up_cos,
            "state/dist":          dist,
            "state/error_y_roll":  error_y_roll,
            "is_success":          float(is_success),
        }
        return total_reward, is_success, info

    def is_done(self, env, current_step, max_steps):
        body_id = env.model.body('yuyuan').id
        pos = env.data.xpos[body_id].copy()
        if np.linalg.norm(pos - env.target_pos) < self.goals.success_dist:
            return True, "success"
        if current_step >= max_steps:
            return True, "timeout"
        return False, None

    def get_obs(self, env):
        raw = env.sensors.get_raw_data()
        pos_world = env.data.xpos[env.robot.body_id]
        rot_mat   = env.data.xmat[env.robot.body_id].reshape(3, 3)

        target_vec_body = rot_mat.T @ (env.target_pos - pos_world)
        gravity_body    = rot_mat.T @ np.array([0.0, 0.0, -1.0])

        depth = env.WATER_SURFACE_Z - pos_world[2]

        obs = np.concatenate([
            np.clip(target_vec_body / self.goals.max_dist, -1.0, 1.0),   # 3
            np.clip(raw['dvl']   / 2.0, -1.0, 1.0),                       # 3
            np.clip(raw['gyro']  / 6.0, -1.0, 1.0),                       # 3
            raw['quat'],                                                    # 4
            gravity_body,                                                   # 3
            [np.clip(depth / 50.0, 0.0, 1.0)],                            # 1
            np.clip(raw.get('sonar', np.zeros(15)) / 12.0, 0.0, 1.0),    # 15
            [np.clip(raw.get('altitude', 0.0) / 50.0, 0.0, 1.0)],        # 1
            np.clip(raw['accel'] / 9.81, -3.0, 3.0),                      # 3
        ]).astype(np.float32)
        return obs
