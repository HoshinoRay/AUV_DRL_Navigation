import numpy as np

from .base_task import BaseTask


class AvoidanceTask(BaseTask):
    """
    Stage 2 task: navigation with dynamic obstacle avoidance.

    Extends NavigationTask's reward structure with an exponential repulsive
    field from the sonar array: the closer any beam reading is to the
    critical distance, the larger the per-step penalty.
    """

    OBS_DIM = 36

    def __init__(self, config):
        super().__init__(config)
        self.cfg   = config.reward_weights
        self.goals = config.goals

        safety = getattr(config, 'safety', None)
        self.warning_distance  = getattr(safety, 'warning_distance',  4.0) if safety else 4.0
        self.critical_distance = getattr(safety, 'critical_distance', 0.4) if safety else 0.4

        self.obs_dim = self.OBS_DIM
        self.last_action = None
        self.last_potential = None
        self.current_is_collision = False

    def get_obs_dim(self):
        return self.obs_dim

    def reset(self, env):
        self.last_action = None
        self.current_is_collision = False
        env.target_pos = getattr(env, 'fixed_target_pos', np.array([30.0, 0.0, 10.0]))
        self.last_potential = self._calc_potential(env)

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _calc_potential(self, env) -> float:
        pos  = env.data.xpos[env.robot.body_id]
        dist = np.linalg.norm(pos - env.target_pos)
        return -(dist / self.goals.max_dist) * self.cfg.phi_dist

    def _get_posture_metrics(self, pos, target, rot_mat):
        body_x   = rot_mat[:, 0]
        body_z   = rot_mat[:, 2]
        world_up = np.array([0.0, 0.0, 1.0])

        vec  = target - pos
        dist = np.linalg.norm(vec)
        desired_x = vec / (dist + 1e-6)

        if abs(desired_x[2]) > 0.99:
            desired_y = np.array([0.0, 1.0, 0.0])
        else:
            desired_y = np.cross(world_up, desired_x)
            desired_y /= np.linalg.norm(desired_y)

        desired_z = np.cross(desired_x, desired_y)

        align_cos    = float(np.dot(body_x, desired_x))
        up_cos       = float(np.dot(body_z, desired_z))
        error_y_roll = 1.0 - abs(rot_mat[:, 1][2])
        return dist, align_cos, up_cos, error_y_roll

    # ------------------------------------------------------------------
    # Task interface
    # ------------------------------------------------------------------

    def compute_reward(self, env, action, obs):
        raw     = env.sensors.get_raw_data()
        body_id = env.model.body('yuyuan').id
        rot_mat = env.data.xmat[body_id].reshape(3, 3)
        pos     = env.data.xpos[body_id].copy()
        target  = env.target_pos

        dist, align_cos, up_cos, error_y_roll = self._get_posture_metrics(pos, target, rot_mat)

        # Potential-based shaping
        current_potential = self._calc_potential(env)
        reward_shaping = (current_potential - self.last_potential) * 10.0
        reward_align   = 0.7 * (align_cos + 1.0) * self.cfg.w_align_err
        reward_roll    = 0.5 * (up_cos    + 1.0) * self.cfg.w_roll_err

        # Obstacle repulsive field
        sonar_dists      = raw.get('sonar', np.ones(15) * 12.0)
        min_sonar_dist   = float(np.min(sonar_dists))
        obstacle_penalty = 0.0
        self.current_is_collision = False

        if min_sonar_dist < self.critical_distance:
            self.current_is_collision = True
            obstacle_penalty = self.cfg.w_collision
        else:
            for d in sonar_dists:
                if d < self.warning_distance:
                    # Exponential repulsion: smooth but intensifying near the wall
                    penalty_factor = np.exp(-1.5 * (d - self.critical_distance))
                    obstacle_penalty += self.cfg.w_danger_zone * penalty_factor

        # Success
        reward_success     = 0.0
        reward_final_bonus = 0.0
        is_success = dist < self.goals.success_dist and not self.current_is_collision

        if is_success:
            reward_success = self.cfg.success
            w_bonus = getattr(self.cfg, 'w_final_bonus', 500.0)
            align_score = (align_cos + 1.0) / 2.0
            up_score    = (up_cos    + 1.0) / 2.0
            reward_final_bonus = 0.2 * w_bonus * (align_score + up_score)
            time_penalty = 0.0
        else:
            time_penalty = self.cfg.time_penalty

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
            - obstacle_penalty
            - cost_energy
            - cost_action
            - cost_smooth
            + bonus_roll
            - time_penalty
        )

        self.last_potential = current_potential
        self.last_action    = action.copy()

        info = {
            "rew/shaping":          reward_shaping,
            "rew/align":            reward_align,
            "rew/obstacle_penalty": -obstacle_penalty,
            "state/align_cos":      align_cos,
            "state/dist":           dist,
            "state/min_sonar_dist": min_sonar_dist,
            "is_success":           float(is_success),
            "is_collision":         float(self.current_is_collision),
        }
        return total_reward, is_success, info

    def is_done(self, env, current_step, max_steps):
        body_id = env.model.body('yuyuan').id
        pos     = env.data.xpos[body_id].copy()
        if np.linalg.norm(pos - env.target_pos) < self.goals.success_dist:
            return True, "success"
        if current_step >= max_steps:
            return True, "timeout"
        return False, None

    def get_obs(self, env):
        raw     = env.sensors.get_raw_data()
        pos_world = env.data.xpos[env.robot.body_id]
        rot_mat   = env.data.xmat[env.robot.body_id].reshape(3, 3)

        target_vec_body = rot_mat.T @ (env.target_pos - pos_world)
        gravity_body    = rot_mat.T @ np.array([0.0, 0.0, -1.0])
        depth = env.WATER_SURFACE_Z - pos_world[2]

        return np.concatenate([
            np.clip(target_vec_body / self.goals.max_dist, -1.0, 1.0),
            np.clip(raw['dvl']   / 2.0, -1.0, 1.0),
            np.clip(raw['gyro']  / 6.0, -1.0, 1.0),
            raw['quat'],
            gravity_body,
            [np.clip(depth / 50.0, 0.0, 1.0)],
            np.clip(raw.get('sonar', np.zeros(15)) / 12.0, 0.0, 1.0),
            [np.clip(raw.get('altitude', 0.0) / 50.0, 0.0, 1.0)],
            np.clip(raw['accel'] / 9.81, -3.0, 3.0),
        ]).astype(np.float32)
