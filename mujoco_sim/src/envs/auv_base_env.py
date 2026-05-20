import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from src.core.hydro_plugin import HydroDynamicsPlugin
from src.core.robot import YuyuanRobot
from src.core.scene_builder import SceneBuilder
from src.core.sensors import SensorManager
from src.envs.tasks import TASK_REGISTRY


class AUVGymEnv(gym.Env):
    """
    Gymnasium environment for the Yuyuan AUV.

    Responsibilities
    ----------------
    - Owns the MuJoCo model and data objects.
    - Instantiates and wires together Robot, SensorManager,
      HydroDynamicsPlugin, and SceneBuilder.
    - Delegates observation construction, reward computation, and
      termination logic to a pluggable Task instance.
    - Exposes ``set_stage`` for curriculum-learning callbacks.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg_env, cfg_task):
        super().__init__()

        self.cfg_env = cfg_env
        self.cfg_task = cfg_task

        # MuJoCo engine
        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.render_mode = cfg_env.render_mode

        # Core components
        self.robot = YuyuanRobot(self.model, self.data)
        self.sensors = SensorManager(self.model, self.data)
        self.WATER_SURFACE_Z = self.sensors.WATER_SURFACE_Z

        self.scene_builder = SceneBuilder(self.model, self.data, max_obstacles=10)

        self.current_stage = (
            cfg_task.curriculum.initial_stage
            if hasattr(cfg_task, 'curriculum') else 0
        )

        self.hydro = HydroDynamicsPlugin(
            cfg_env.weights.mlp,
            cfg_env.weights.scaler_x,
            cfg_env.weights.scaler_y,
            self.dt,
        )

        # Task
        if cfg_task.name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{cfg_task.name}'. "
                f"Available: {list(TASK_REGISTRY.keys())}"
            )
        self.task = TASK_REGISTRY[cfg_task.name](cfg_task)

        # Gym spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        obs_dim = self.task.get_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.max_steps = cfg_env.max_steps
        self.current_step = 0

        self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "yuyuan")
        self.base_mass = self.model.body_mass[self.robot_body_id]

        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        self.target_mocap_id = (
            self.model.body_mocapid[target_body_id] if target_body_id != -1 else -1
        )

        self.fixed_start_pos  = np.array(cfg_task.goals.start_pos)
        self.base_target_pos  = np.array(cfg_task.goals.target_pos)
        self.fixed_target_pos = self.base_target_pos.copy()
        self.target_pos       = self.fixed_target_pos

    # ------------------------------------------------------------------
    # Curriculum interface
    # ------------------------------------------------------------------

    def set_stage(self, stage: int):
        """Advance the environment to a higher curriculum stage."""
        print(f"[Curriculum] Environment advancing to stage {stage}.")
        self.current_stage = stage

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.hydro.reset()
        self.current_step = 0
        self.model.body_mass[self.robot_body_id] = self.base_mass

        # Domain randomisation
        rand_cfg = self.cfg_env.randomization
        pos_noise_range = np.array(rand_cfg.pos_noise)
        start_pos = self.fixed_start_pos + np.random.uniform(-pos_noise_range, pos_noise_range)
        start_pos[2] = min(start_pos[2], 14.0)
        self.data.qpos[0:3] = start_pos

        angle_noise = np.random.uniform(
            -np.array(rand_cfg.angle_noise),
             np.array(rand_cfg.angle_noise),
        )
        self.data.qpos[3:7] = self._euler_to_quat(*angle_noise)

        target_noise = np.random.uniform(-pos_noise_range, pos_noise_range)
        self.fixed_target_pos = self.base_target_pos + target_noise
        self.target_pos = self.fixed_target_pos

        self.scene_builder.reset_scene(
            stage=self.current_stage,
            start_pos=start_pos,
            target_pos=self.target_pos,
        )
        mujoco.mj_forward(self.model, self.data)

        # Warm up the hydro plugin and Kalman filter before the episode starts
        for _ in range(20):
            self.hydro.apply_hydrodynamics(self.robot)
            mujoco.mj_step(self.model, self.data)

        self.task.reset(self)

        if self.target_mocap_id != -1:
            self.data.mocap_pos[self.target_mocap_id] = self.target_pos

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        clamped_action = np.clip(action, -1.0, 1.0)

        self.robot.set_thrusters_6dof(clamped_action)
        self.hydro.apply_hydrodynamics(self.robot)

        # Cache applied forces so they persist across the frame-skip substeps
        applied_force_cache = self.data.xfrc_applied.copy()
        for _ in range(self.cfg_env.frame_skip):
            self.data.xfrc_applied = applied_force_cache
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, _, reward_info = self.task.compute_reward(self, clamped_action, obs)
        terminated, reason = self.task.is_done(self, self.current_step, self.max_steps)
        truncated = (reason == "timeout")

        info = {
            "termination_reason": reason,
            "stats/mean_thrust": float(np.mean(np.abs(clamped_action))),
            **reward_info,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return self.task.get_obs(self)

    def render(self):
        pass   # rendering is driven externally via mujoco.viewer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _euler_to_quat(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert ZYX Euler angles to a unit quaternion [w, x, y, z]."""
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ])
