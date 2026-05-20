"""
Interactive evaluation script — launches a MuJoCo passive viewer.

Usage:
    cd mujoco_sim
    python scripts/evaluate.py
"""

import os
import sys
import numpy as np
import mujoco.viewer
from omegaconf import OmegaConf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.envs.auv_base_env import AUVGymEnv
from src.utils.logger import DataLogger


def main():
    cfg_env  = OmegaConf.load(os.path.join(project_root, "configs/env/default.yaml"))
    cfg_task = OmegaConf.load(os.path.join(project_root, "configs/task/stage1_navigate.yaml"))

    env = AUVGymEnv(cfg_env, cfg_task)

    logger = DataLogger()
    LOG_INTERVAL = 0.1
    next_log_time = 0.0

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            obs, _ = env.reset()

            while viewer.is_running():
                action = np.zeros(6)
                if env.data.time > 0.1:
                    action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

                obs, reward, terminated, truncated, info = env.step(action)

                vel = env.robot.get_body_state()[0]

                if env.data.time >= next_log_time:
                    next_log_time += LOG_INTERVAL
                    print(f"t={env.data.time:.2f} | Vx={vel[0]:.3f} | reward={reward:.2f}")

                viewer.sync()

                if terminated or truncated:
                    env.reset()
    finally:
        logger.close()


if __name__ == "__main__":
    main()
