"""
Environment sanity checker — validates buoyancy, thruster mapping, rewards, and observations.

Usage:
    cd mujoco_sim
    python scripts/check_env.py
"""

import os
import sys
import numpy as np
from omegaconf import OmegaConf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.envs.auv_base_env import AUVGymEnv


def sanity_check():
    cfg_env  = OmegaConf.load(os.path.join(project_root, "configs/env/default.yaml"))
    cfg_task = OmegaConf.load(os.path.join(project_root, "configs/task/stage1_navigate.yaml"))

    print("Initialising environment ...")
    try:
        env = AUVGymEnv(cfg_env, cfg_task)
    except Exception as e:
        print(f"Environment init failed: {e}")
        return

    obs, info = env.reset()
    body_id = env.robot.body_id
    print("Environment initialised.")

    # --- Test 1: buoyancy (near-neutral buoyancy with zero thrust) ---
    print("\n[Test 1] Zero thrust for 50 steps (buoyancy check) ...")
    z_start = env.data.xpos[body_id][2]
    action = np.zeros(env.action_space.shape[0])
    for _ in range(50):
        env.step(action)
    z_end = env.data.xpos[body_id][2]
    z_diff = z_end - z_start
    print(f"  Z drift over 50 steps: {z_diff:.4f} m")
    if abs(z_diff) < 0.5:
        print("  PASS: near-neutral buoyancy.")
    elif z_diff < -0.5:
        print("  WARN: sinking fast — check BUOYANCY_MAGNITUDE.")
    else:
        print("  WARN: rising fast — check BUOYANCY_MAGNITUDE.")

    # --- Test 2: forward thrust ---
    print("\n[Test 2] Forward thrust for 100 steps ...")
    env.reset()
    # Surge command: positive surge, all others zero
    action = np.zeros(env.action_space.shape[0])
    action[0] = 1.0
    x_start  = env.data.xpos[body_id][0]
    d_start  = np.linalg.norm(env.data.xpos[body_id] - env.target_pos)
    rewards  = []
    for _ in range(100):
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    x_end = env.data.xpos[body_id][0]
    d_end = np.linalg.norm(env.data.xpos[body_id] - env.target_pos)
    print(f"  X displacement: {x_end - x_start:.4f} m")
    print(f"  Distance to target: {d_start:.2f} -> {d_end:.2f} m")
    if x_end - x_start > 0.1:
        print("  PASS: vehicle moves in +X direction.")
    elif x_end - x_start < -0.1:
        print("  WARN: vehicle moves in -X — check thruster sign convention.")
    else:
        print("  FAIL: vehicle barely moved — check max_thrust or drag.")

    # --- Test 3: reward trend ---
    print("\n[Test 3] Reward trend ...")
    if rewards:
        print(f"  First reward: {rewards[0]:.4f}")
        print(f"  Last  reward: {rewards[-1]:.4f}")
        if rewards[-1] > rewards[0]:
            print("  PASS: reward increases as vehicle approaches target.")
        else:
            print("  WARN: reward did not increase — check reward function.")

    # --- Test 4: observation validity ---
    print("\n[Test 4] Observation validity ...")
    print(f"  Obs shape: {obs.shape}  min={np.min(obs):.3f}  max={np.max(obs):.3f}")
    if np.isnan(obs).any():
        print("  FAIL: observation contains NaN — check sensor or physics.")
    else:
        print("  PASS: no NaN in observation.")

    print("\nSanity check complete.")


if __name__ == "__main__":
    sanity_check()
