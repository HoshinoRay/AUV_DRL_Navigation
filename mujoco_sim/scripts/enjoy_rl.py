"""
Visualisation script — runs a trained SAC policy with the MuJoCo passive viewer.

Usage:
    cd mujoco_sim
    python scripts/enjoy_rl.py pretrained.model_path=<path/to/model.zip> \\
                               pretrained.vecnorm_path=<path/to/vec_normalize.pkl>
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import hydra
import hydra.utils
from omegaconf import DictConfig
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.envs.auv_base_env import AUVGymEnv


def to_absolute_path(path):
    """Resolve a relative path against the original working directory (before Hydra changes it)."""
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(hydra.utils.get_original_cwd(), path)


def render_custom_geoms(viewer, raw_env):
    """Overlay waypoints and lookahead point on the MuJoCo viewer scene."""
    task = getattr(raw_env, 'task', getattr(raw_env, '_task', None))
    if task is None:
        for attr_name in dir(raw_env):
            try:
                attr = getattr(raw_env, attr_name)
                if hasattr(attr, 'planner'):
                    task = attr
                    break
            except Exception:
                continue

    if task is None:
        return

    waypoints   = getattr(task, 'waypoints', [])
    current_idx = getattr(task, 'current_wp_idx', 0)
    lookahead_pt = getattr(task, 'smoothed_lookahead_pt',
                           getattr(task, 'current_lookahead_pt', None))

    scn = viewer.user_scn
    if scn.maxgeom == 0:
        return
    scn.ngeom = 0

    # Draw lookahead sphere (red)
    if lookahead_pt is not None:
        l_pt = np.array(lookahead_pt).flatten()
        if l_pt.shape[0] == 3:
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.3, 0, 0],
                pos=l_pt,
                mat=np.eye(3).flatten(),
                rgba=[1.0, 0.0, 0.0, 1.0],
            )
            scn.ngeom += 1

    # Draw remaining path as green capsule segments
    if isinstance(waypoints, list) and len(waypoints) > 0:
        try:
            robot_pos = raw_env.data.xpos[raw_env.robot.body_id].copy()
            # Only draw waypoints that have not yet been passed
            remaining = waypoints[current_idx:]
            pts = [robot_pos] + [np.array(wp).flatten() for wp in remaining]

            for i in range(len(pts) - 1):
                if scn.ngeom >= scn.maxgeom:
                    break
                p1, p2 = pts[i], pts[i + 1]
                if np.linalg.norm(p1 - p2) < 1e-4:
                    continue
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=[0.05, 0, 0],
                    pos=[0, 0, 0],
                    mat=np.eye(3).flatten(),
                    rgba=[0.0, 1.0, 0.0, 1.0],
                )
                mujoco.mjv_makeConnector(
                    geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.05,
                    p1[0], p1[1], p1[2], p2[0], p2[1], p2[2],
                )
                scn.ngeom += 1
        except Exception:
            pass


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def enjoy(cfg: DictConfig):
    model_path    = to_absolute_path(cfg.pretrained.model_path)
    vec_norm_path = to_absolute_path(cfg.pretrained.vecnorm_path)

    if not model_path or not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        print("Set pretrained.model_path in config or pass it on the command line.")
        return

    def make_env():
        env_instance = AUVGymEnv(cfg.env, cfg.task)
        env_instance.action_space.seed(cfg.seed)
        env_instance.observation_space.seed(cfg.seed)
        return Monitor(env_instance, allow_early_resets=True)

    env = DummyVecEnv([make_env])

    if vec_norm_path and os.path.exists(vec_norm_path):
        print(f"Loading VecNormalize stats: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training   = False
        env.norm_reward = False
    else:
        print("Warning: vec_normalize.pkl not found — performance may degrade if used during training.")

    print(f"Loading model: {model_path}  (device: {cfg.device})")
    model = SAC.load(model_path, env=env, device=cfg.device)

    test_stage = 0
    if hasattr(cfg.task, 'curriculum'):
        test_stage = cfg.task.curriculum.initial_stage
    env.env_method("set_stage", test_stage)
    print(f"Environment stage: {test_stage}")

    raw_env = env.venv.envs[0].unwrapped

    obs = env.reset()
    success_count = 0
    crash_count   = 0
    episode_count = 0
    step_count    = 0

    print("Launching viewer (Ctrl+C to exit) ...")

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        while viewer.is_running():
            t0 = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            step_count += 1

            render_custom_geoms(viewer, raw_env)
            viewer.sync()

            if dones[0]:
                episode_count += 1
                info = infos[0]
                ep_steps = info.get('episode', {}).get('l', step_count)

                # Distance to target: try info keys then fall back to direct computation
                dist = -1.0
                for key in ('current_dist', 'distance', 'dist_to_target', 'dist', 'state/dist'):
                    if key in info:
                        dist = info[key]
                        break
                if dist < 0:
                    try:
                        dist = float(np.linalg.norm(
                            raw_env.data.xpos[raw_env.robot.body_id] - raw_env.target_pos
                        ))
                    except Exception:
                        pass

                is_success   = info.get('is_success', False)
                is_collision = info.get('is_collision', False)
                reason       = info.get('termination_reason', 'timeout')

                if is_success:
                    success_count += 1
                    outcome = "success"
                elif is_collision:
                    crash_count += 1
                    outcome = "collision"
                else:
                    outcome = f"failed ({reason})"

                print(
                    f"Episode {episode_count:4d} | {outcome:<20} | "
                    f"dist={dist:.2f}m | steps={ep_steps} | "
                    f"win={success_count/episode_count*100:.1f}% "
                    f"crash={crash_count/episode_count*100:.1f}%"
                )
                step_count = 0

            elapsed = time.time() - t0
            if elapsed < 0.08:
                time.sleep(0.08 - elapsed)


if __name__ == "__main__":
    enjoy()
