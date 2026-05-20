"""
SAC training entry point for the Yuyuan AUV.

Usage:
    cd mujoco_sim
    python scripts/train.py [hydra overrides]

Examples:
    python scripts/train.py task=stage2_avoidance
    python scripts/train.py total_timesteps=3000000 device=cpu
    python scripts/train.py pretrained.model_path=outputs/navigate/seed_1/final_model.zip \\
                            pretrained.vecnorm_path=outputs/navigate/seed_1/vec_normalize.pkl
"""

import os
import sys

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.envs import AUVGymEnv

# Limit CPU thread usage so multiple environments share cores fairly
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(cfg, rank: int):
    def _init():
        env = AUVGymEnv(cfg.env, cfg.task)
        seed = cfg.seed + rank
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return Monitor(env, allow_early_resets=True)
    return _init


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """
    Monitors episodic success rate and advances the curriculum stage
    once the target rate is sustained over a sliding window.
    """

    def __init__(
        self,
        initial_stage: int = 0,
        eval_env=None,
        target_success_rate: float = 0.85,
        window_size: int = 60,
        max_stage: int = 4,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.custom_eval_env = eval_env
        self.target_success_rate = target_success_rate
        self.window_size = window_size
        self.max_stage = max_stage
        self.current_stage = initial_stage
        self.success_history = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals['dones']):
            if not done:
                continue

            info = self.locals['infos'][idx]
            # In VecEnv the real terminal info is wrapped in 'final_info'
            actual_info = info.get('final_info', info)
            self.success_history.append(float(actual_info.get('is_success', 0.0)))
            self.episode_count += 1

            if len(self.success_history) > self.window_size:
                self.success_history.pop(0)

            if self.episode_count % 20 == 0 and self.success_history:
                rate = np.mean(self.success_history)
                print(
                    f"[Curriculum] Stage {self.current_stage} | "
                    f"{len(self.success_history)}/{self.window_size} episodes | "
                    f"success rate: {rate * 100:.1f}% / target: "
                    f"{self.target_success_rate * 100:.1f}%"
                )

            if (len(self.success_history) == self.window_size and
                    np.mean(self.success_history) >= self.target_success_rate and
                    self.current_stage < self.max_stage):
                self.current_stage += 1
                self.training_env.env_method("set_stage", self.current_stage)
                if self.custom_eval_env is not None:
                    self.custom_eval_env.env_method("set_stage", self.current_stage)
                self.success_history = []
                self.episode_count = 0
                print(f"\n[Curriculum] Advancing to stage {self.current_stage}!\n")
                if wandb.run is not None:
                    wandb.log({"curriculum/stage": self.current_stage})

        return True


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    try:
        save_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        save_dir = os.getcwd()

    print(f"Experiment: {cfg.project_name}")
    print(f"Save directory: {save_dir}")

    run = wandb.init(
        project=cfg.project_name,
        name=f"{cfg.task.name}_{os.path.basename(save_dir)}",
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        dir=save_dir,
    )

    # SAC is most sample-efficient with a single environment
    env = DummyVecEnv([make_env(cfg, 0)])

    stats_path = cfg.pretrained.vecnorm_path
    if stats_path and not os.path.isabs(stats_path):
        import hydra.utils
        stats_path = os.path.join(hydra.utils.get_original_cwd(), stats_path)

    if stats_path and os.path.exists(stats_path):
        print(f"Loading VecNormalize stats: {stats_path}")
        env = VecNormalize.load(stats_path, env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=300.0)

    # Evaluation environment
    eval_env = DummyVecEnv([make_env(cfg, 100)])
    if stats_path and os.path.exists(stats_path):
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    # Model — load pretrained or initialise fresh
    model_path = cfg.pretrained.model_path
    if model_path and not os.path.isabs(model_path):
        import hydra.utils
        model_path = os.path.join(hydra.utils.get_original_cwd(), model_path)

    if model_path and os.path.exists(model_path):
        print(f"Loading pretrained model: {model_path}")
        model = SAC.load(
            model_path,
            env=env,
            device=cfg.device,
            custom_objects={"learning_rate": cfg.hyperparams.learning_rate, "learning_starts": 0},
            tensorboard_log=save_dir,
        )
        if cfg.pretrained.load_buffer and cfg.pretrained.buffer_path:
            buffer_path = cfg.pretrained.buffer_path
            if not os.path.isabs(buffer_path):
                buffer_path = os.path.join(hydra.utils.get_original_cwd(), buffer_path)
            if os.path.exists(buffer_path):
                print(f"Loading replay buffer: {buffer_path}")
                model.load_replay_buffer(buffer_path)
    else:
        print("Initialising new SAC model.")
        hyperparams = dict(cfg.hyperparams)
        hyperparams.setdefault("train_freq", 1)
        hyperparams.setdefault("gradient_steps", 1)
        model = SAC(
            "MlpPolicy", env,
            verbose=1,
            tensorboard_log=save_dir,
            device=cfg.device,
            **hyperparams,
        )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=max(10000, 1),
        deterministic=True,
        render=False,
    )
    wandb_cb = WandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(save_dir, "wandb_models"),
        verbose=2,
    )
    callbacks = [checkpoint_cb, eval_cb, wandb_cb]

    if "curriculum" in cfg.task:
        curr_cfg = cfg.task.curriculum
        env.env_method("set_stage", curr_cfg.initial_stage)
        eval_env.env_method("set_stage", curr_cfg.initial_stage)
        callbacks.append(CurriculumCallback(
            initial_stage=curr_cfg.initial_stage,
            target_success_rate=curr_cfg.target_success_rate,
            window_size=curr_cfg.window_size,
            max_stage=curr_cfg.max_stage,
            eval_env=eval_env,
        ))

    # Train
    print("Starting training ...")
    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=cfg.pretrained.reset_timesteps,
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        print("Saving final model ...")
        model.save(os.path.join(save_dir, "final_model"))
        env.save(os.path.join(save_dir, "vec_normalize.pkl"))
        if cfg.pretrained.save_buffer_at_end:
            model.save_replay_buffer(os.path.join(save_dir, "final_replay_buffer"))
        run.finish()
        print(f"Done. Artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
