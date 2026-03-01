import os
import sys
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig 
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
from stable_baselines3.common.logger import configure 
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize 
from stable_baselines3.common.callbacks import BaseCallback
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.envs import AUVGymEnv

# 限制 PyTorch 的 CPU 线程数为 1， each process with one core
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# 限制 NumPy 的线程数 (视情况而定)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def make_env(cfg, rank):
    """
    Hydra配置创建环境
    """
    def _init():
        env = AUVGymEnv(cfg.env, cfg.task)
        
        # 2. 【关键】设置随机种子
        # 保证每个环境的随机性不同 (seed + rank)
        # 如果你的环境支持 reset(seed=...) (新版 Gym 标准):
        seed = cfg.seed + rank
        #env.reset(seed=seed) 
        
        # 如果你的环境比较老，或者 reset 不接受 seed，可以用这个:
        # env.seed(seed) 
        
        # 同时设置动作空间和观察空间的种子
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        # 3. 监控包装
        env = Monitor(env, allow_early_resets=True) 
        return env
    return _init


class CurriculumCallback(BaseCallback):
    def __init__(self, initial_stage=0, eval_env=None, target_success_rate=0.85, window_size=60, max_stage=4, verbose=1):
        super().__init__(verbose)
        self.custom_eval_env = eval_env
        self.target_success_rate = target_success_rate
        self.window_size = window_size
        self.max_stage = max_stage
        self.current_stage = initial_stage 
        self.success_history = []
        self.episode_count = 0 # 记录跑了多少局
        
    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][idx]
                
                # 【防坑指南】: 穿透 SB3 的 Auto-reset 机制寻找真正的 info
                # 在 VecEnv 中，真实的 info 被包裹在 'final_info' 或 'terminal_info' 里
                actual_info = info.get('final_info', info)
                
                # 获取成功标志
                is_success = actual_info.get('is_success', 0.0)
                self.success_history.append(float(is_success))
                self.episode_count += 1
                
                if len(self.success_history) > self.window_size:
                    self.success_history.pop(0)
                    
                # 打印当前收集进度 (每 20 局打印一次，避免刷屏)
                if self.episode_count % 20 == 0:
                    current_rate = np.mean(self.success_history) if len(self.success_history) > 0 else 0.0
                    print(f"📊 [Curriculum] Stage {self.current_stage} 评估进度: {len(self.success_history)}/{self.window_size} 局 | 当前胜率: {current_rate*100:.1f}% / 目标: {self.target_success_rate*100:.1f}%")
                    
                if len(self.success_history) == self.window_size:
                    success_rate = np.mean(self.success_history)
                    
                    if success_rate >= self.target_success_rate and self.current_stage < self.max_stage:
                        self.current_stage += 1
                        # 【核心修复 2】：同时升级 train 和 eval 环境
                        self.training_env.env_method("set_stage", self.current_stage)
                        if self.custom_eval_env is not None:
                            self.custom_eval_env.env_method("set_stage", self.current_stage)
                            print(f"🔄 Eval Environment 同步升级至 Stage {self.current_stage}")
                        self.success_history = [] # 清空历史
                        self.episode_count = 0
                        
                        msg = f"🎉🎉🎉 [Curriculum] 成功率达到 {success_rate*100:.1f}%! 恭喜晋级到 Stage {self.current_stage}!"
                        print(f"\n{'='*60}\n{msg}\n{'='*60}\n")
                        
                        if wandb.run is not None:
                            wandb.log({"curriculum/stage": self.current_stage})
                        
        return True


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ---------------------------------------------------------
    # [核心修改 1]: 获取 Hydra 生成的唯一输出目录 (outputs/日期/时间)
    # 无论 hydra.job.chdir 是 True 还是 False，这都能拿到正确的保存位置
    # ---------------------------------------------------------
    try:
        save_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        # 如果不是通过 hydra 启动（比如直接调试），回退到当前目录
        save_dir = os.getcwd()

    print(f"🚀 启动实验: {cfg.project_name}")
    print(f"📂 实际保存路径 (save_dir): {save_dir}") 

    # ---------------------------------------------------------
    # [新增] 1. 初始化 WandB
    # ---------------------------------------------------------
    run = wandb.init(
        project=cfg.project_name, # 项目名称，会自动在网页上创建
        name=f"{cfg.task.name}_{os.path.basename(save_dir)}", # 实验名称，加上时间戳防重名
        config=OmegaConf.to_container(cfg, resolve=True), # 【关键】将 Hydra 配置转为字典上传，便于后续分析参数
        sync_tensorboard=True,    # 【关键】自动把 SB3 的 Tensorboard 数据同步到 WandB
        monitor_gym=True,         # 自动记录环境视频（如果 render_mode='rgb_array'）
        save_code=True,           # 保存当前代码备份
        dir=save_dir,             # wandb 的本地缓存存到 hydra 目录下
    )
    
    # 获取想要并行的环境数量
    # 如果 config 中没有定义 num_envs，默认使用 8 (你可以根据 CPU 核数修改这个默认值)
    n_envs = cfg.num_envs
    print(f"⚡ 正在启动 {n_envs} 个并行环境 (SubprocVecEnv)...")

    # [核心修改]: 将 DummyVecEnv 替换为 SubprocVecEnv
    # make_env(cfg) 返回的是一个函数，我们执行 n_envs 次生成一个列表
    if n_envs > 1:
        env = SubprocVecEnv([make_env(cfg, i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(cfg, 0)])
    
    # [保持不变]: 确保 vecnorm 加载路径是正确的 (如果是绝对路径则不变，如果是相对路径需注意)
    # 建议在 config 中尽量使用绝对路径，或者相对于项目根目录的路径
    stats_path = cfg.pretrained.vecnorm_path 
    if stats_path and not os.path.isabs(stats_path):
        # 如果 config 里写的是相对路径，这里建议用 get_original_cwd() 拼接，防止找不到文件
        import hydra.utils
        stats_path = os.path.join(hydra.utils.get_original_cwd(), stats_path)

    if stats_path and os.path.exists(stats_path):
        print(f"🔄 加载环境统计数据: {stats_path}")
        # VecNormalize 可以直接包裹 SubprocVecEnv，接口和 Dummy 是一样的
        env = VecNormalize.load(stats_path, env)
        env.training = True 
        env.norm_reward = True
    else:
        print("🆕 初始化新的环境归一化")
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=300.0)

     # ---------------------------------------------------------
    # 3. 准备评估环境 (Eval Env)
    # ---------------------------------------------------------
    # Eval 环境必须和 Train 环境有一致的 Observation 归一化参数，
    # 否则模型在评估时会看到它没见过的数值范围。
    eval_env = DummyVecEnv([make_env(cfg, 100)]) # 使用不同的 rank/seed
    
    if stats_path and os.path.exists(stats_path):
        print(f"🔄 [Eval] 加载环境统计数据: {stats_path}")
        # 如果有预训练统计，加载它
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.training = False # 评估时不更新均值方差
        eval_env.norm_reward = False # 评估时不归一化奖励，我们要看真实分数
    else:
        # [陷阱回避] 如果是新训练，Eval Env 初始是空的。
        # 理想做法是编写 Callback 定期把 env 的 stats 同步给 eval_env。
        # 但为简化，这里我们初始化一个新的，注意：这在训练初期会导致评估不准，
        # 随着训练进行，如果 save_vecnormalize=True，再次加载时会变准。
        # 或者：使用 SB3 的 VecNormalize 直接包裹，但在代码层面很难实时共享内存。
        # **解决方案**: 暂时让 eval_env 保持独立，但在 Checkpoint 时我们保存了 env 的 stats，
        # 以后测试载入 best_model 时一定要载入对应的 vec_normalize.pkl。
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    # 2. 准备模型
    # ----------------------------------------
    model = None
    
    # 处理预训练模型路径 (同上，建议处理绝对路径)
    model_path = cfg.pretrained.model_path
    if model_path and not os.path.isabs(model_path):
        import hydra.utils
        model_path = os.path.join(hydra.utils.get_original_cwd(), model_path)

    if model_path and os.path.exists(model_path):
        print(f"📥 加载预训练模型: {model_path}")
        custom_objects = {
            "learning_rate": cfg.hyperparams.learning_rate,
            "learning_starts": 0 
        }
        
        # [核心修改 2]: tensorboard_log 指向 save_dir
        model = SAC.load(
            model_path,
            env=env,
            device=cfg.device,
            custom_objects=custom_objects,
            tensorboard_log=save_dir 
        )
        
        # 处理 Buffer 路径
        buffer_path = cfg.pretrained.buffer_path
        if cfg.pretrained.load_buffer and buffer_path:
            if not os.path.isabs(buffer_path):
                 buffer_path = os.path.join(hydra.utils.get_original_cwd(), buffer_path)
            
            if os.path.exists(buffer_path):
                print(f"♻️  加载经验池: {buffer_path}")
                model.load_replay_buffer(buffer_path)
            else:
                print(f"⚠️  警告: 文件不存在: {buffer_path}")
    else:
        print("✨ 初始化全新 SAC 模型")
        # [核心修改 3]: tensorboard_log 指向 save_dir
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=save_dir, 
            device=cfg.device,
            **cfg.hyperparams 
        )

    # 3. 回调函数
    # ----------------------------------------
    # [核心修改 4]: 所有 save_path 和 log_path 都使用 os.path.join(save_dir, ...)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(save_dir, "checkpoints"), # <--- 修改
        name_prefix="model",
        save_replay_buffer=True, 
        save_vecnormalize=True   
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"), 
        log_path=os.path.join(save_dir, "eval_logs"),             
        eval_freq=max(10000 // n_envs, 1), # 【重要】多核会加快 step 计数，eval_freq 需要除以核数
        deterministic=True,
        render=False
    )

    # ---------------------------------------------------------
    # [新增] 2. 创建 WandB 回调函数
    # ---------------------------------------------------------
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,   # 每1000步记录一次梯度直方图（用于分析是否梯度消失/爆炸）
        model_save_path=os.path.join(save_dir, "wandb_models"), # 可选：把模型传到云端
        verbose=2,
    )

    # [修正] 必须把 wandb_callback 放进列表里！
    callbacks_list = [checkpoint_callback, eval_callback, wandb_callback]

    # 动态判断：如果是 avoidance 任务，则挂载课程学习模块
    if "curriculum" in cfg.task: 
        print("🎓 探测到避障任务，已启用 Curriculum Learning 模块！")
        
        # --- 关键：在这里定义 curr_cfg ---
        curr_cfg = cfg.task.curriculum 

        # -----------------------------------------------------------------
        # [核心新增]：在训练开始前，强制让底层并行环境同步到你指定的 initial_stage
        # 无论包装了多少层 VecNormalize，env_method 都能穿透调用到底层 env 的 set_stage
        # -----------------------------------------------------------------
        env.env_method("set_stage", curr_cfg.initial_stage)
        eval_env.env_method("set_stage", curr_cfg.initial_stage)
        print(f"🔄 已强制将 Train/Eval 环境初始化为 Stage {curr_cfg.initial_stage}")
        
        curriculum_callback = CurriculumCallback(
            initial_stage=curr_cfg.initial_stage, # 现在 curr_cfg 定义过了，不会报错了
            target_success_rate=curr_cfg.target_success_rate,
            window_size=curr_cfg.window_size,
            max_stage=curr_cfg.max_stage,
            eval_env=eval_env
        )
        callbacks_list.append(curriculum_callback)
    # 4. 开始训练
    # ----------------------------------------
    print("🔥 开始训练...")
    try:
        model.learn(
            total_timesteps=cfg.total_timesteps, 
            callback=callbacks_list,
            progress_bar=True,
            reset_num_timesteps=cfg.pretrained.reset_timesteps
        )
    except KeyboardInterrupt:
        print("🛑 训练被手动中断")
    finally:
        # 5. 保存最终结果
        # [核心修改 5]: 最终保存路径也加上 save_dir
        print("💾 保存最终模型...")
        final_model_path = os.path.join(save_dir, "final_model")
        final_vec_path = os.path.join(save_dir, "vec_normalize.pkl")
        final_buffer_path = os.path.join(save_dir, "final_replay_buffer")

        model.save(final_model_path)
        env.save(final_vec_path) 
        
        if cfg.pretrained.save_buffer_at_end:
            model.save_replay_buffer(final_buffer_path)

        # [新增] 结束 WandB run
        run.finish()
            
        print(f"✅ 训练结束，数据已保存至: {save_dir}")

if __name__ == "__main__":
    main()