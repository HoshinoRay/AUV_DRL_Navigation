import os
import sys
import time
import numpy as np
import mujoco.viewer
import hydra
from omegaconf import DictConfig
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import hydra.utils

# 路径补丁：确保能识别项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.envs.auv_base_env import AUVGymEnv

def to_absolute_path(path):
    """辅助函数：处理 Hydra 改变工作目录后的绝对路径转换"""
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    # 强制将相对路径转换为项目根目录下的绝对路径
    return os.path.join(hydra.utils.get_original_cwd(), path)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def enjoy(cfg: DictConfig):
    # --- 1. 路径与设备处理 ---
    model_path = to_absolute_path(cfg.pretrained.model_path)
    vec_norm_path = to_absolute_path(cfg.pretrained.vecnorm_path)

    if not model_path or not os.path.exists(model_path):
        print(f"❌ 错误：模型路径不存在: {model_path}")
        print("请检查 configs/config.yaml 或使用命令行覆盖: python scripts/enjoy.py pretrained.model_path='...'")
        return

    # --- 2. 核心修复：创建带 Monitor 且锁定种子的环境闭包 ---
    def make_env():
        # 实例化原生环境
        env_instance = AUVGymEnv(cfg.env, cfg.task)
        
        # 注入 config.yaml 中的全局种子，保证测试可复现
        seed = cfg.seed
        env_instance.action_space.seed(seed)
        env_instance.observation_space.seed(seed)
        
        # 【关键修复】必须包裹 Monitor，否则 info 里永远拿不到 'episode' 字典
        return Monitor(env_instance, allow_early_resets=True)

    # 创建单进程 VecEnv
    env = DummyVecEnv([make_env])
    
    # --- 3. 加载标准化统计量 ---
    if vec_norm_path and os.path.exists(vec_norm_path):
        print(f"📦 加载标准化参数: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        # 关键：测试模式下不要更新统计量，也不要归一化奖励（我们要看控制输出和物理真实得分）
        env.training = False
        env.norm_reward = False
    else:
        print("⚠️ 未找到 vec_normalize.pkl，如果训练时使用了 VecNormalize，现在的效果会很差！")

    # --- 4. 加载 SAC 模型 ---
    print(f"🧠 正在加载模型: {model_path} (Device: {cfg.device})")
    model = SAC.load(model_path, env=env, device=cfg.device)

    # ==========================================
    # [新增核心]: 动态设置测试关卡 (Stage)
    # ==========================================
    # 尝试从 config 中读取测试关卡（如果是基础导航任务，没有这个字段也没关系，默认是0）
    test_stage = 0
    if hasattr(cfg.task, 'curriculum'):
        test_stage = cfg.task.curriculum.initial_stage
    
    # 通过 env_method 广播给底层环境 (必须在 env.reset() 之前调用)
    env.env_method("set_stage", test_stage)
    print(f"🎯 环境已强制设置为 Stage: {test_stage}")

    # --- 5. 获取底层 MuJoCo 对象用于渲染 ---
    # 【优雅解法】不论外层包了 VecNormalize 还是 DummyVecEnv 还是 Monitor
    # 使用 .unwrapped 可以直接穿透所有 SB3 的 Wrapper，安全拿到原生的 AUVGymEnv
    raw_env = env.venv.envs[0].unwrapped
    
    obs = env.reset()
    success_count = 0
    crash_count = 0
    episode_count = 0
    current_step_cnt = 0

    print("🚀 启动可视化窗口 (按 Ctrl+C 退出)...")
    
    # 使用 Passive Viewer，确保 MuJoCo 能正确绑定 model 和 data
    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        viewer.sync()
        
        while viewer.is_running():
            # 1. 模型预测动作 (deterministic=True 确保测试时动作不带探索噪声)
            action, _ = model.predict(obs, deterministic=True)
            
            # 2. 环境步进
            obs, reward, dones, infos = env.step(action)
            current_step_cnt += 1
            
            done = dones[0]
            info = infos[0] 
            
            # 3. 同步渲染画面
            viewer.sync()
            
            # 4. 处理回合结束逻辑
            if done:
                episode_count += 1
                
                # --- 获取步数 (现在肯定能从 Monitor 拿到了) ---
                ep_info = info.get('episode', {})
                steps = ep_info.get('l', current_step_cnt)
                
                # --- 获取距离 (兼容多种命名兜底) ---
                dist = -1.0
                possible_keys = ['current_dist', 'distance', 'dist_to_target', 'dist', 'state/dist']
                for k in possible_keys:
                    if k in info:
                        dist = info[k]
                        break
                
                # 终极兜底：直接算三维欧氏距离
                if dist < 0:
                    try:
                        robot_id = raw_env.robot.body_id
                        curr_pos = raw_env.data.xpos[robot_id]
                        target_pos = raw_env.target_pos
                        dist = np.linalg.norm(curr_pos - target_pos)
                    except Exception:
                        pass

                # --- 输出状态 ---
                reason = info.get('termination_reason', 'unknown')
                is_success = info.get('is_success', False)
                is_collision = info.get('is_collision', False)
                
                if is_success:
                    success_count += 1
                    print(f"✅ 成功! | 距目标: {dist:.2f}m | 耗时: {steps} 步")
                elif is_collision:
                    crash_count += 1
                    print(f"💥 撞毁! (撞击障碍物) | 距目标: {dist:.2f}m | 存活: {steps} 步")
                else:
                    print(f"❌ 失败 ({reason}) | 距目标: {dist:.2f}m | 耗时: {steps} 步")
                
                print(f"📊 统计 | 局数: {episode_count} | 胜率: {(success_count/episode_count)*100:.1f}% | 撞毁率: {(crash_count/episode_count)*100:.1f}%")
                print("-" * 50)
                
                current_step_cnt = 0

            # 5. 控制帧率 (25 FPS) 避免画面闪得太快
            time.sleep(0.08)

if __name__ == "__main__":
    enjoy()