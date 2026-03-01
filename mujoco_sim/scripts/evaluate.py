import os
import sys
import numpy as np
import mujoco.viewer
from omegaconf import OmegaConf  # 建议使用 OmegaConf 加载 yaml

# 1. 解决路径问题：将项目根目录加入系统路径
# 获取当前脚本所在目录的上一级目录（即 mujoco_sim 根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 现在可以正常导入 src 了
from src.envs.auv_base_env import AUVGymEnv  # 确认你的文件名是 auv_env.py
from src.utils.logger import DataLogger

def main():
    # 2. 加载配置文件 (匹配你的新架构)
    # 假设你在根目录下执行，或者手动指定路径
    cfg_env = OmegaConf.load(os.path.join(project_root, "configs/env/default.yaml"))
    cfg_task = OmegaConf.load(os.path.join(project_root, "configs/task/stage1_navigate.yaml"))
    
    # 3. 如果需要覆盖配置文件中的权重路径，可以在这里改
    # cfg_env.weights.mlp = "./weights/best_hydro_model.pth" 
    
    # 4. 初始化环境
    env = AUVGymEnv(cfg_env, cfg_task)

    logger = DataLogger()
    LOG_INTERVAL = 0.1 
    next_log_time = 0.0

    try:
        # 使用 env.model 和 env.data
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            obs, _ = env.reset()
            
            while viewer.is_running():
                # 5. 设定 Action
                # 现在环境接受 6 维动作 [-1, 1]
                action = np.zeros(6)
                if env.data.time > 0.1:
                    # 示例：向前进 0.5 强度
                    action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]) 
                
                # 6. 执行 Step
                # 注意：AUVGymEnv.step 返回的是标准 Gym 格式 (obs, reward, term, trunc, info)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 获取速度和水动力（从 info 或 robot 中获取）
                vel = env.robot.get_body_state()[0] # 获取当前速度
                # 如果你在 HydroDynamicsPlugin 中返回了力，可以通过 info 传递出来
                
                if env.data.time >= next_log_time:
                    # logger.log(env.data.time, vel, ...) 
                    next_log_time += LOG_INTERVAL
                    print(f"t={env.data.time:.2f} | Vx={vel[0]:.3f} | Reward={reward:.2f}")

                viewer.sync()
                
                if terminated or truncated:
                    env.reset()
                    
    finally:
        # logger.close()
        pass

if __name__ == "__main__":
    main()