import numpy as np
import time
from src.envs.auv_base_env import AUVGymEnv

# 配置路径 (根据你的实际结构调整)
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(current_dir, "assets", "yuyuan.xml")
# 确保这里指向真实的权重文件，如果没有，可以用随机数临时替代测试
MODEL_WEIGHTS = {
    'mlp': "./weights/best_hydro_model.pth", 
    'scaler_x': "./weights/scaler_X.pkl",
    'scaler_y': "./weights/scaler_Y.pkl"
}

def sanity_check():
    print("⚡ 正在初始化环境...")
    try:
        env = AUVGymEnv(XML_PATH, MODEL_WEIGHTS)
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        return

    obs, info = env.reset()
    print("✅ 环境初始化成功！")
    print(f"   初始距离: {info['target_dist']:.4f} m")
    print(f"   初始 Z 坐标: {env.data.xpos[env.robot_body_id][2]:.4f} (应为 10.0)")
    
    # --- 测试 1: 静止测试 (检查浮力) ---
    print("\n🧪 测试 1: 静止 50 步 (检查是否沉底)...")
    z_start = env.data.xpos[env.robot_body_id][2]
    for _ in range(50):
        # 所有推力为 0
        action = np.zeros(8)
        obs, reward, terminated, truncated, info = env.step(action)
    
    z_end = env.data.xpos[env.robot_body_id][2]
    z_diff = z_end - z_start
    print(f"   50步后 Z 轴变化: {z_diff:.4f} m")
    if z_diff < -0.5:
        print("⚠️ 警告: 机器人下沉过快！请检查浮力参数 (BUOYANCY_MAGNITUDE)。")
    elif z_diff > 0.5:
        print("⚠️ 警告: 机器人上浮过快！")
    else:
        print("✅ 浮力平衡检查通过 (接近中性浮力)。")

    # --- 测试 2: 动力测试 (检查推力映射) ---
    print("\n🧪 测试 2: 全力向前 100 步...")
    env.reset()
    
    # 假设前4个是水平推力，根据你的布局调整
    # 如果你是 X 型布局，通常 [1, 1, 1, 1, 0, 0, 0, 0] 或者 [1, 1, -1, -1 ...] 会产生向前推力
    # 这里简单粗暴全给 1.0 (或者根据你的电机 ID 只给水平电机)
    # 你的 Robot 类里写了: t0_hfr, t1_hfl, t2_hrr, t3_hrl (前4个是水平)
    # 假设 X 构型向前走需要前4个都转 (注意正反桨)
    # 先试着给前4个正推力，观察 x 坐标变化
    action = np.array([1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0]) 
    
    x_start = env.data.xpos[env.robot_body_id][0]
    dist_start = np.linalg.norm(env.data.xpos[env.robot_body_id] - env.target_pos)
    
    trajectory = []
    rewards = []
    
    for i in range(100):
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(env.data.xpos[env.robot_body_id].copy())
        rewards.append(reward)
        
        # 简单渲染一下 (如果是 render_mode='human')
        # env.render() 
        
    x_end = env.data.xpos[env.robot_body_id][0]
    dist_end = np.linalg.norm(env.data.xpos[env.robot_body_id] - env.target_pos)
    
    moved_dist = x_end - x_start
    print(f"   X 轴位移: {moved_dist:.4f} m")
    print(f"   目标距离变化: {dist_start:.2f} -> {dist_end:.2f}")
    
    if moved_dist > 0.1:
        print("✅ 推进系统工作正常 (向 X+ 方向移动)。")
    elif moved_dist < -0.1:
        print("⚠️ 警告: 机器人在倒退！请反转推力指令符号或检查电机安装角度。")
    else:
        print("❌ 错误: 机器人几乎没动！检查最大推力(max_thrust)或水阻力过大。")

    # --- 测试 3: 奖励函数检查 ---
    print("\n🧪 测试 3: 奖励函数逻辑...")
    # 既然我们在靠近目标，奖励应该是增加的 (或者负得越来越少)
    print(f"   初始奖励: {rewards[0]:.4f}")
    print(f"   结束奖励: {rewards[-1]:.4f}")
    
    if rewards[-1] > rewards[0]:
        print("✅ 奖励函数趋势正确 (随着靠近目标奖励提升)。")
    else:
        print("⚠️ 警告: 奖励没有提升，请检查 reward 计算逻辑。")

    # --- 测试 4: 传感器数据范围 ---
    print("\n🧪 测试 4: 观测数据检查...")
    print(f"   Obs min: {np.min(obs):.3f}, Max: {np.max(obs):.3f}")
    if np.isnan(obs).any():
        print("❌ 致命错误: 观测数据包含 NaN！检查传感器或物理计算。")
    else:
        print("✅ 观测数据数值正常 (无 NaN)。")

    print("\n🎉 验证完成！如果以上全绿，可以开始训练。")

if __name__ == "__main__":
    sanity_check()