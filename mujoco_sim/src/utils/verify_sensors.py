import sys
import os
import numpy as np
import mujoco
import time

# 1. 路径设置 (必须在导入 src 之前)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. 导入项目组件
from src.core.robot import YuyuanRobot
from src.core.hydro_plugin import HydroDynamicsPlugin
from src.core.sensors import SensorManager

XML_PATH = os.path.join(project_root, "assets", "yuyuan.xml")
MODEL_WEIGHTS = {
    'mlp': os.path.join(project_root, "weights", "best_hydro_model.pth"),
    'scaler_x': os.path.join(project_root, "weights", "scaler_X.pkl"),
    'scaler_y': os.path.join(project_root, "weights", "scaler_Y.pkl")
}

def test_sensor_manager():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    robot = YuyuanRobot(model, data)
    sensors = SensorManager(model, data)
    hydro = HydroDynamicsPlugin(MODEL_WEIGHTS['mlp'], MODEL_WEIGHTS['scaler_x'], MODEL_WEIGHTS['scaler_y'], dt)

    # === Test 1: Heave 方向 Added Mass 测试 ===
    print("\n=== Test 1: Heave Acceleration (Added Mass Check) ===")
    mujoco.mj_resetData(model, data)
    
    test_force_heave = 3000.0  # 施加 3000N 的向上推力
    
    # 我们需要运行 20 步，让 Kalman Filter 能够根据速度变化计算出加速度
    print(f"  Applying {test_force_heave}N Heave force and warming up KF...")
    
    for _ in range(20):
        # 顺序极其重要：
        # 1. 清空上一帧的力
        data.xfrc_applied[:] = 0
        
        # 2. 注入水动力（计算 Added Mass 反向力）
        hydro.apply_hydrodynamics(robot)
        
        # 3. 【手动补丁】如果你的插件用的是 = 而不是 +=，
        # 我们必须在插件运行完后，再把测试力“加回”到 Z 轴上
        data.xfrc_applied[robot.body_id][2] += (-3000)
        
        # 4. 物理步进
        mujoco.mj_step(model, data)

    # 读取 IMU 
    accel = sensors.get_imu_data()[0][2] # 读取 Z 轴加速度
    
    print(f"  Observed Heave Accel: {accel:.4f} m/s^2")
    
    # 逻辑判定
    # 理论值：如果不计入 Added Mass，数值应在 15-17 左右
    # 如果计入了 Added Mass，数值应在 9-11 左右
    if accel < 12.0:
        print(f"  ✅ SUCCESS: Added Mass detected in Heave! (Accel is restricted)")
    else:
        print(f"  ❌ FAILURE: Accel is too high ({accel:.2f}). Added Mass effect not found.")

# === Test 2: 修复后的力矩累积测试 ===
    print("\n=== Test 2: Torque Accumulation (Fixed) ===")
    mujoco.mj_resetData(model, data)
    test_torque = 1000.0
    
    print(f"  Applying {test_torque}Nm Torque...")
    for i in range(101):
        # 每一帧都必须：清零 -> 算水动力 -> 叠加力矩
        data.xfrc_applied[:] = 0
        hydro.apply_hydrodynamics(robot)
        data.xfrc_applied[robot.body_id][5] += test_torque # 叠加在 Yaw 轴
        
        mujoco.mj_step(model, data)
        
        if i % 20 == 0:
            _, gyro = sensors.get_imu_data()
            # 此时角速度应该线性增长
            print(f"    Step {i:3d}: Time {data.time:.3f}s | Gyro Z: {gyro[2]:.4f} rad/s")

    _, final_gyro = sensors.get_imu_data()
    if abs(final_gyro[2]) > 0.1:
        print(f"  ✅ Angular velocity accumulated successfully.")
    else:
        print(f"  ❌ Angular velocity failed to accumulate. Check logic.")

    # --- 测试 3: 传感器逻辑验证 (Depth & Sonar) ---
    print("\n=== Test 3: Static Sensors ===")
    raw_z = data.qpos[2]
    print(f"  True Z (World): {raw_z:.4f} m")
    print(f"  Observed Depth: {sensors.get_depth():.4f} m")
    print(f"  Sonar [Front, Down]: {sensors.get_sonar_dist()}")

    print("\n=== 🏁 Sensor Manager Test Completed ===")

if __name__ == "__main__":
    test_sensor_manager()