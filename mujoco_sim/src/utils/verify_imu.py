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

def get_body_rotation_matrix(model, data, body_id):
    """获取从 Body Frame 到 World Frame 的旋转矩阵"""
    # xmat 是展平的 9 元素数组，重塑为 3x3
    return data.xmat[body_id].reshape(3, 3)

def test_imu_logic():
    print("\n========== IMU Ground Truth Verification ==========")
    
    # 1. 初始化
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    
    robot = YuyuanRobot(model, data)
    sensors = SensorManager(model, data)
    hydro = HydroDynamicsPlugin(MODEL_WEIGHTS['mlp'], MODEL_WEIGHTS['scaler_x'], MODEL_WEIGHTS['scaler_y'], dt)

    mujoco.mj_resetData(model, data)
    
    # 2. 仿真参数
    steps = 2000
    thrust_surge = 5000.0  # 向前推力
    
    # 用于计算 Ground Truth 的历史速度
    prev_vel_body = np.zeros(6)
    total_mass = 290.0
    
    print(f"{'Step':<5} | {'Source':<10} | {'Surge (X)':<10} | {'Heave (Z)':<10} | {'Status'}")
    print("-" * 70)

    for i in range(steps):
        # --- A. 施加控制力 ---
        data.xfrc_applied[:] = 0
        
        # 你的水动力插件 (包含了 Added Mass 计算)
        # 注意：这里 hydro 内部调用了 KF.update(vel)，所以 KF 已经更新了
        current_vel, hydro_force = hydro.apply_hydrodynamics(robot)
        
        # 施加测试推力 (Surge 方向)
        data.xfrc_applied[robot.body_id][2] += thrust_surge
        
        # --- B. MuJoCo 物理步进 ---
        mujoco.mj_step(model, data)
        
        # --- C. 获取数据进行验证 ---
        
        # 1. 读取 IMU 数据 (Sensor Frame = usually Body Frame)
        # 假设 sensors.get_imu_data() 返回的是 (accel, gyro)
        # accel 单位通常是 m/s^2, 包含重力分量
        imu_accel, imu_gyro = sensors.get_imu_data()
        
        # 2. 计算 Ground Truth (GT) 加速度
        # 方法：(当前速度 - 上一帧速度) / dt
        # 注意：MuJoCo 的 qvel 对于 Free Joint 通常是在 Body Frame 下的 (check your xml definition)
        # 如果你的 robot.get_body_state() 返回的是 Body Frame 速度，直接用它
        vel_body_now, _ = robot.get_body_state()
        
        # 计算真实的运动学加速度 (Kinematic Acceleration)
        # acc = dv/dt
        gt_kinematic_accel = (vel_body_now - prev_vel_body) / dt
        
        # 3. 处理重力分量 (关键！)
        # IMU 测量的是: a_measured = a_kinematic + R^T * g_world
        # MuJoCo 的重力通常是 [0, 0, -9.81]
        g_world = model.opt.gravity # 通常是 [0, 0, -9.81]
        rot_mat = get_body_rotation_matrix(model, data, robot.body_id)
        
        # 将重力转到 Body Frame (注意旋转矩阵的转置/逆)
        # g_body = R^T * g_world
        g_body = rot_mat.T @ g_world 
        
        # 理论上 IMU 应该读到的值
        expected_imu_reading = gt_kinematic_accel[:3] - g_body 
        # 注意：MuJoCo accelerometer sensor 具体的符号定义通常是 acc_measured = acc_body - g_body
        # 如果静止平放，g_body = [0,0,-9.8]，read = 0 - (-9.8) = +9.8 (向上)
        
        # 4. 获取你的 KF 预测值 (用于对比)
        # 你的 KalmanFilter6D 存储在 hydro.kf 中
        # 获取 KF 刚刚预测出的加速度 (State 向量的第二个分量)
        kf_accel_est = np.array([hydro.kf.x[dim][1] for dim in range(6)])

        # --- D. 数据更新 ---
        prev_vel_body = vel_body_now.copy()

        # --- E. 打印与判定 (每10步打印一次) ---
        if i > 5 and i % 10 == 0: # 前几步跳过，因为差分计算不稳定
            
            # 比较 X轴 (Surge)
            # 1. IMU 读数
            imu_x = imu_accel[0]
            # 2. 系统真实加速度 (Ground Truth) - 纯运动
            gt_x = gt_kinematic_accel[0]
            # 3. 期望 IMU 读数 (GT + Gravity)
            expect_x = expected_imu_reading[0]
            
            # 比较 Z轴 (Heave)
            imu_z = imu_accel[2]
            gt_z = gt_kinematic_accel[2]
            expect_z = expected_imu_reading[2]
            
            # 判定标准：MuJoCo IMU 读数是否符合物理规律
            # 误差容忍度
            tol = 0.5 
            match = abs(imu_x - expect_x) < tol and abs(imu_z - expect_z) < tol
            status = "✅ OK" if match else "❌ FAIL"
            
            print(f"{i:<5} | IMU Raw   | {imu_x:10.4f} | {imu_z:10.4f} |")
            print(f"{'':<5} | GT (Kin)  | {gt_x:10.4f} | {gt_z:10.4f} | <-- 真实的物体运动加速度")
            print(f"{'':<5} | GT + Grav | {expect_x:10.4f} | {expect_z:10.4f} | <-- 理论上IMU应读到的值")
            print(f"{'':<5} | KF Pred   | {kf_accel_est[0]:10.4f} | {kf_accel_est[2]:10.4f} | <-- 你的KF估算值")
            print(f"{'':<5} | {'Diff':<10}| {abs(imu_x - expect_x):10.4f} | {abs(imu_z - expect_z):10.4f} | {status}")
            print("-" * 70)

if __name__ == "__main__":
    test_imu_logic()