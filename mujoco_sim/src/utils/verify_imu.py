"""
IMU ground-truth verification.

Compares MuJoCo IMU readings against kinematic ground truth and the Kalman Filter
acceleration estimate for both surge (X) and heave (Z) axes.

Usage:
    cd mujoco_sim
    python src/utils/verify_imu.py
"""

import sys
import os
import numpy as np
import mujoco

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.robot import YuyuanRobot
from src.core.hydro_plugin import HydroDynamicsPlugin
from src.core.sensors import SensorManager

XML_PATH = os.path.join(project_root, "assets", "yuyuan.xml")
MODEL_WEIGHTS = {
    'mlp':      os.path.join(project_root, "weights", "best_hydro_model.pth"),
    'scaler_x': os.path.join(project_root, "weights", "scaler_X.pkl"),
    'scaler_y': os.path.join(project_root, "weights", "scaler_Y.pkl"),
}


def get_body_rotation_matrix(data, body_id) -> np.ndarray:
    """Return the 3×3 rotation matrix from body frame to world frame."""
    return data.xmat[body_id].reshape(3, 3)


def test_imu_logic():
    print("\n========== IMU Ground Truth Verification ==========")

    model   = mujoco.MjModel.from_xml_path(XML_PATH)
    data    = mujoco.MjData(model)
    dt      = model.opt.timestep
    robot   = YuyuanRobot(model, data)
    sensors = SensorManager(model, data)
    hydro   = HydroDynamicsPlugin(
        MODEL_WEIGHTS['mlp'], MODEL_WEIGHTS['scaler_x'], MODEL_WEIGHTS['scaler_y'], dt
    )

    mujoco.mj_resetData(model, data)

    steps         = 2000
    thrust_surge  = 5000.0
    prev_vel_body = np.zeros(6)

    print(f"{'Step':<5} | {'Source':<10} | {'Surge (X)':<10} | {'Heave (Z)':<10} | Status")
    print("-" * 70)

    for i in range(steps):
        data.xfrc_applied[:] = 0
        current_vel, _ = hydro.apply_hydrodynamics(robot)
        data.xfrc_applied[robot.body_id][2] += thrust_surge
        mujoco.mj_step(model, data)

        imu_accel, _ = sensors.get_imu_data()
        vel_body_now, _ = robot.get_body_state()

        # Kinematic ground-truth acceleration (finite difference)
        gt_kinematic_accel = (vel_body_now - prev_vel_body) / dt

        # Expected IMU reading = kinematic_accel - R^T * g_world
        # (MuJoCo accelerometers measure specific force, not pure acceleration)
        g_world  = model.opt.gravity
        rot_mat  = get_body_rotation_matrix(data, robot.body_id)
        g_body   = rot_mat.T @ g_world
        expected_imu = gt_kinematic_accel[:3] - g_body

        kf_accel = np.array([hydro.kf.x[dim][1] for dim in range(6)])
        prev_vel_body = vel_body_now.copy()

        if i > 5 and i % 10 == 0:
            tol   = 0.5
            match = (abs(imu_accel[0] - expected_imu[0]) < tol and
                     abs(imu_accel[2] - expected_imu[2]) < tol)
            status = "OK" if match else "FAIL"

            print(f"{i:<5} | {'IMU Raw':<10} | {imu_accel[0]:10.4f} | {imu_accel[2]:10.4f} |")
            print(f"{'':<5} | {'GT (kin)':<10} | {gt_kinematic_accel[0]:10.4f} | {gt_kinematic_accel[2]:10.4f} |  <- true kinematic accel")
            print(f"{'':<5} | {'GT+Grav':<10} | {expected_imu[0]:10.4f} | {expected_imu[2]:10.4f} |  <- expected IMU reading")
            print(f"{'':<5} | {'KF Pred':<10} | {kf_accel[0]:10.4f} | {kf_accel[2]:10.4f} |  <- Kalman filter estimate")
            print(f"{'':<5} | {'Diff':<10} | {abs(imu_accel[0]-expected_imu[0]):10.4f} | {abs(imu_accel[2]-expected_imu[2]):10.4f} | {status}")
            print("-" * 70)


if __name__ == "__main__":
    test_imu_logic()
