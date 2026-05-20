"""
Sensor and hydrodynamics integration test.

Verifies added mass effect in heave, torque accumulation, and static sensor outputs.

Usage:
    cd mujoco_sim
    python src/utils/verify_sensors.py
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


def test_sensor_manager():
    model   = mujoco.MjModel.from_xml_path(XML_PATH)
    data    = mujoco.MjData(model)
    dt      = model.opt.timestep
    robot   = YuyuanRobot(model, data)
    sensors = SensorManager(model, data)
    hydro   = HydroDynamicsPlugin(
        MODEL_WEIGHTS['mlp'], MODEL_WEIGHTS['scaler_x'], MODEL_WEIGHTS['scaler_y'], dt
    )

    # --- Test 1: Heave added mass ---
    print("\n=== Test 1: Heave Acceleration (Added Mass Check) ===")
    mujoco.mj_resetData(model, data)
    test_force_heave = 3000.0
    print(f"  Applying {test_force_heave} N heave force for 20 steps ...")

    for _ in range(20):
        data.xfrc_applied[:] = 0
        hydro.apply_hydrodynamics(robot)
        data.xfrc_applied[robot.body_id][2] += (-test_force_heave)
        mujoco.mj_step(model, data)

    accel_z = sensors.get_imu_data()[0][2]
    print(f"  Observed heave accel: {accel_z:.4f} m/s^2")
    if accel_z < 12.0:
        print("  PASS: added mass detected in heave (acceleration is damped).")
    else:
        print(f"  FAIL: accel too high ({accel_z:.2f}) — added mass not applied.")

    # --- Test 2: Torque accumulation ---
    print("\n=== Test 2: Torque Accumulation ===")
    mujoco.mj_resetData(model, data)
    test_torque = 1000.0
    print(f"  Applying {test_torque} Nm yaw torque ...")

    for i in range(101):
        data.xfrc_applied[:] = 0
        hydro.apply_hydrodynamics(robot)
        data.xfrc_applied[robot.body_id][5] += test_torque
        mujoco.mj_step(model, data)

        if i % 20 == 0:
            _, gyro = sensors.get_imu_data()
            print(f"    step {i:3d}: t={data.time:.3f}s  gyro_z={gyro[2]:.4f} rad/s")

    _, final_gyro = sensors.get_imu_data()
    if abs(final_gyro[2]) > 0.1:
        print("  PASS: angular velocity accumulated successfully.")
    else:
        print("  FAIL: angular velocity did not accumulate — check torque logic.")

    # --- Test 3: Static sensors ---
    print("\n=== Test 3: Static Sensors ===")
    print(f"  True Z (world frame):  {data.qpos[2]:.4f} m")
    print(f"  Depth reading:         {sensors.get_depth():.4f} m")
    print(f"  Sonar [front, down]:   {sensors.get_sonar_dist()}")

    print("\n=== Sensor tests complete ===")


if __name__ == "__main__":
    test_sensor_manager()
