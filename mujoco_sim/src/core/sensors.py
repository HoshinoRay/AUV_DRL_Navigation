import mujoco
import numpy as np


class SensorManager:
    """
    Reads and injects synthetic noise into all onboard sensors.

    Sensor suite
    ------------
    - Depth (pressure-derived)
    - DVL  (3-axis Doppler velocity log, with simulated packet dropout)
    - IMU  (accelerometer + gyroscope)
    - Sonar array (15 rangefinder beams for obstacle avoidance)
    - Altimeter (downward-facing rangefinder)
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "yuyuan")

        # Z coordinate of the water surface in world frame
        self.WATER_SURFACE_Z = 15.0

        # Sensor names must match the MuJoCo XML exactly
        self._imu_names = ["accel", "gyro", "dvl"]

        self._sonar_beam_names = [
            "rf_L60", "rf_L50", "rf_L40", "rf_L30", "rf_L20", "rf_L10",
            "rf_C",
            "rf_R10", "rf_R20", "rf_R30", "rf_R40", "rf_R50", "rf_R60",
            "rf_Up30", "rf_Down30",
        ]
        self._altimeter_name = "altimeter"

        # Pre-cache sensor addresses for fast per-step access
        self._sensor_adrs = {}
        for name in self._imu_names:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid != -1:
                self._sensor_adrs[name] = model.sensor_adr[sid]
            else:
                print(f"[SensorManager] Warning: sensor '{name}' not found in XML.")

        self._sonar_adrs = []
        for name in self._sonar_beam_names:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid != -1:
                self._sonar_adrs.append(model.sensor_adr[sid])
            else:
                print(f"[SensorManager] Warning: sonar beam '{name}' not found.")
                self._sonar_adrs.append(None)

        alt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, self._altimeter_name)
        self._alt_adr = model.sensor_adr[alt_id] if alt_id != -1 else None

        # Noise / dropout configuration (matches expected Sim2Real gap)
        self.config = {
            'depth_noise':   0.02,   # m
            'dvl_noise':     0.05,   # m/s
            'dvl_dropout':   0.01,   # packet-loss probability
            'accel_noise':   0.15,   # m/s²
            'gyro_noise':    0.01,   # rad/s
            'sonar_noise':   0.05,   # m
            'sonar_max_dist': 12.0,  # must match XML cutoff
            'alt_max_dist':  50.0,   # m
        }

    def get_raw_data(self) -> dict:
        """
        Return a dict of noisy sensor readings:
            depth    : float  (m below surface)
            dvl      : (3,)   body-frame velocity (m/s)
            accel    : (3,)   body-frame acceleration (m/s²)
            gyro     : (3,)   body-frame angular velocity (rad/s)
            sonar    : (15,)  range readings (m)
            altitude : float  (m above seabed)
            quat     : (4,)   orientation quaternion [w, x, y, z] (ground truth)
        """
        # Depth
        real_z = self.data.xpos[self.body_id][2]
        depth = (self.WATER_SURFACE_Z - real_z) + np.random.normal(0, self.config['depth_noise'])

        # DVL
        if "dvl" in self._sensor_adrs:
            adr = self._sensor_adrs["dvl"]
            dvl_vel = self.data.sensordata[adr:adr + 3].copy()
            if np.random.random() < self.config['dvl_dropout']:
                dvl_vel[:] = 0.0
            else:
                dvl_vel += np.random.normal(0, self.config['dvl_noise'], 3)
        else:
            dvl_vel = np.zeros(3)

        # Accelerometer
        if "accel" in self._sensor_adrs:
            adr = self._sensor_adrs["accel"]
            accel = self.data.sensordata[adr:adr + 3].copy()
            accel += np.random.normal(0, self.config['accel_noise'], 3)
        else:
            accel = np.zeros(3)

        # Gyroscope
        if "gyro" in self._sensor_adrs:
            adr = self._sensor_adrs["gyro"]
            gyro = self.data.sensordata[adr:adr + 3].copy()
            gyro += np.random.normal(0, self.config['gyro_noise'], 3)
        else:
            gyro = np.zeros(3)

        # Sonar array
        max_dist = self.config['sonar_max_dist']
        sonar_readings = []
        for adr in self._sonar_adrs:
            val = max_dist if adr is None else self.data.sensordata[adr]
            if val < 0:   # MuJoCo returns -1 for out-of-range
                val = max_dist
            val = np.clip(val + np.random.normal(0, self.config['sonar_noise']), 0, max_dist)
            sonar_readings.append(val)
        sonar_array = np.array(sonar_readings, dtype=np.float32)

        # Altimeter
        if self._alt_adr is not None:
            alt_val = self.data.sensordata[self._alt_adr]
            if alt_val < 0:
                alt_val = self.config['alt_max_dist']
            alt_val = np.clip(
                alt_val + np.random.normal(0, self.config['sonar_noise']),
                0, self.config['alt_max_dist'],
            )
        else:
            alt_val = 0.0

        return {
            "depth":    depth,
            "dvl":      dvl_vel,
            "accel":    accel,
            "gyro":     gyro,
            "sonar":    sonar_array,
            "altitude": alt_val,
            "quat":     self.data.qpos[3:7].copy(),
        }
