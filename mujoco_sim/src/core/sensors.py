import numpy as np
import mujoco

class SensorManager:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "yuyuan")
        
        # 1. 物理环境参数
        # 如果你的 XML 里由 worldbody 决定水面位置，通常 Z=0 或 Z=10
        # 这里假设 Z=0 是水面，往下是负坐标
        self.WATER_SURFACE_Z = 15.0 
        
        # 2. 传感器名称定义 (必须与 XML 完全一致)
        self.imu_names = ["accel", "gyro", "dvl"]
        
        # 定义避障声呐组 (顺序很重要！建议从左到右，方便后续神经网络理解空间关系)
        self.sonar_beam_names = [
            # 左侧 (Left)
            "rf_L60", "rf_L50", "rf_L40", "rf_L30", "rf_L20", "rf_L10",
            # 中间 (Center)
            "rf_C",
            # 右侧 (Right)
            "rf_R10", "rf_R20", "rf_R30", "rf_R40", "rf_R50", "rf_R60",
            # 垂直 (Vertical)
            "rf_Up30", "rf_Down30"
        ]
        
        # 独立的高度计 (垂直向下)
        self.altimeter_name = "altimeter"

        # 3. 预先获取 ID 和 Address (以此提高 step 运行时的速度)
        self.sensor_adrs = {}
        
        # A. 注册 IMU & DVL
        for name in self.imu_names:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid != -1:
                self.sensor_adrs[name] = model.sensor_adr[sid]
            else:
                print(f"[Warning] Sensor {name} not found in XML!")

        # B. 注册声呐阵列
        self.sonar_beam_adrs = []
        for name in self.sonar_beam_names:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid != -1:
                self.sonar_beam_adrs.append(model.sensor_adr[sid])
            else:
                # 如果没找到，给个 None，防止报错崩溃，但会有警告
                print(f"[Warning] Sonar Beam {name} not found!")
                self.sonar_beam_adrs.append(None)
                
        # C. 注册高度计
        alt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, self.altimeter_name)
        self.alt_adr = model.sensor_adr[alt_id] if alt_id != -1 else None

        # 4. 噪声配置 (Sim2Real Gap)
        self.config = {
            'depth_noise': 0.02,   # 深度计误差 (m)
            'dvl_noise': 0.05,     # 速度计误差 (m/s)
            'dvl_dropout': 0.01,   # 速度计丢包概率
            'accel_noise': 0.15,   # 加速度计白噪声
            'gyro_noise': 0.01,    # 陀螺仪白噪声
            'sonar_noise': 0.05,   # 声呐测距误差 (m)
            'sonar_max_dist': 12.0,# 必须与 XML cutoff 一致
            'alt_max_dist': 50.0   # 高度计最大量程
        }

    def get_raw_data(self):
        """
        获取带噪声的传感器数据字典
        """
        # ---------------------------------------------------
        # A. 深度计 (Depth Sensor)
        # ---------------------------------------------------
        real_z = self.data.xpos[self.body_id][2]
        depth = self.WATER_SURFACE_Z - real_z
        # 添加噪声
        depth += np.random.normal(0, self.config['depth_noise'])

        # ---------------------------------------------------
        # B. DVL (多普勒速度计)
        # ---------------------------------------------------
        if "dvl" in self.sensor_adrs:
            adr = self.sensor_adrs["dvl"]
            # MuJoCo 速度计返回 3维数据 [vx, vy, vz] (Local frame)
            dvl_vel = self.data.sensordata[adr:adr+3].copy()
            
            # 模拟 DVL 偶尔丢包的情况 (读数为0)
            if np.random.random() < self.config['dvl_dropout']:
                dvl_vel[:] = 0.0
            else:
                dvl_vel += np.random.normal(0, self.config['dvl_noise'], 3)
        else:
            dvl_vel = np.zeros(3)

        # ---------------------------------------------------
        # C. IMU (加速度计 + 陀螺仪)
        # ---------------------------------------------------
        # Accel
        if "accel" in self.sensor_adrs:
            adr = self.sensor_adrs["accel"]
            accel = self.data.sensordata[adr:adr+3].copy()
            accel += np.random.normal(0, self.config['accel_noise'], 3)
        else:
            accel = np.zeros(3)
            
        # Gyro
        if "gyro" in self.sensor_adrs:
            adr = self.sensor_adrs["gyro"]
            gyro = self.data.sensordata[adr:adr+3].copy()
            gyro += np.random.normal(0, self.config['gyro_noise'], 3)
        else:
            gyro = np.zeros(3)

        # ---------------------------------------------------
        # D. 高密度声呐阵列 (15 Beams)
        # ---------------------------------------------------
        sonar_readings = []
        max_dist = self.config['sonar_max_dist']
        
        for adr in self.sonar_beam_adrs:
            if adr is None:
                val = max_dist
            else:
                val = self.data.sensordata[adr]
            
            # 处理 MuJoCo 的空值: -1 通常代表未检测到 (Infinite/Cutoff)
            if val < 0:
                val = max_dist
            
            # 添加噪声 (距离越远误差通常越大，这里简化为固定高斯噪声)
            noise = np.random.normal(0, self.config['sonar_noise'])
            val = np.clip(val + noise, 0, max_dist)
            
            sonar_readings.append(val)

        # 转为 numpy 数组 (Shape: [15,])
        sonar_array = np.array(sonar_readings, dtype=np.float32)

        # ---------------------------------------------------
        # E. 高度计 (Altimeter)
        # ---------------------------------------------------
        if self.alt_adr is not None:
            alt_val = self.data.sensordata[self.alt_adr]
            if alt_val < 0: 
                alt_val = self.config['alt_max_dist']
            alt_val += np.random.normal(0, self.config['sonar_noise'])
            alt_val = np.clip(alt_val, 0, self.config['alt_max_dist'])
        else:
            alt_val = 0.0

        # ---------------------------------------------------
        # 返回打包数据
        # ---------------------------------------------------
        return {
            "depth": depth,       # scalar (m)
            "dvl": dvl_vel,       # [3,] (m/s)
            "accel": accel,       # [3,] (m/s^2)
            "gyro": gyro,         # [3,] (rad/s)
            "sonar": sonar_array, # [15,] (m) - 避障核心数据
            "altitude": alt_val,  # scalar (m) - 离底高度
            "quat": self.data.qpos[3:7].copy() # [4,] (w,x,y,z) - 姿态真值
        }