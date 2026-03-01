import numpy as np
from src.utils.kalman_filter import KalmanFilter6D 
import torch
from src.core.models import DeepHydroMLP 
import joblib

class HydroInference:
    def __init__(self, model_path, scaler_x_path, scaler_y_path, device='cpu'):
        self.device = torch.device(device)
        
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.model = DeepHydroMLP(input_dim=12, output_dim=6).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval() 
        # MuJoCo (ENU) <-> Stonefish (NED)
        # Scequence: [Surge, Sway, Heave, Roll, Pitch, Yaw]
        self.coord_mask = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])

    def predict(self, velocity, acceleration, gain=-1.0):
        # ENU --> NED 
        vel_ned = velocity * self.coord_mask
        acc_ned = acceleration * self.coord_mask

        MAX_LIN_VEL = 1.3  # m/s
        MAX_ANG_VEL = 1.0  # rad/s 
        
        vel_ned[:3] = np.clip(vel_ned[:3], -MAX_LIN_VEL, MAX_LIN_VEL)
        vel_ned[3:] = np.clip(vel_ned[3:], -MAX_ANG_VEL, MAX_ANG_VEL)

        # 拼接与推理
        input_vector = np.concatenate([vel_ned, acc_ned]).reshape(1, -1)
        input_scaled = self.scaler_x.transform(input_vector)
        with torch.no_grad():
            tensor_in = torch.tensor(input_scaled, dtype=torch.float32).to(self.device)
            tensor_out = self.model(tensor_in)
            output_scaled = tensor_out.cpu().numpy()

        # 反归一化 (Scaler Y) -> 得到真实的力
        output_force = self.scaler_y.inverse_transform(output_scaled)

        force_mujoco = output_force * self.coord_mask 
        final_force = force_mujoco * gain
        final_force = np.clip(final_force, -180.0, 180.0)

        return final_force.flatten()
    
class HydroDynamicsPlugin:
    def __init__(self, model_path, scaler_x, scaler_y, dt, simplified_mode=True):
        self.dt = dt # 保存 dt
        self.process_noise = 10.0
        self.measure_noise = 0.001
        self.simplified_mode = simplified_mode 

        # 初始化推理引擎
        self.predictor = HydroInference(model_path, scaler_x, scaler_y)

        # 初始化状态 (提取出来放到 reset 里)
        self.reset()

        # 物理常量
        self.MA = np.array([0.0, 102.31, 121.68, 0.0, 0.0, 0.0])
        self.BUOYANCY_MAGNITUDE = 1790.0
        self.VEL_DEADBAND = 0.05

    def reset(self):
        """重置内部滤波器和噪声状态"""
        # 重置卡尔曼滤波，防止上一局的加速度污染下一局
        self.kf = KalmanFilter6D(dt=self.dt, process_noise=self.process_noise, measure_noise=self.measure_noise)
        
        # 重置噪声生成器的状态
        self.surge_noise_state = 0.0
        self.noise_smoothing = 0.1 # 越小越平滑

    def apply_hydrodynamics(self, robot):
        data = robot.data
        model = robot.model
        
        # 1. 获取基础状态 (无论什么模式都需要)
        current_vel_body, rot_mat = robot.get_body_state()
        accel_body_filtered = self.kf.update(current_vel_body)

        # 初始化力矢量
        hydro_force_body = np.zeros(6)
        total_hydro_body = np.zeros(6)

        # 2. 根据模式决定是否计算水阻力
        if self.simplified_mode:
            # 简化模式：阻力和附加质量力保持为0
            # 仅保留附加质量力，依然关闭水阻力 (MLP阻力和Surge阻力)
            added_mass_force_body = -1.0 * self.MA * accel_body_filtered
            total_hydro_body = added_mass_force_body
            
        else:
            # --- 完整物理模式 (原来的逻辑全部移到这里) ---
            
            # A. MLP 生成全自由度基础力
            hydro_force_body = self.predictor.predict(current_vel_body, accel_body_filtered, gain=-1.0)

            # B. 手动覆盖 Surge (Index 0)
            u = current_vel_body[0]
            k_linear, k_quad = 30.0, 12.0
            base_drag_mag = k_linear * abs(u) + k_quad * (u**2)

            # 平滑噪声
            raw_noise = np.random.uniform(-35.0, 35.0)
            self.surge_noise_state = (1 - self.noise_smoothing) * self.surge_noise_state + \
                                     self.noise_smoothing * raw_noise
            
            total_mag = np.clip(base_drag_mag + self.surge_noise_state, 0, 200.0)
            
            if abs(u) > 1e-3:
                hydro_force_body[0] = -1.0 * np.sign(u) * total_mag
            else:
                hydro_force_body[0] = 0.0

            # C. 死区处理 
            for i in range(6):
                vel_abs = abs(current_vel_body[i])
                if vel_abs < self.VEL_DEADBAND:
                    hydro_force_body[i] *= (vel_abs / self.VEL_DEADBAND)

            # D. 叠加附加质量力 (Added Mass)
            added_mass_force_body = -1.0 * self.MA * accel_body_filtered
            total_hydro_body = 0.7 * hydro_force_body + added_mass_force_body

        # 3. 坐标转换与浮力计算 (浮力无论如何都要保留，否则会沉底)
        f_hydro_world = rot_mat @ total_hydro_body[:3]
        t_hydro_world = rot_mat @ total_hydro_body[3:]

        pos_com, pos_cob = robot.get_world_pose()
        f_buoyancy_world = np.array([0.0, 0.0, self.BUOYANCY_MAGNITUDE])
        t_buoyancy_world = np.cross(pos_cob - pos_com, f_buoyancy_world)

        # 4. 施加力到 MuJoCo 模型
        # 使用 = 覆盖之前步骤可能残留的力
        data.xfrc_applied[robot.body_id] = np.concatenate([
            f_buoyancy_world + f_hydro_world, 
            t_buoyancy_world + t_hydro_world
        ])
        
        return current_vel_body, hydro_force_body, total_hydro_body