import numpy as np

class KalmanFilter6D:
    def __init__(self, dt, process_noise=0.1, measure_noise=0.01):
        """
        dt: 时间步长 see xml
        process_noise (Q): 多大程度上允许加速度发生突变 越大越灵敏，越小越平滑。
        measure_noise (R): MuJoCo的速度数据有多大噪声(仿真里通常很小)
        """
        self.dt = dt
        
        # 状态向量 x: [6, 2] -> 6个自由度，每个有[vel, accel]
        self.x = np.zeros((6, 2)) 
        
        # 协方差矩阵 P: [6, 2, 2] -> 估计的不确定性 (初始化为单位阵)
        self.P = np.array([np.eye(2) for _ in range(6)])
        
        # 状态转移矩阵 F: [[1, dt], [0, 1]]
        self.F = np.array([[1, dt], [0, 1]])
        
        # 观测矩阵 H: [[1, 0]] -> 我们只测量速度
        self.H = np.array([1, 0])
        
        # 过程噪声协方差 Q
        # 这里假设加速度的变化是主要噪声源
        self.Q = np.array([
            [0.25 * dt**4, 0.5 * dt**3],
            [0.5 * dt**3,    dt**2]
        ]) * process_noise
        
        # 测量噪声 R
        self.R = measure_noise

    def update(self, v_meas_batch):
        """
        输入: v_meas_batch (6维向量: [u, v, w, p, q, r])
        输出: 滤波后的加速度 (6维向量)
        """
        estimated_accel = np.zeros(6)
        
        for i in range(6):
            z = v_meas_batch[i] # 当前轴的测量速度
            
            # --- 1. 预测 (Predict) ---
            # x_pred = F * x
            x_pred = self.F @ self.x[i]
            # P_pred = F * P * F.T + Q
            P_pred = self.F @ self.P[i] @ self.F.T + self.Q
            
            # --- 2. 更新 (Update) ---
            # 计算卡尔曼增益 K
            # S = H * P * H.T + R
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T * (1/S) # 标量除法
            
            # 更新状态 x = x_pred + K * (z - H * x_pred)
            y = z - self.H @ x_pred # 残差 (测量值 - 预测值)
            self.x[i] = x_pred + K * y
            
            # 更新协方差 P = (I - K * H) * P_pred
            self.P[i] = (np.eye(2) - np.outer(K, self.H)) @ P_pred
            
            # 提取加速度 (状态向量的第二个元素)
            estimated_accel[i] = self.x[i][1]
            
        return estimated_accel