import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# ==========================================
# 1. 基础容器类 (只负责存数和取数)
# ==========================================
class HydroDataset(Dataset):
    def __init__(self, X, Y):
        """
        X: 归一化后的输入特征 (numpy array)
        Y: 归一化后的标签 (numpy array)
        """
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==========================================
# 2. 数据管理器类 (负责洗牌、切分、预处理)
# ==========================================
class HydroDataManager:
    def __init__(self, csv_path, save_dir='../models/'):
        self.csv_path = csv_path
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 定义列名 (硬编码以确保安全)
        self.input_cols = [
            'u(m/s)', 'v(m/s)', 'w(m/s)', 'p(rad/s)', 'q(rad/s)', 'r(rad/s)',
            'Accel_u(m/s2)', 'Accel_v(m/s2)', 'Accel_w(m/s2)', 
            'Accel_p(m/s2)', 'Accel_q(m/s2)', 'Accel_r(m/s2)'
        ]
        self.target_cols = [
            'F_Fluid_u(N)', 'F_Fluid_v(N)', 'F_Fluid_w(N)', 
            'F_Fluid_p(N)', 'F_Fluid_q(N)', 'F_Fluid_r(N)'
        ]

    def get_dataloaders(self, batch_size=256, val_split=0.1, test_split=0.05):
        """
        核心函数：读取CSV -> 洗牌 -> 切分 -> 归一化 -> 封装DataLoader
        """
        print(f"[Data Manager] Loading data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        
        # 1. 检查 NaN/Inf (清洗数据)
        if df.isnull().values.any():
            print("[Warning] NaN detected! Dropping rows...")
            df = df.dropna()
            
        # 提取原始数据
        X_raw = df[self.input_cols].values.astype(np.float32)
        Y_raw = df[self.target_cols].values.astype(np.float32)
        
        print(f"[Data Manager] Total samples: {len(X_raw)}")

        # 2. 全局打乱与切分 (Global Shuffle & Split)
        # 这里的 random_state=42 保证了每次运行切分结果一致 (可复现性)
        # 我们先切出 Train+Val 和 Test
        X_temp, X_test, Y_temp, Y_test = train_test_split(
            X_raw, Y_raw, test_size=test_split, random_state=42, shuffle=True
        )
        
        # 再从 Temp 中切出 Train 和 Val
        # 注意：这里的 val_split 是相对于剩余数据的比例
        adjusted_val_split = val_split / (1.0 - test_split)
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_temp, Y_temp, test_size=adjusted_val_split, random_state=42, shuffle=True
        )

        print(f"[Data Manager] Split Result -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # 3. 归一化 (Normalization)
        # 关键原则：Scaler 只能在 Train 集上拟合！防止信息泄露到验证集。
        print("[Data Manager] Fitting scalers on Training set...")
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        Y_train_scaled = scaler_Y.fit_transform(Y_train)
        
        # 应用到 Val 和 Test
        X_val_scaled = scaler_X.transform(X_val)
        Y_val_scaled = scaler_Y.transform(Y_val)
        X_test_scaled = scaler_X.transform(X_test)
        Y_test_scaled = scaler_Y.transform(Y_test)

        # 4. 保存 Scaler (Sim2Real 时必须用这个)
        joblib.dump(scaler_X, os.path.join(self.save_dir, 'scaler_X.pkl'))
        joblib.dump(scaler_Y, os.path.join(self.save_dir, 'scaler_Y.pkl'))
        print(f"[Data Manager] Scalers saved to {self.save_dir}")

        # 5. 封装 Dataset 和 DataLoader
        train_ds = HydroDataset(X_train_scaled, Y_train_scaled)
        val_ds   = HydroDataset(X_val_scaled, Y_val_scaled)
        test_ds  = HydroDataset(X_test_scaled, Y_test_scaled) # 测试集留着最后评估用

        # pin_memory=True 加速 CPU 到 GPU 传输
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # Test loader 不需要 shuffle
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        return train_loader, val_loader, test_loader