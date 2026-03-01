import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from model import DeepHydroMLP

# ================= 配置区域 =================
MODEL_PATH = '../models/best_hydro_model.pth'
SCALER_DIR = '../models/'
CSV_PATH = '/home/ray/Disk_ext/DeepSim_RL/hydro_MLP/data/phy_processed/mission_log_processed_Physics.csv' # 原始数据路径
DEVICE = torch.device("cpu") # 评估用 CPU 足够，且方便

# 选择一段连续的数据进行可视化 (例如：取 CSV 中间的一段)
# 200 个点通常能看清波形，不要超过 500
VISUALIZE_START_IDX = 290000 
VISUALIZE_LEN = 300 
# ===========================================

def load_and_prep():
    print("1. Loading resources...")
    # 加载 Scalers
    scaler_X = joblib.load(os.path.join(SCALER_DIR, 'scaler_X.pkl'))
    scaler_Y = joblib.load(os.path.join(SCALER_DIR, 'scaler_Y.pkl'))
    
    # 加载模型
    model = DeepHydroMLP(input_dim=12, output_dim=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    return model, scaler_X, scaler_Y

def get_contiguous_data(csv_path, start_idx, length):
    print(f"2. Extracting contiguous data slice [{start_idx}:{start_idx+length}]...")
    df = pd.read_csv(csv_path)
    
    # 定义列名
    input_cols = [
        'u(m/s)', 'v(m/s)', 'w(m/s)', 'p(rad/s)', 'q(rad/s)', 'r(rad/s)',
        'Accel_u(m/s2)', 'Accel_v(m/s2)', 'Accel_w(m/s2)', 
        'Accel_p(m/s2)', 'Accel_q(m/s2)', 'Accel_r(m/s2)'
    ]
    target_cols = [
        'F_Fluid_u(N)', 'F_Fluid_v(N)', 'F_Fluid_w(N)', 
        'F_Fluid_p(N)', 'F_Fluid_q(N)', 'F_Fluid_r(N)'
    ]
    
    # 截取切片
    df_slice = df.iloc[start_idx : start_idx + length]
    
    X_raw = df_slice[input_cols].values.astype(np.float32)
    Y_raw = df_slice[target_cols].values.astype(np.float32)
    
    return X_raw, Y_raw

def main():
    # 1. 准备
    model, scaler_X, scaler_Y = load_and_prep()
    
    # 2. 获取连续数据
    X_raw, Y_true = get_contiguous_data(CSV_PATH, VISUALIZE_START_IDX, VISUALIZE_LEN)
    
    # 3. 预测
    print("3. Running Inference...")
    # 归一化输入
    X_scaled = scaler_X.transform(X_raw)
    X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
    
    # 模型推理
    with torch.no_grad():
        Y_pred_scaled = model(X_tensor).numpy()
    
    # 反归一化输出 (还原为真实物理量)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    
    # ===========================================
    # 4. 数值评估 (Metrics)
    # ===========================================
    print("\n" + "="*50)
    print(f"{'DOF':<15} | {'R2 Score (拟合度)':<15} | {'MAE (平均误差)':<15} | {'Max Error':<15}")
    print("-" * 70)
    
    dof_names = ['Fx (Surge)', 'Fy (Sway)', 'Fz (Heave)', 'Tx (Roll)', 'Ty (Pitch)', 'Tz (Yaw)']
    units     = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']
    
    for i in range(6):
        # R2: 1.0 是完美，0.0 是瞎猜，负数是比瞎猜还烂
        r2 = r2_score(Y_true[:, i], Y_pred[:, i])
        
        # MAE: 平均绝对误差
        mae = mean_absolute_error(Y_true[:, i], Y_pred[:, i])
        
        # Max Error: 最大误差
        max_err = np.max(np.abs(Y_true[:, i] - Y_pred[:, i]))
        
        print(f"{dof_names[i]:<15} | {r2:>15.4f} | {mae:>10.4f} {units[i]} | {max_err:>10.4f} {units[i]}")
    print("="*50 + "\n")

    # ===========================================
    # 5. 可视化 (Visualization)
    # ===========================================
    print("4. Plotting detailed comparison...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12)) # 3行2列
    axes = axes.flatten()
    
    time_steps = np.arange(VISUALIZE_LEN)
    
    for i in range(6):
        ax = axes[i]
        
        # 画真值 (用实线)
        ax.plot(time_steps, Y_true[:, i], 'k-', linewidth=2.5, alpha=0.6, label='Ground Truth')
        
        # 画预测 (用虚线 + 圆点，这样能看清是否重合)
        ax.plot(time_steps, Y_pred[:, i], 'r--o', markersize=3, linewidth=1.5, label='AI Prediction')
        
        # 画误差区域 (填充颜色，直观看到哪里误差大)
        ax.fill_between(time_steps, Y_true[:, i], Y_pred[:, i], color='gray', alpha=0.2, label='Error Gap')
        
        ax.set_title(f"{dof_names[i]} Comparison", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"Force/Torque ({units[i]})")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 只在第一张图显示图例，避免遮挡
        if i == 0:
            ax.legend(loc='upper right')

    plt.suptitle(f"Deep Hydro-Sim Evaluation (Slice: {VISUALIZE_START_IDX}-{VISUALIZE_START_IDX+VISUALIZE_LEN})", fontsize=16)
    plt.tight_layout()
    
    save_file = '../logs/evaluation_detail.png'
    plt.savefig(save_file, dpi=300) # 高清保存
    print(f"HD Plot saved to {save_file}")
    plt.show()

if __name__ == '__main__':
    main()