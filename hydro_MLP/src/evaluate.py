import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from model import DeepHydroMLP # 必须导入模型结构！
from dataset import HydroDataManager # 用来加载测试集

# ================= 配置 =================
MODEL_PATH = '../models/best_hydro_model.pth'
SCALER_DIR = '../models/'
CSV_PATH = '../data/phy_processed/mission_log_processed_Physics.csv' # 你的数据路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print(f"Evaluating on {DEVICE}...")

    # 1. 加载“翻译官” (Scalers)
    print("Loading scalers...")
    scaler_X = joblib.load(os.path.join(SCALER_DIR, 'scaler_X.pkl'))
    scaler_Y = joblib.load(os.path.join(SCALER_DIR, 'scaler_Y.pkl'))

    # 2. 加载“肉体”和“灵魂” (Model)
    model = DeepHydroMLP(input_dim=12, output_dim=6).to(DEVICE)
    # map_location确保即使在GPU训练的模型也能在CPU上跑
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # 切换到评估模式（关闭Dropout等）

    # 3. 获取测试数据 (利用我们之前写的Manager)
    # 我们只取最后 5% 的数据作为测试，保证这些数据模型从未见过
    data_manager = HydroDataManager(CSV_PATH, save_dir=SCALER_DIR)
    _, _, test_loader = data_manager.get_dataloaders(batch_size=1000, test_split=0.05)
    
    # 4. 批量预测
    all_preds = []
    all_targets = []
    
    print("Running inference...")
    with torch.no_grad(): # 不计算梯度，省内存
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            # 存下来用于画图
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    # 拼接所有批次
    pred_scaled = np.concatenate(all_preds, axis=0)
    true_scaled = np.concatenate(all_targets, axis=0)

    # 5. 反归一化 (将 0.1, -0.5 这种数字还原成真实的 牛顿 N)
    pred_real = scaler_Y.inverse_transform(pred_scaled)
    true_real = scaler_Y.inverse_transform(true_scaled)

    # 计算误差 (MSE)
    mse = np.mean((pred_real - true_real) ** 2)
    print(f"Test MSE Loss (Physical Scale): {mse:.4f}")

    # 6. 画图 (取前 500 个点展示，太多了看不清)
    plot_len = 500
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    labels = ['Fx (Surge)', 'Fy (Sway)', 'Fz (Heave)', 'Tx (Roll)', 'Ty (Pitch)', 'Tz (Yaw)']

    for i in range(6):
        ax = axes[i]
        ax.plot(true_real[:plot_len, i], 'k-', label='Ground Truth', linewidth=1.5, alpha=0.7)
        ax.plot(pred_real[:plot_len, i], 'r--', label='AI Prediction', linewidth=1.5)
        ax.set_title(labels[i])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    save_path = '../logs/evaluation_result.png'
    plt.savefig(save_path)
    print(f"Result plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    evaluate()