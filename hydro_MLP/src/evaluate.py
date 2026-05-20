"""
Evaluate the trained DeepHydroMLP on the held-out test split.

Usage (from hydro_MLP/):
    python src/evaluate.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import HydroDataManager
from model import DeepHydroMLP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = '../models/best_hydro_model.pth'
SCALER_DIR = '../models/'
CSV_PATH = '../data/phy_processed/mission_log_processed_Physics.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLOT_LEN = 500  # number of consecutive samples to visualise
# ---------------------------------------------------------------------------


def evaluate():
    print(f"Evaluating on: {DEVICE}")

    scaler_X = joblib.load(os.path.join(SCALER_DIR, 'scaler_X.pkl'))
    scaler_Y = joblib.load(os.path.join(SCALER_DIR, 'scaler_Y.pkl'))

    model = DeepHydroMLP(input_dim=12, output_dim=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    data_manager = HydroDataManager(CSV_PATH, save_dir=SCALER_DIR)
    _, _, test_loader = data_manager.get_dataloaders(batch_size=1000, test_split=0.05)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(DEVICE))
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    pred_scaled = np.concatenate(all_preds, axis=0)
    true_scaled = np.concatenate(all_targets, axis=0)

    pred_real = scaler_Y.inverse_transform(pred_scaled)
    true_real = scaler_Y.inverse_transform(true_scaled)

    mse = np.mean((pred_real - true_real) ** 2)
    print(f"Test MSE (physical scale): {mse:.4f}")

    labels = ['Fx (Surge)', 'Fy (Sway)', 'Fz (Heave)', 'Tx (Roll)', 'Ty (Pitch)', 'Tz (Yaw)']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(true_real[:PLOT_LEN, i], 'k-', label='Ground Truth', linewidth=1.5, alpha=0.7)
        ax.plot(pred_real[:PLOT_LEN, i], 'r--', label='Prediction', linewidth=1.5)
        ax.set_title(labels[i])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    os.makedirs('../logs', exist_ok=True)
    save_path = '../logs/evaluation_result.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    evaluate()
