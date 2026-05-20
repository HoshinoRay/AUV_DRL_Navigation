"""
Detailed per-DOF evaluation with R² / MAE / max-error metrics and
a high-resolution comparison plot on a contiguous data slice.

Usage (from hydro_MLP/):
    python src/evaluate_detail.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from model import DeepHydroMLP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = '../models/best_hydro_model.pth'
SCALER_DIR = '../models/'
CSV_PATH = '../data/phy_processed/mission_log_processed_Physics.csv'
DEVICE = torch.device('cpu')

# Contiguous slice to visualise (index range within the CSV)
VISUALIZE_START_IDX = 290000
VISUALIZE_LEN = 300
# ---------------------------------------------------------------------------

INPUT_COLS = [
    'u(m/s)', 'v(m/s)', 'w(m/s)', 'p(rad/s)', 'q(rad/s)', 'r(rad/s)',
    'Accel_u(m/s2)', 'Accel_v(m/s2)', 'Accel_w(m/s2)',
    'Accel_p(m/s2)', 'Accel_q(m/s2)', 'Accel_r(m/s2)',
]
TARGET_COLS = [
    'F_Fluid_u(N)', 'F_Fluid_v(N)', 'F_Fluid_w(N)',
    'F_Fluid_p(N)', 'F_Fluid_q(N)', 'F_Fluid_r(N)',
]
DOF_NAMES = ['Fx (Surge)', 'Fy (Sway)', 'Fz (Heave)', 'Tx (Roll)', 'Ty (Pitch)', 'Tz (Yaw)']
UNITS = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']


def main():
    # Load resources
    scaler_X = joblib.load(os.path.join(SCALER_DIR, 'scaler_X.pkl'))
    scaler_Y = joblib.load(os.path.join(SCALER_DIR, 'scaler_Y.pkl'))

    model = DeepHydroMLP(input_dim=12, output_dim=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Extract a contiguous slice from the CSV
    df = pd.read_csv(CSV_PATH)
    df_slice = df.iloc[VISUALIZE_START_IDX: VISUALIZE_START_IDX + VISUALIZE_LEN]
    X_raw = df_slice[INPUT_COLS].values.astype(np.float32)
    Y_true = df_slice[TARGET_COLS].values.astype(np.float32)

    # Inference
    X_scaled = scaler_X.transform(X_raw)
    with torch.no_grad():
        Y_pred_scaled = model(torch.FloatTensor(X_scaled)).numpy()
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

    # Metrics
    print(f"\n{'DOF':<15} | {'R2':>10} | {'MAE':>14} | {'Max Error':>14}")
    print("-" * 60)
    for i in range(6):
        r2 = r2_score(Y_true[:, i], Y_pred[:, i])
        mae = mean_absolute_error(Y_true[:, i], Y_pred[:, i])
        max_err = np.max(np.abs(Y_true[:, i] - Y_pred[:, i]))
        print(
            f"{DOF_NAMES[i]:<15} | {r2:>10.4f} | "
            f"{mae:>10.4f} {UNITS[i]} | {max_err:>10.4f} {UNITS[i]}"
        )
    print()

    # Visualisation
    time_steps = np.arange(VISUALIZE_LEN)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(time_steps, Y_true[:, i], 'k-', linewidth=2.5, alpha=0.6, label='Ground Truth')
        ax.plot(time_steps, Y_pred[:, i], 'r--o', markersize=3, linewidth=1.5, label='Prediction')
        ax.fill_between(time_steps, Y_true[:, i], Y_pred[:, i], color='gray', alpha=0.2, label='Error')
        ax.set_title(f"{DOF_NAMES[i]}", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"({UNITS[i]})")
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend(loc='upper right')

    plt.suptitle(
        f"DeepHydroSim Evaluation  [slice {VISUALIZE_START_IDX}:{VISUALIZE_START_IDX + VISUALIZE_LEN}]",
        fontsize=14,
    )
    plt.tight_layout()

    os.makedirs('../logs', exist_ok=True)
    save_path = '../logs/evaluation_detail.png'
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    main()
