"""
Visualise the processed physics CSV: force decomposition time series and
velocity-vs-drag scatter plots.

Usage (from hydro_MLP/):
    python src/data_analysis.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FILE_PATH = '../data/phy_processed/mission_log_processed_Physics.csv'
# ---------------------------------------------------------------------------

DOFS = [
    ('u', 'Surge (Longitudinal)'),
    ('v', 'Sway (Lateral)'),
    ('w', 'Heave (Vertical)'),
    ('p', 'Roll'),
    ('q', 'Pitch'),
    ('r', 'Yaw'),
]


def check_data_quality():
    if not os.path.exists(FILE_PATH):
        print(f"[Error] File not found: {FILE_PATH}")
        return

    output_dir = os.path.dirname(FILE_PATH)
    base = os.path.splitext(os.path.basename(FILE_PATH))[0]
    save_path_decomp = os.path.join(output_dir, f"{base}_ForceDecomposition.png")
    save_path_drag   = os.path.join(output_dir, f"{base}_VelocityDrag.png")

    print(f"Loading: {FILE_PATH}")
    df = pd.read_csv(FILE_PATH)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    sns.set(style='whitegrid')
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['axes.unicode_minus'] = False

    # ---- Plot 1: Force decomposition time series ----
    fig1, axes1 = plt.subplots(3, 2, figsize=(18, 12))
    fig1.suptitle('Dynamics Check: Tau (Propulsion) = Inertial + Fluid (Drag)', fontsize=16)
    t = df['Time(s)']
    for i, (axis, label) in enumerate(DOFS):
        ax = axes1[i // 2, i % 2]
        ax.plot(t, df[f'Tau_{axis}(N)'],          label='Tau (Propulsion)', color='green',  alpha=0.6, linewidth=1.5)
        ax.plot(t, df[f'F_Fluid_{axis}(N)'],       label='F_Fluid (Drag)',   color='blue',   alpha=0.8, linewidth=1.5)
        ax.plot(t, df[f'F_Inertial_{axis}(N)'],    label='F_Inertial (Ma)',  color='red',    alpha=0.3, linewidth=1.0, linestyle='--')
        ax.set_title(f'{label} — Force Decomposition')
        ax.set_ylabel('Force / Torque (N / Nm)')
        ax.legend(loc='upper right', fontsize='small')
    axes1[2, 0].set_xlabel('Time (s)')
    axes1[2, 1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(save_path_decomp)
    print(f"Saved: {save_path_decomp}")

    # ---- Plot 2: Velocity vs. drag scatter ----
    fig2, axes2 = plt.subplots(3, 2, figsize=(18, 12))
    fig2.suptitle('System ID: Velocity vs. Fluid Drag Force', fontsize=16)
    for i, (axis, label) in enumerate(DOFS):
        ax = axes2[i // 2, i % 2]
        vel_col = f'{axis}(m/s)' if i < 3 else f'{axis}(rad/s)'
        x = df[vel_col]
        y = df[f'F_Fluid_{axis}(N)']
        ax.scatter(x, y, alpha=0.15, s=10, color='darkblue')
        try:
            sort_idx = np.argsort(x)
            p = np.poly1d(np.polyfit(x, y, 2))
            ax.plot(x[sort_idx], p(x[sort_idx]), 'r--', linewidth=2, label='Quadratic Fit')
        except Exception:
            pass
        ax.set_title(f'{label}: Velocity vs Drag')
        ax.set_xlabel('Velocity (m/s or rad/s)')
        ax.set_ylabel('Fluid Force (N or Nm)')
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax.axvline(0, color='black', linewidth=1, alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path_drag)
    print(f"Saved: {save_path_drag}")


if __name__ == '__main__':
    check_data_quality()
