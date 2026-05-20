"""
Plot velocity vs. hydrodynamic force/torque from a DataLogger CSV.

Usage:
    cd mujoco_sim
    python src/utils/check_mlp.py [path/to/log.csv]

If no path is given, the most recent CSV in ./logs/ is used.
"""

import os
import sys
import glob
import matplotlib.pyplot as plt


def _find_latest_log(log_dir: str) -> str | None:
    pattern = os.path.join(log_dir, "hydro_debug_*.csv")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def plot_all_dofs(log_path: str, save_path: str = "hydro_analysis.png"):
    import pandas as pd

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    print(f"Reading log: {log_path}")
    df = pd.read_csv(log_path)

    dofs = [
        ('u', 'Fx_H', 'Surge (Forward/Back)'),
        ('v', 'Fy_H', 'Sway (Left/Right)'),
        ('w', 'Fz_H', 'Heave (Up/Down)'),
        ('p', 'Tx_H', 'Roll (Rotation X)'),
        ('q', 'Ty_H', 'Pitch (Rotation Y)'),
        ('r', 'Tz_H', 'Yaw (Rotation Z)'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    ax_flat = axes.flatten()

    for i, (vel_col, force_col, title) in enumerate(dofs):
        if vel_col not in df.columns or force_col not in df.columns:
            print(f"Warning: column '{vel_col}' or '{force_col}' missing — skipping.")
            continue

        ax1 = ax_flat[i]
        ax2 = ax1.twinx()
        unit        = "Nm" if i >= 3 else "N"
        force_label = "Torque" if i >= 3 else "Force"

        color_vel   = 'tab:blue'
        color_force = 'tab:orange'

        line1, = ax1.plot(df['Time'], df[vel_col],   color=color_vel,   label=f'Vel ({vel_col})',   linewidth=2)
        line2, = ax2.plot(df['Time'], df[force_col], color=color_force, label=f'{force_label} ({force_col})',
                          linestyle='--', linewidth=1.5)

        ax1.set_ylabel('Velocity',                    color=color_vel,   fontweight='bold')
        ax2.set_ylabel(f'{force_label} ({unit})',     color=color_force, fontweight='bold')

        v_max = max(abs(df[vel_col]).max(),   1e-5)
        f_max = max(abs(df[force_col]).max(), 1e-5)
        ax1.set_ylim(-v_max * 1.2, v_max * 1.2)
        ax2.set_ylim(-f_max * 1.2, f_max * 1.2)

        ax1.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax1.grid(True, which='both', linestyle=':', alpha=0.5)
        ax1.set_title(title)
        ax1.legend([line1, line2], ['Velocity', force_label], loc='upper left')

    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    print(f"Saving plot: {save_path}")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir    = os.path.join(script_dir, "..", "..", "logs")
        log_path   = _find_latest_log(log_dir)
        if log_path is None:
            print(f"No hydro_debug_*.csv found in {log_dir}")
            sys.exit(1)
        print(f"Using most recent log: {log_path}")

    save_path = os.path.join(os.path.dirname(os.path.abspath(log_path)), "hydro_analysis.png")
    plot_all_dofs(log_path, save_path)
