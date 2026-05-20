"""
Damping identification tool.

Loads a DataCollector CSV, reconstructs propulsion forces from motor setpoints,
and plots hydrodynamic damping vs. velocity for each DOF to identify drag coefficients.

Usage:
    python analysing_damping.py [path/to/log.csv]

If no path is given, the most recent GeneralMission_*.csv in ../logs/ is used.
"""

import os
import sys
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def _find_latest_log(log_dir: str) -> str | None:
    files = sorted(glob.glob(os.path.join(log_dir, "GeneralMission_*.csv")))
    return files[-1] if files else None


# Vehicle inertial parameters
MASS_RIGID     = 151.349                                       # kg
INERTIA_RIGID  = np.array([27.612, 35.3413, 61.4596])        # Ixx, Iyy, Izz  [kg·m²]
ADDED_MASS_PCT = {'u': 0.217, 'v': 0.893, 'w': 1.021,
                  'p': 0.796, 'q': 0.632, 'r': 1.103}

# Thruster geometry (metres) — GIRONA500 layout
X_OFFSET = 0.45   # fore-aft arm length
Y_OFFSET = 0.30   # lateral arm length


def calculate_mass_matrix() -> np.ndarray:
    """Return the 6-element diagonal of the total mass matrix (rigid + added)."""
    m_u = MASS_RIGID * (1 + ADDED_MASS_PCT['u'])
    m_v = MASS_RIGID * (1 + ADDED_MASS_PCT['v'])
    m_w = MASS_RIGID * (1 + ADDED_MASS_PCT['w'])
    i_p = INERTIA_RIGID[0] * (1 + ADDED_MASS_PCT['p'])
    i_q = INERTIA_RIGID[1] * (1 + ADDED_MASS_PCT['q'])
    i_r = INERTIA_RIGID[2] * (1 + ADDED_MASS_PCT['r'])
    return np.array([m_u, m_v, m_w, i_p, i_q, i_r])


def compute_propulsion_forces(df: pd.DataFrame) -> np.ndarray:
    """
    Map 8-thruster PWM setpoints in [-1, 1] to 6-DOF body-frame wrench.

    Thruster layout (GIRONA500, X-type horizontal + vertical):
      Horizontal: FL(0), FR(1), RL(2), RR(3)
      Vertical:   VFL(4), VFR(5), VRL(6), VRR(7)
    """
    m   = df[['M_FL', 'M_FR', 'M_RL', 'M_RR',
               'M_VFL', 'M_VFR', 'M_VRL', 'M_VRR']].values
    tau = np.zeros((len(df), 6))

    # Surge (X)
    tau[:, 0] = m[:, 0] + m[:, 1] + m[:, 2] + m[:, 3]
    # Sway (Y)
    tau[:, 1] = -m[:, 0] + m[:, 1] - m[:, 2] + m[:, 3]
    # Heave (Z) — positive downward
    tau[:, 2] = -(m[:, 4] + m[:, 5] + m[:, 6] + m[:, 7])
    # Roll (K)
    tau[:, 3] = (m[:, 4] - m[:, 5] + m[:, 6] - m[:, 7]) * Y_OFFSET
    # Pitch (M)
    tau[:, 4] = (-m[:, 4] - m[:, 5] + m[:, 6] + m[:, 7]) * X_OFFSET
    # Yaw (N)
    tau[:, 5] = (-m[:, 0] + m[:, 1] - m[:, 2] + m[:, 3]) * Y_OFFSET

    return tau


def main(file_path: str):
    print(f"Loading data from {file_path} ...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: file not found: {file_path}")
        return

    time         = df['Time(s)'].values
    velocities   = df[['u(m/s)', 'v(m/s)', 'w(m/s)',
                        'p(rad/s)', 'q(rad/s)', 'r(rad/s)']].values

    # Smooth velocities before differentiating (Savitzky-Golay)
    window_length = 51
    polyorder     = 3
    if len(df) > window_length:
        velocities_smooth = savgol_filter(velocities, window_length, polyorder, axis=0)
    else:
        velocities_smooth = velocities

    # Finite-difference acceleration
    accelerations = np.zeros_like(velocities_smooth)
    for i in range(6):
        accelerations[:, i] = np.gradient(velocities_smooth[:, i], time)

    M_diag    = calculate_mass_matrix()
    F_inertial = accelerations * M_diag
    F_prop    = compute_propulsion_forces(df)
    # Damping force: F_d = F_prop - M*a  (Coriolis and gravity neglected)
    F_damping = F_prop - F_inertial

    dof_names   = ['Surge (u)', 'Sway (v)', 'Heave (w)', 'Roll (p)', 'Pitch (q)', 'Yaw (r)']
    vel_units   = ['m/s'] * 3 + ['rad/s'] * 3
    force_units = ['N']   * 3 + ['N·m']   * 3

    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('Velocity vs. Damping Force  (F = −d·v·|v|  expected)',     fontsize=16)
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Signed v² vs. Damping Force  (linear if quadratic drag)',   fontsize=16)

    # Skip sway (index 1) — often under-excited; plot the other 5 DOFs
    plot_indices = [0, 2, 3, 4, 5]

    for plot_idx, i in enumerate(plot_indices):
        row, col = divmod(plot_idx, 3)

        v       = velocities_smooth[:, i]
        f       = F_damping[:, i]
        v_abs_v = v * np.abs(v)

        mask    = np.abs(v) > 0.01
        v_filt  = v[mask];  f_filt = f[mask];  v2_filt = v_abs_v[mask]

        ax1 = axes1[row, col]
        ax1.scatter(v_filt, f_filt, alpha=0.1, s=2, c='blue', label='Data')
        ax1.set_title(dof_names[i])
        ax1.set_xlabel(f'Velocity ({vel_units[i]})')
        ax1.set_ylabel(f'Damping Force ({force_units[i]})')
        ax1.grid(True, alpha=0.3)

        if len(v_filt) > 10:
            coeff = np.linalg.lstsq(v2_filt.reshape(-1, 1), f_filt, rcond=None)[0][0]
            v_fit = np.linspace(v_filt.min(), v_filt.max(), 100)
            ax1.plot(v_fit, coeff * v_fit * np.abs(v_fit), 'r--', lw=2,
                     label=f'Fit: {coeff:.2f}·v|v|')
            ax1.legend()

        ax2 = axes2[row, col]
        ax2.scatter(v2_filt, f_filt, alpha=0.1, s=2, c='green')
        ax2.set_title(dof_names[i])
        ax2.set_xlabel(f'Signed v² ({vel_units[i]}²)')
        ax2.set_ylabel(f'Damping Force ({force_units[i]})')
        ax2.grid(True, alpha=0.3)

        if len(v_filt) > 10:
            x_fit = np.linspace(v2_filt.min(), v2_filt.max(), 100)
            ax2.plot(x_fit, coeff * x_fit, 'r--', lw=2, label=f'Slope: {coeff:.2f}')
            ax2.legend()

    axes1[1, 2].axis('off')
    axes2[1, 2].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_dir = os.path.dirname(os.path.abspath(file_path))
    fig1.savefig(os.path.join(save_dir, "Damping_Analysis_F.png"))
    fig2.savefig(os.path.join(save_dir, "Damping_Analysis_F2.png"))
    print(f"Analysis complete. Plots saved to {save_dir}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fp = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir    = os.path.join(script_dir, "..", "..", "logs")
        fp = _find_latest_log(log_dir)
        if fp is None:
            print(f"No GeneralMission_*.csv found in {log_dir}")
            sys.exit(1)
        print(f"Using most recent log: {fp}")

    main(fp)
