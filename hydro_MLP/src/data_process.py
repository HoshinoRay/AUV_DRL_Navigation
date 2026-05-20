"""
Inverse-dynamics processor: derive fluid hydrodynamic forces from raw
Stonefish simulation logs.

Given body-frame velocity, thruster setpoints, and physical parameters,
the script computes:
    F_fluid = Tau_prop - M_total * accel

where Tau_prop is the 6-DOF propulsion wrench and M_total is the
rigid-body + added-mass matrix. The result is appended to the CSV and
saved as the ground-truth target for DeepHydroMLP training.

Usage (from hydro_MLP/):
    python src/data_process.py
"""

import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV_PATH = '../data/phy_processed/mission_log_processed_Physics.csv'
OUTPUT_CSV_PATH = '../data/phy_processed/mission_log_processed_Physics_out.csv'

# Rigid-body parameters
MASS_RIGID = 151.349        # kg
INERTIA_RIGID = np.array([27.612, 35.3413, 61.4596])  # Ixx, Iyy, Izz  (kg·m²)

# Added-mass fractions: M_total = M_rigid * (1 + fraction)
ADDED_MASS_PCT = {
    'u': 0.217,   # Surge
    'v': 0.893,   # Sway
    'w': 1.021,   # Heave
    'p': 0.796,   # Roll
    'q': 0.632,   # Pitch
    'r': 1.103,   # Yaw
}

# Thruster moment arms (m)
IN_X  = 0.33   # horizontal inner ring, fore-aft
IN_Y  = 0.137  # horizontal inner ring, port-starboard
OUT_X = 0.45   # vertical outer ring, fore-aft
OUT_Y = 0.60   # vertical outer ring, port-starboard

# Savitzky-Golay smoothing for numerical differentiation
SMOOTH_WINDOW = 51   # must be odd
POLY_ORDER = 3
# ---------------------------------------------------------------------------


def _total_mass_vector() -> np.ndarray:
    """Return [m_u, m_v, m_w, I_p, I_q, I_r] including added mass."""
    return np.array([
        MASS_RIGID * (1 + ADDED_MASS_PCT['u']),
        MASS_RIGID * (1 + ADDED_MASS_PCT['v']),
        MASS_RIGID * (1 + ADDED_MASS_PCT['w']),
        INERTIA_RIGID[0] * (1 + ADDED_MASS_PCT['p']),
        INERTIA_RIGID[1] * (1 + ADDED_MASS_PCT['q']),
        INERTIA_RIGID[2] * (1 + ADDED_MASS_PCT['r']),
    ])


def process_data():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"[Error] File not found: {INPUT_CSV_PATH}")
        return

    print(f"[DataProcess] Reading: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    # ---- 1. Extract time and velocity ----
    time_arr = df['Time(s)'].values
    dt_arr = np.gradient(time_arr)
    dt_arr[dt_arr == 0] = 1e-5

    vel_cols = ['u(m/s)', 'v(m/s)', 'w(m/s)', 'p(rad/s)', 'q(rad/s)', 'r(rad/s)']
    velocities = df[vel_cols].values

    # ---- 2. Extract thruster setpoints ----
    motor_cols = ['M_FL', 'M_FR', 'M_RL', 'M_RR', 'M_VFL', 'M_VFR', 'M_VRL', 'M_VRR']
    thrusts = df[motor_cols].values
    FL, FR, RL, RR = thrusts[:, 0], thrusts[:, 1], thrusts[:, 2], thrusts[:, 3]
    VFL, VFR, VRL, VRR = thrusts[:, 4], thrusts[:, 5], thrusts[:, 6], thrusts[:, 7]

    # ---- 3. Compute acceleration via Savitzky-Golay smoothing + differentiation ----
    print("[DataProcess] Computing accelerations ...")
    accelerations = np.zeros_like(velocities)
    window = SMOOTH_WINDOW if len(df) > SMOOTH_WINDOW else (len(df) // 2 * 2 + 1)
    window = max(window, 3)
    for i in range(6):
        vel_smooth = savgol_filter(velocities[:, i], window_length=window, polyorder=POLY_ORDER)
        accelerations[:, i] = np.gradient(vel_smooth, time_arr)

    # ---- 4. Inertial wrench  F_inertial = M_total * a ----
    M_vec = _total_mass_vector()
    print(f"[DataProcess] Total mass vector (rigid + added): {M_vec}")
    F_inertial = accelerations * M_vec

    # ---- 5. Propulsion wrench (6-DOF mix matrix) ----
    print("[DataProcess] Computing propulsion wrench ...")
    tau_prop = np.zeros((len(df), 6))
    tau_prop[:, 0] = FL + FR + RL + RR                       # Surge
    tau_prop[:, 1] = 0.0                                      # Sway (parallel layout)
    tau_prop[:, 2] = VFL + VFR + VRL + VRR                   # Heave
    tau_prop[:, 3] = (VFR + VRR - VFL - VRL) * OUT_Y         # Roll
    tau_prop[:, 4] = (VRL + VRR - VFL - VFR) * OUT_X         # Pitch
    tau_prop[:, 5] = (FL + RL - FR - RR) * IN_Y              # Yaw

    # ---- 6. Fluid force: inverse dynamics ----
    print("[DataProcess] Running inverse dynamics ...")
    F_fluid = tau_prop - F_inertial

    # ---- 7. Write result CSV ----
    result_df = df.copy()
    dof_names = ['u', 'v', 'w', 'p', 'q', 'r']
    for i, ax in enumerate(dof_names):
        result_df[f'Accel_{ax}(m/s2)'] = accelerations[:, i]
        result_df[f'Tau_{ax}(N)'] = tau_prop[:, i]
        result_df[f'F_Inertial_{ax}(N)'] = F_inertial[:, i]
        result_df[f'F_Fluid_{ax}(N)'] = F_fluid[:, i]

    result_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')
    print(f"[DataProcess] Saved to: {OUTPUT_CSV_PATH}")

    # Quick sanity check on the last row
    last = len(result_df) - 1
    for ax_idx, ax in enumerate(['u', 'w']):
        label = 'Surge' if ax == 'u' else 'Heave'
        print(
            f"  {label}: Tau={tau_prop[last, ax_idx * 2]:.4f} N  "
            f"M*a={F_inertial[last, ax_idx * 2]:.4f} N  "
            f"F_fluid={F_fluid[last, ax_idx * 2]:.4f} N"
        )


if __name__ == '__main__':
    process_data()
