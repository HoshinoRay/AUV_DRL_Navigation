import pandas as pd
import numpy as np
import os
from scipy.signal import savgol_filter

# ================= 配置区域 (Configuration) =================

# 请修改这里的路径为你实际 CSV 文件的路径
INPUT_CSV_PATH = '/home/ray/Disk_ext/DeepSim_RL/hydro_MLP/data/phy_processed/mission_log_processed_Physics.csv' 
OUTPUT_CSV_PATH = 'mission_log_processed_Physics.csv'

# --- 1. 刚体参数 (Rigid Body Parameters) ---
MASS_RIGID = 151.349  # kg
INERTIA_RIGID = np.array([27.612, 35.3413, 61.4596]) # Ixx, Iyy, Izz (kg*m^2)

# --- 2. 附加质量系数 (Added Mass Percentages) ---
# 物理公式: M_total = M_rigid * (1 + percentage)
ADDED_MASS_PCT = {
    'u': 0.217,  # Surge
    'v': 0.893,  # Sway
    'w': 1.021,  # Heave
    'p': 0.796,  # Roll
    'q': 0.632,  # Pitch
    'r': 1.103   # Yaw
}

# --- 3. 推进器几何参数 (Thruster Geometry) ---
IN_X = 0.33    # 内圈前后
IN_Y = 0.137   # 内圈左右
OUT_X = 0.45   # 外圈前后
OUT_Y = 0.60   # 外圈左右

# --- 4. 数据平滑参数 (Signal Processing) ---
# 用于计算加速度。窗口越大越平滑，但会丢失高频细节。
# 必须是奇数。建议: 51 或 101 (取决于你的采样率，假设是 50-100Hz)
SMOOTH_WINDOW = 51 
POLY_ORDER = 3

# ==========================================================

def calculate_total_mass_matrix():
    """
    计算包含附加质量的总质量对角矩阵向量
    Returns: np.array [m_u, m_v, m_w, I_p, I_q, I_r]
    """
    # 线性质量 (kg)
    m_u = MASS_RIGID * (1 + ADDED_MASS_PCT['u'])
    m_v = MASS_RIGID * (1 + ADDED_MASS_PCT['v'])
    m_w = MASS_RIGID * (1 + ADDED_MASS_PCT['w'])
    
    # 转动惯量 (kg*m^2)
    i_p = INERTIA_RIGID[0] * (1 + ADDED_MASS_PCT['p'])
    i_q = INERTIA_RIGID[1] * (1 + ADDED_MASS_PCT['q'])
    i_r = INERTIA_RIGID[2] * (1 + ADDED_MASS_PCT['r'])
    
    return np.array([m_u, m_v, m_w, i_p, i_q, i_r])

def process_data():
    print(f"[Info] 正在读取文件: {INPUT_CSV_PATH}")
    
    # 检查文件是否存在
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"[Error] 文件未找到: {INPUT_CSV_PATH}")
        # 为了演示，如果文件不存在，我创建一个模拟的 DataFrame 防止报错，实际使用请忽略
        return 

    df = pd.read_csv(INPUT_CSV_PATH)
    
    # ================= 1. 数据提取 =================
    # 提取时间
    time_arr = df['Time(s)'].values
    dt_arr = np.gradient(time_arr)
    dt_arr[dt_arr == 0] = 1e-5 # 防止除以0
    
    # 提取速度 (根据你的 CSV 列名)
    # u, v, w, p, q, r
    vel_cols = ['u(m/s)', 'v(m/s)', 'w(m/s)', 'p(rad/s)', 'q(rad/s)', 'r(rad/s)']
    velocities = df[vel_cols].values

    # 提取电机推力 (根据你的 CSV 列名)
    motor_cols = ['M_FL', 'M_FR', 'M_RL', 'M_RR', 'M_VFL', 'M_VFR', 'M_VRL', 'M_VRR']
    thrusts = df[motor_cols].values
    
    # 分解推力变量，方便计算
    FL, FR, RL, RR = thrusts[:,0], thrusts[:,1], thrusts[:,2], thrusts[:,3]
    VFL, VFR, VRL, VRR = thrusts[:,4], thrusts[:,5], thrusts[:,6], thrusts[:,7]

    # ================= 2. 计算加速度 (微分) =================
    print("[Info] 正在计算平滑加速度...")
    accelerations = np.zeros_like(velocities)
    
    # 对 6 个自由度分别进行平滑和求导
    for i in range(6):
        # 步骤 A: Savitzky-Golay 平滑 (去除传感器噪声)
        # 如果数据行数少于窗口大小，自动调整窗口
        current_window = SMOOTH_WINDOW if len(df) > SMOOTH_WINDOW else (len(df) // 2 * 2 + 1)
        if current_window < 3: current_window = 3 # 最小窗口
        
        vel_smooth = savgol_filter(velocities[:, i], window_length=current_window, polyorder=POLY_ORDER)
        
        # 步骤 B: 对时间求导得到加速度 a = dv/dt
        accelerations[:, i] = np.gradient(vel_smooth, time_arr)

    # ================= 3. 计算惯性力 (F_inertial) =================
    # F = M_total * a
    M_vec = calculate_total_mass_matrix()
    print(f"[Info] 总质量矩阵向量 (Rigid+Added): \n       {M_vec}")
    
    # 广播乘法: (N, 6) * (6,)
    F_inertial = accelerations * M_vec

    # ================= 4. 计算推进力 (Tau_prop) =================
    # 根据你的 Mix Matrix 逻辑
    print("[Info] 正在计算 6-DOF 推进力...")
    tau_prop = np.zeros((len(df), 6))
    
    # [0] Surge (+X): 所有水平推力向前
    tau_prop[:, 0] = FL + FR + RL + RR
    
    # [1] Sway (+Y): 假设水平推进器是平行布置，无侧向力
    # 如果是矢量布置(45度)，这里需要修改公式。根据上一段对话，这里设为0。
    tau_prop[:, 1] = 0.0 
    
    # [2] Heave (+Z): 所有垂直推力向下
    tau_prop[:, 2] = VFL + VFR + VRL + VRR
    
    # [3] Roll (+P): (右垂 - 左垂) * 力臂
    # 注意: VFR(前右)+VRR(后右) - VFL(前左)-VRL(后左)
    tau_prop[:, 3] = (VFR + VRR - VFL - VRL) * OUT_Y
    
    # [4] Pitch (+Q): (后垂 - 前垂) * 力臂
    # 注意: VRL(后左)+VRR(后右) - VFL(前左)-VFR(前右)
    tau_prop[:, 4] = (VRL + VRR - VFL - VFR) * OUT_X
    
    # [5] Yaw (+R): (左水平 - 右水平) * 力臂
    # 注意: FL(前左)+RL(后左) - FR(前右)-RR(后右)
    tau_prop[:, 5] = (FL + RL - FR - RR) * IN_Y

    # ================= 5. 计算流体阻尼力 (Ground Truth) =================
    # 核心逆动力学公式: F_fluid = Tau - M*a
    print("[Info] 执行逆动力学解算...")
    F_fluid = tau_prop - F_inertial

    # ================= 6. 保存结果 =================
    # 创建结果 DataFrame
    result_df = df.copy()

    dof_names = ['u', 'v', 'w', 'p', 'q', 'r'] # 对应 Surge, Sway, Heave, Roll, Pitch, Yaw
    
    # 1. 追加加速度列
    for i, axis in enumerate(dof_names):
        result_df[f'Accel_{axis}(m/s2)'] = accelerations[:, i]

    # 2. 追加推进力列 (Tau)
    for i, axis in enumerate(dof_names):
        result_df[f'Tau_{axis}(N)'] = tau_prop[:, i]

    # 3. 追加惯性力列 (F_inertial)
    for i, axis in enumerate(dof_names):
        result_df[f'F_Inertial_{axis}(N)'] = F_inertial[:, i]

    # 4. 追加流体阻尼力列 (F_Fluid - 最终目标)
    for i, axis in enumerate(dof_names):
        result_df[f'F_Fluid_{axis}(N)'] = F_fluid[:, i]

    # 保存文件
    print(f"[Success] 保存处理后的数据到: {OUTPUT_CSV_PATH}")
    result_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')

    # === 打印校验信息 (Sanity Check) ===
    print("\n" + "="*40)
    print("   数据校验 (最后一行数据)")
    print("="*40)
    last_idx = len(result_df) - 1
    
    # 以 Surge (u) 和 Heave (w) 为例
    print(f"Surge (u):")
    print(f"  推进力 (Tau):      {tau_prop[last_idx, 0]:.4f} N")
    print(f"  - 惯性力 (M*a):    {F_inertial[last_idx, 0]:.4f} N (a={accelerations[last_idx, 0]:.4f})")
    print(f"  = 流体阻力 (Fluid): {F_fluid[last_idx, 0]:.4f} N")
    print("-" * 20)
    print(f"Heave (w):")
    print(f"  推进力 (Tau):      {tau_prop[last_idx, 2]:.4f} N")
    print(f"  - 惯性力 (M*a):    {F_inertial[last_idx, 2]:.4f} N (a={accelerations[last_idx, 2]:.4f})")
    print(f"  = 流体阻力 (Fluid): {F_fluid[last_idx, 2]:.4f} N")
    print("="*40)

if __name__ == "__main__":
    process_data()