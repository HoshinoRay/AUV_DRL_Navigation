import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

FILE_PATH = '/home/ray/Disk_ext/DeepSim_RL/cpp_sim/logs/GeneralMission_2026-01-30_12-47-05.csv'
MASS_RIGID = 151.349  # kg
INERTIA_RIGID = np.array([27.612, 35.3413, 61.4596]) # Ixx, Iyy, Izz
ADDED_MASS_PCT = {
    'u': 0.217,'v': 0.893,'w': 1.021,'p': 0.796,'q': 0.632,'r': 1.103
}
# 推进器几何布局 (米) - 基于 GIRONA500 典型布局估算
# 用于计算力矩 (Torque)
X_OFFSET = 0.45  # 前后距离中心
Y_OFFSET = 0.30  # 左右距离中心 (估计值)

# ===========================================

def calculate_mass_matrix():
    """构建包含附加质量的总质量矩阵 (对角阵)"""
    m_u = MASS_RIGID * (1 + ADDED_MASS_PCT['u'])
    m_v = MASS_RIGID * (1 + ADDED_MASS_PCT['v'])
    m_w = MASS_RIGID * (1 + ADDED_MASS_PCT['w'])
    i_p = INERTIA_RIGID[0] * (1 + ADDED_MASS_PCT['p'])
    i_q = INERTIA_RIGID[1] * (1 + ADDED_MASS_PCT['q'])
    i_r = INERTIA_RIGID[2] * (1 + ADDED_MASS_PCT['r'])
    
    return np.array([m_u, m_v, m_w, i_p, i_q, i_r])

def compute_propulsion_forces(df):
    """
    将 8 个电机的 PWM (-1 ~ 1) 转换为 6 自由度力/力矩。
    基于 GIRONA500 的典型 X 型布局 (Vector) 和 垂直布局。
    """
    # 获取电机数据 (假设列名对应你的 DataCollector)
    # 0-3: Horz (FL, FR, RL, RR), 4-7: Vert (VFL, VFR, VRL, VRR)
    m = df[['M_FL', 'M_FR', 'M_RL', 'M_RR', 'M_VFL', 'M_VFR', 'M_VRL', 'M_VRR']].values 

    # 初始化 6DOF 力数组
    tau = np.zeros((len(df), 6))
    # Surge (X): 全部向前推
    # FL(0), FR(1), RL(2), RR(3) 
    # 假设正转是向前推 (根据具体螺旋桨方向可能需要取反)
    tau[:, 0] = (m[:, 0] + m[:, 1] + m[:, 2] + m[:, 3]) 

    # Sway (Y): 左右平移 (FL-FR-RL+RR 类似逻辑)
    # 这取决于具体安装角度，这里采用标准矢量分布
    tau[:, 1] = (-m[:, 0] + m[:, 1] - m[:, 2] + m[:, 3]) 

    # Yaw (N): 差速转向
    # 力矩 = 力 * 力臂。力臂约为 sqrt(x^2 + y^2)
    # 简化：左侧推 - 右侧推
    tau[:, 5] = (-m[:, 0] + m[:, 1] - m[:, 2] + m[:, 3]) * Y_OFFSET # 简化力臂计算

    # --- 垂直推进器 (4,5,6,7) ---
    # Heave (Z): 全部向下/向上
    tau[:, 2] = -(m[:, 4] + m[:, 5] + m[:, 6] + m[:, 7]) # 注意 Z 轴向下为正，如果推力向上则为负

    # Roll (K): 左右差速
    tau[:, 3] = (m[:, 4] - m[:, 5] + m[:, 6] - m[:, 7]) * Y_OFFSET

    # Pitch (M): 前后差速
    tau[:, 4] = (-m[:, 4] - m[:, 5] + m[:, 6] + m[:, 7]) * X_OFFSET

    return tau

def main():
    print(f"Loading data from {FILE_PATH}...")
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print("Error: File not found. Please check the path.")
        return

    # 2. 数据预处理与平滑
    # 仿真数据的 dt 可能有微小抖动，计算平均 dt 或逐点 dt
    time = df['Time(s)'].values
    dt_array = np.gradient(time)
    
    # 提取速度向量 [u, v, w, p, q, r]
    velocities = df[['u(m/s)', 'v(m/s)', 'w(m/s)', 'p(rad/s)', 'q(rad/s)', 'r(rad/s)']].values
    
    # 平滑速度 (重要！直接微分噪声极大)
    # window_length 必须是奇数，根据数据频率调整
    window_length = 51 
    polyorder = 3
    if len(df) > window_length:
        velocities_smooth = savgol_filter(velocities, window_length, polyorder, axis=0)
    else:
        velocities_smooth = velocities

    # 3. 计算加速度 (nu_dot)
    # axis=0 对每一列求导
    accelerations = np.zeros_like(velocities_smooth)
    for i in range(6):
        accelerations[:, i] = np.gradient(velocities_smooth[:, i], time)

    # 4. 计算合力与阻尼力
    # Total Mass Matrix
    M_diag = calculate_mass_matrix()
    
    # F_inertial = M * a
    F_inertial = accelerations * M_diag  # 广播乘法 (每列乘对应质量)

    # F_propulsion (从电机 PWM 估算)
    F_prop = compute_propulsion_forces(df)

    # 核心公式: F_damping = F_prop - M*a
    # (忽略科里奥利和重力)
    F_damping = F_prop - F_inertial

    # 5. 可视化
    dof_names = ['Surge (u)', 'Sway (v)', 'Heave (w)', 'Roll (p)', 'Pitch (q)', 'Yaw (r)']
    units = ['m/s', 'm/s', 'm/s', 'rad/s', 'rad/s', 'rad/s']
    force_units = ['N', 'N', 'N', 'N.m', 'N.m', 'N.m']

    # --- 图 1: 速度 vs 阻尼力 (检查线性/二次关系) ---
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('Velocity vs. Calculated Damping Force\nExpectation: Quadratic Curve (F = -d*v*|v|) or Linear (F = -d*v)', fontsize=16)

    # --- 图 2: 速度平方 (signed) vs 阻尼力 (检查是否是一条直线) ---
    # 如果 F ~ v^2，那么 F vs v*|v| 应该是一条直线
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Velocity^2 (Signed) vs. Calculated Damping Force\nExpectation: Linear Line for Quadratic Drag', fontsize=16)

    plot_indices = [0, 2, 3, 4, 5]

    # 使用 enumerate 获取：
    # plot_idx: 用于排版 (0, 1, 2, 3, 4) -> 决定画在第几个格子里
    # dof_idx:  用于取数据 (0, 2, 3, 4, 5) -> 决定取哪一列数据
    for plot_idx, i in enumerate(plot_indices):
        # 根据 plot_idx 计算子图位置，这样图会紧凑排列，不会空出中间的格子
        row, col = divmod(plot_idx, 3) 
        
        # 下面的逻辑完全不变，只是变量名要注意：
        # 数据获取依然使用 'i' (代表真实的 DOF 索引)
        v = velocities_smooth[:, i]
        f = F_damping[:, i]
        v_abs_v = v * np.abs(v) 

        mask = np.abs(v) > 0.01
        v_filt = v[mask]
        f_filt = f[mask]
        v2_filt = v_abs_v[mask]

        # Plot 1: F vs V
        ax1 = axes1[row, col]
        ax1.scatter(v_filt, f_filt, alpha=0.1, s=2, c='blue', label='Data')
        ax1.set_title(f'{dof_names[i]}')  # 使用 i 获取正确的名字
        ax1.set_xlabel(f'Velocity ({units[i]})')
        ax1.set_ylabel(f'Damping Force ({force_units[i]})')
        ax1.grid(True, alpha=0.3)
        
        if len(v_filt) > 10:
            coeff = np.linalg.lstsq(v2_filt.reshape(-1, 1), f_filt, rcond=None)[0][0]
            v_fit = np.linspace(min(v_filt), max(v_filt), 100)
            f_fit = coeff * v_fit * np.abs(v_fit)
            ax1.plot(v_fit, f_fit, 'r--', lw=2, label=f'Fit: {coeff:.2f}*v|v|')
            ax1.legend()

        # Plot 2: F vs V*|V|
        ax2 = axes2[row, col]
        ax2.scatter(v2_filt, f_filt, alpha=0.1, s=2, c='green')
        ax2.set_title(f'{dof_names[i]}')
        ax2.set_xlabel(f'Signed V^2 ({units[i]}^2)')
        ax2.set_ylabel(f'Damping Force ({force_units[i]})')
        ax2.grid(True, alpha=0.3)
        
        if len(v_filt) > 10:
             x_fit = np.linspace(min(v2_filt), max(v2_filt), 100)
             y_fit = coeff * x_fit
             ax2.plot(x_fit, y_fit, 'r--', lw=2, label=f'Slope: {coeff:.2f}')
             ax2.legend()

    # 隐藏最后一张空白的子图 (第6个格子，即 row=1, col=2)
    axes1[1, 2].axis('off')
    axes2[1, 2].axis('off')
    # =========== 修改结束 ===========

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图片到同一目录
    save_dir = FILE_PATH.rsplit('/', 1)[0]
    fig1.savefig(f"{save_dir}/Damping_Analysis_F.png")
    fig2.savefig(f"{save_dir}/Damping_Analysis_F2.png")
    
    print(f"Analysis Complete. Plots saved to {save_dir}")
    plt.show()

if __name__ == "__main__":
    main()