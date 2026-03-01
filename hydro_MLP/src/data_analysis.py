import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ================= 配置区域 =================
FILE_PATH = '/home/ray/Disk_ext/DeepSim_RL/hydro_MLP/data/phy_processed/mission_log_processed_Physics.csv' 
# ===========================================

def check_data_quality():
    if not os.path.exists(FILE_PATH):
        print(f"错误: 找不到文件 {FILE_PATH}")
        return

    output_dir = os.path.dirname(FILE_PATH)
    base_filename = os.path.splitext(os.path.basename(FILE_PATH))[0]
    save_path_1 = os.path.join(output_dir, f"{base_filename}_ForceDecomposition.png")
    save_path_2 = os.path.join(output_dir, f"{base_filename}_VelocityDrag.png")

    print(f"[{'='*10} 开始处理 {'='*10}]")
    print(f"正在读取数据: {FILE_PATH} ...")
    df = pd.read_csv(FILE_PATH)
    print(f"数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")

    # === 修改点 1: 移除特定的字体设置，使用默认字体 ===
    sns.set(style="whitegrid")
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

    # === 修改点 2: 标签改为全英文，彻底解决乱码 ===
    dofs = [
        ('u', 'Surge (Longitudinal)'), 
        ('v', 'Sway (Lateral)'), 
        ('w', 'Heave (Vertical)'),
        ('p', 'Roll'), 
        ('q', 'Pitch'), 
        ('r', 'Yaw')
    ]

    # =========================================================
    # 图表 1: 时序分解
    # =========================================================
    print(f"\n正在绘制图表 1: 动力学时序分解...")
    fig1, axes1 = plt.subplots(3, 2, figsize=(18, 12))
    # 标题改为英文
    fig1.suptitle('Dynamics Check: Tau (Propulsion) = Inertial + Fluid (Drag)', fontsize=16, y=0.98)

    df_plot = df 
    t = df_plot['Time(s)']

    for i, (axis, label) in enumerate(dofs):
        ax = axes1[i//2, i%2]
        ax.plot(t, df_plot[f'Tau_{axis}(N)'], label='Tau (Propulsion)', color='green', alpha=0.6, linewidth=1.5)
        ax.plot(t, df_plot[f'F_Fluid_{axis}(N)'], label='F_Fluid (Drag)', color='blue', alpha=0.8, linewidth=1.5)
        ax.plot(t, df_plot[f'F_Inertial_{axis}(N)'], label='F_Inertial (Ma)', color='red', alpha=0.3, linewidth=1, linestyle='--')

        ax.set_title(f'{label} - Force Decomposition')
        ax.set_ylabel('Force / Torque (N / Nm)')
        ax.legend(loc='upper right', fontsize='small')
        
        if i < 3:
            ax.set_ylim(df_plot[[f'Tau_{axis}(N)', f'F_Fluid_{axis}(N)']].min().min()*1.2, 
                        df_plot[[f'Tau_{axis}(N)', f'F_Fluid_{axis}(N)']].max().max()*1.2)

    axes1[2, 0].set_xlabel('Time (s)')
    axes1[2, 1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(save_path_1)
    print(f"[Success] 图表 1 已保存至: {save_path_1}")

    # =========================================================
    # 图表 2: 阻尼特性曲线
    # =========================================================
    print(f"\n正在绘制图表 2: 阻尼特性曲线...")
    fig2, axes2 = plt.subplots(3, 2, figsize=(18, 12))
    # 标题改为英文
    fig2.suptitle('System ID View: Velocity vs. Fluid Drag Force', fontsize=16, y=0.98)

    for i, (axis, label) in enumerate(dofs):
        ax = axes2[i//2, i%2]
        x = df[f'{axis}(m/s)'] if i < 3 else df[f'{axis}(rad/s)']
        y = df[f'F_Fluid_{axis}(N)']

        ax.scatter(x, y, alpha=0.15, s=10, color='darkblue')

        try:
            sort_idx = np.argsort(x)
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)
            ax.plot(x[sort_idx], p(x[sort_idx]), "r--", linewidth=2, label='Trend Fit')
        except:
            pass

        ax.set_title(f'{label}: Velocity vs Drag')
        ax.set_xlabel('Velocity (m/s or rad/s)')
        ax.set_ylabel('Fluid Force (N or Nm)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax.axvline(0, color='black', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path_2)
    print(f"[Success] 图表 2 已保存至: {save_path_2}")
    
    print(f"\n[{'='*10} 处理完成 {'='*10}]")

if __name__ == "__main__":
    check_data_quality()