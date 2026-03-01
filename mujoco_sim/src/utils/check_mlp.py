import pandas as pd
import matplotlib.pyplot as plt
import os

# 建议：使用相对路径或动态查找最新日志，这里先改回匹配你新架构的路径
LOG_PATH = '/home/ray/Disk_ext/DeepSim_RL/mujoco_sim/logs/hydro_debug_20260204_104614.csv'
SAVE_PATH = '../../logs/hydro_analysis_standard.png'

def plot_all_dofs():
    if not os.path.exists(LOG_PATH):
        print(f"错误：找不到日志文件 {LOG_PATH}")
        return

    print(f"正在读取日志: {LOG_PATH} ...")
    df = pd.read_csv(LOG_PATH)

    # === 关键修改点：更新列名以匹配新的 DataLogger ===
    # 格式：(速度列名, 力/力矩列名, 图表标题)
    dofs = [
        ('u', 'Fx_H', 'Surge (Forward/Back)'),
        ('v', 'Fy_H', 'Sway (Left/Right)'),
        ('w', 'Fz_H', 'Heave (Up/Down)'),
        ('p', 'Tx_H', 'Roll (Rotation X)'),
        ('q', 'Ty_H', 'Pitch (Rotation Y)'),
        ('r', 'Tz_H', 'Yaw (Rotation Z)')
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    ax_flat = axes.flatten()

    for i, (vel_col, force_col, title) in enumerate(dofs):
        if vel_col not in df.columns or force_col not in df.columns:
            print(f"警告：列 {vel_col} 或 {force_col} 不在 CSV 中，跳过该图。")
            continue

        ax1 = ax_flat[i]
        ax2 = ax1.twinx()
        
        unit = "N" if i < 3 else "Nm"
        y_label_force = "Torque" if i >= 3 else "Force"

        # 绘制速度 (左轴 - 蓝色)
        color_vel = 'tab:blue'
        line1, = ax1.plot(df['Time'], df[vel_col], color=color_vel, label=f'Vel ({vel_col})', linewidth=2)
        ax1.set_ylabel('Velocity', color=color_vel, fontweight='bold')
        
        # 绘制力 (右轴 - 橙色)
        color_force = 'tab:orange'
        line2, = ax2.plot(df['Time'], df[force_col], color=color_force, label=f'{y_label_force} ({force_col})', linestyle='--', linewidth=1.5)
        ax2.set_ylabel(f'{y_label_force} ({unit})', color=color_force, fontweight='bold')

        # 强制零位对齐逻辑
        v_max = max(abs(df[vel_col]).max(), 1e-5)
        f_max = max(abs(df[force_col]).max(), 1e-5)
        ax1.set_ylim(-v_max * 1.2, v_max * 1.2)
        ax2.set_ylim(-f_max * 1.2, f_max * 1.2)

        # 辅助线
        ax1.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax1.grid(True, which='both', linestyle=':', alpha=0.5)

        ax1.set_title(title)
        ax1.legend([line1, line2], ["Velocity", y_label_force], loc='upper left')

    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    print(f"正在保存图表至: {SAVE_PATH} ...")
    plt.savefig(SAVE_PATH, dpi=150)
    print("保存成功！")
    
    plt.show()

if __name__ == "__main__":
    plot_all_dofs()