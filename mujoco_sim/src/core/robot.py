import mujoco
import numpy as np

class YuyuanRobot:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # 映射执行器 ID
        self.act_names = ["t0_hfr", "t1_hfl", "t2_hrr", "t3_hrl", "t4_vfr", "t5_vfl", "t6_vrr", "t7_vrl"]
        self.act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.act_names]
        self.max_thrust = 155.0

        # 映射用于可视化的 Site ID
        self.site_names = [f"thruster_{i}" for i in range(8)]
        self.site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in self.site_names]
        
        # 基础属性
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "yuyuan")
        self.cob_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cob_site")

    def set_thrusters(self, thrust_cmds):  #8 thrusters
        """下发推力指令并更新可视化颜色"""
        for i, cmd in enumerate(thrust_cmds):
            self.data.ctrl[self.act_ids[i]] = cmd
            
            force = self.data.actuator_force[self.act_ids[i]]
            if force < 0:
                self.model.site_rgba[self.site_ids[i]] = [1, 0, 0, 1]  # 负推力红色
            else:
                self.model.site_rgba[self.site_ids[i]] = [0, 0.5, 0.5, 1] # 正推力青色

    def set_thrusters_5dof(self, actions_5dof):
        """
        actions_5dof: [surge, heave, roll, pitch, yaw] 范围 [-1, 1]
        """
        # 1. 解包
        surge, heave, roll, pitch, yaw = actions_5dof

        # 2. 混合逻辑 (Mixer) - 计算出的值可能超过 [-1, 1]
        # 水平组 (t0-t3)
        # 注意：这里需要根据你的实际安装角度确认正负号
        # 假设：Yaw>0 右转 -> 左推(+, forward)，右拉(-, backward)
        t0 = surge - yaw # 右前
        t1 = surge + yaw # 左前
        t2 = surge - yaw # 右后
        t3 = surge + yaw # 左后

        # 垂直组 (t4-t7)
        t4 = heave - roll + pitch # 右前
        t5 = heave + roll + pitch # 左前
        t6 = heave - roll - pitch # 右后
        t7 = heave + roll - pitch # 左后

        raw_cmds = np.array([t0, t1, t2, t3, t4, t5, t6, t7])
        
        # 3. 【关键步骤】归一化处理
        # 找出绝对值最大的那个数
        max_val = np.max(np.abs(raw_cmds))
        
        # 如果最大的数超过 1.0，则所有数除以它
        # 例子：[1.5, 0.5] -> 除以 1.5 -> [1.0, 0.33]
        # 这样保持了 3:1 的比例，不会改变转向意图
        if max_val > 1.0:
            final_cmds = raw_cmds / max_val
        else:
            final_cmds = raw_cmds
            
        # 4. 下发给 MuJoCo
        # 因为你的 XML 里 ctrlrange="-1 1" 且 gear="155"，
        # 所以这里直接下发 [-1, 1] 的值即可。MuJoCo 会自动乘 155。
        for i, cmd in enumerate(final_cmds):
            self.data.ctrl[self.act_ids[i]] = cmd
            
            # 可视化颜色更新 (可选)
            if cmd < 0:
                self.model.site_rgba[self.site_ids[i]] = [1, 0, 0, 1] # 反转红
            else:
                self.model.site_rgba[self.site_ids[i]] = [0, 0.5, 0.5, 1] # 正转绿
        
        # 5. 【重要】返回这一步计算出的归一化指令，供 Env 计算能耗奖励
        return final_cmds
    
    def set_thrusters_6dof(self, actions_6dof):
        """
        修改说明：
        1. 保持函数名不变以兼容接口，但输入 actions_6dof 必须包含 6 个元素。
        2. 顺序: [surge, sway, heave, roll, pitch, yaw] 范围 [-1, 1]
        """
        # 1. 解包 (现在接受 6 个自由度，新增 Sway)
        surge, sway, heave, roll, pitch, yaw = actions_6dof

        # 2. 混合逻辑 (Mixer) - 6DOF 全驱动模式
        # -----------------------------------------------------------
        # 水平组 (t0-t3) - 45度安装 X型布局
        # 假设：
        # - Surge > 0: 整体向前 (所有推力 +)
        # - Sway  > 0: 整体向右 (左侧推力 +, 右侧推力 -) 需配合安装角度
        # - Yaw   > 0: 车头右转 (左侧推力 +, 右侧推力 -)
        
        # 这里的符号矩阵对应标准的 BlueROV2 矢量布局：
        # t0 (前右 HFR): 向后喷射带左分量 -> 需减去 Sway 和 Yaw
        t0 = surge - sway - yaw 
        
        # t1 (前左 HFL): 向后喷射带右分量 -> 需加上 Sway 和 Yaw
        t1 = surge + sway + yaw 
        
        # t2 (后右 HRR): 向后喷射带左分量 -> 需加上 Sway (因为在后方，向左喷产生向右力?) 
        # *修正逻辑*: 对于X型布局，后右推进器通常与前左平行。
        # 要横移向右(Sway > 0): t1(推), t2(推), t0(拉), t3(拉)
        # 要原地右转(Yaw > 0):  t1(推), t3(推), t0(拉), t2(拉)
        t2 = surge + sway - yaw 
        
        # t3 (后左 HRL): 与前右平行
        t3 = surge - sway + yaw 

        # -----------------------------------------------------------
        # 垂直组 (t4-t7) - 保持不变
        # -----------------------------------------------------------
        t4 = heave - roll + pitch # 右前
        t5 = heave + roll + pitch # 左前
        t6 = heave - roll - pitch # 右后
        t7 = heave + roll - pitch # 左后

        raw_cmds = np.array([t0, t1, t2, t3, t4, t5, t6, t7])
        
        # 3. 【关键步骤】归一化处理 (保持比例)
        max_val = np.max(np.abs(raw_cmds))
        if max_val > 1.0:
            final_cmds = raw_cmds / max_val
        else:
            final_cmds = raw_cmds
            
        # 4. 下发给 MuJoCo
        for i, cmd in enumerate(final_cmds):
            self.data.ctrl[self.act_ids[i]] = cmd
            
            # 可视化颜色更新
            if cmd < 0:
                self.model.site_rgba[self.site_ids[i]] = [1, 0, 0, 1] # 反转红
            else:
                self.model.site_rgba[self.site_ids[i]] = [0, 0.5, 0.5, 1] # 正转绿
        
        # 5. 返回指令供奖励函数计算
        return final_cmds

    def get_body_state(self):
        """获取 Body 系下的 6-DOF 速度"""
        rot_mat = self.data.xmat[self.body_id].reshape(3, 3)
        v_lin_world = self.data.qvel[0:3]
        v_ang_body = self.data.qvel[3:6]
        
        # 线速度转换: World -> Body
        v_lin_body = rot_mat.T @ v_lin_world
        return np.concatenate([v_lin_body, v_ang_body]), rot_mat

    def get_world_pose(self):
        """获取重心和浮心位置"""
        pos_com = self.data.xipos[self.body_id]
        pos_cob = self.data.site_xpos[self.cob_site_id]
        return pos_com, pos_cob