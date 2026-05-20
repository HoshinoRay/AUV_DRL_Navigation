import mujoco
import numpy as np


class YuyuanRobot:
    """
    Encapsulates actuator control and state queries for the Yuyuan AUV.

    Thruster layout (8 thrusters, index order matches the XML actuator list):
        t0 HFR  — horizontal front-right
        t1 HFL  — horizontal front-left
        t2 HRR  — horizontal rear-right
        t3 HRL  — horizontal rear-left
        t4 VFR  — vertical front-right
        t5 VFL  — vertical front-left
        t6 VRR  — vertical rear-right
        t7 VRL  — vertical rear-left
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.act_names = ["t0_hfr", "t1_hfl", "t2_hrr", "t3_hrl",
                          "t4_vfr", "t5_vfl", "t6_vrr", "t7_vrl"]
        self.act_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.act_names
        ]
        self.max_thrust = 155.0

        self.site_names = [f"thruster_{i}" for i in range(8)]
        self.site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in self.site_names
        ]

        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "yuyuan")
        self.cob_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cob_site")

    # ------------------------------------------------------------------
    # Actuator commands
    # ------------------------------------------------------------------

    def set_thrusters(self, thrust_cmds):
        """Apply raw per-thruster commands and update visualisation colours."""
        for i, cmd in enumerate(thrust_cmds):
            self.data.ctrl[self.act_ids[i]] = cmd
            force = self.data.actuator_force[self.act_ids[i]]
            # Red = reverse thrust, teal = forward thrust
            self.model.site_rgba[self.site_ids[i]] = (
                [1, 0, 0, 1] if force < 0 else [0, 0.5, 0.5, 1]
            )

    def set_thrusters_5dof(self, actions_5dof: np.ndarray) -> np.ndarray:
        """
        Mix a 5-DOF command [surge, heave, roll, pitch, yaw] ∈ [-1, 1]
        into 8 individual thruster commands.

        Returns the normalised per-thruster command array.
        """
        surge, heave, roll, pitch, yaw = actions_5dof

        # Horizontal group (X layout, no sway)
        t0 = surge - yaw   # front-right
        t1 = surge + yaw   # front-left
        t2 = surge - yaw   # rear-right
        t3 = surge + yaw   # rear-left

        # Vertical group
        t4 = heave - roll + pitch   # front-right
        t5 = heave + roll + pitch   # front-left
        t6 = heave - roll - pitch   # rear-right
        t7 = heave + roll - pitch   # rear-left

        raw_cmds = np.array([t0, t1, t2, t3, t4, t5, t6, t7])
        max_val = np.max(np.abs(raw_cmds))
        final_cmds = raw_cmds / max_val if max_val > 1.0 else raw_cmds

        for i, cmd in enumerate(final_cmds):
            self.data.ctrl[self.act_ids[i]] = cmd
            self.model.site_rgba[self.site_ids[i]] = (
                [1, 0, 0, 1] if cmd < 0 else [0, 0.5, 0.5, 1]
            )
        return final_cmds

    def set_thrusters_6dof(self, actions_6dof: np.ndarray) -> np.ndarray:
        """
        Mix a 6-DOF command [surge, sway, heave, roll, pitch, yaw] ∈ [-1, 1]
        into 8 individual thruster commands using an X-frame layout.

        Returns the normalised per-thruster command array.
        """
        surge, sway, heave, roll, pitch, yaw = actions_6dof

        # Horizontal group — X-frame BlueROV2-style layout
        t0 = surge - sway - yaw   # front-right
        t1 = surge + sway + yaw   # front-left
        t2 = surge + sway - yaw   # rear-right
        t3 = surge - sway + yaw   # rear-left

        # Vertical group
        t4 = heave - roll + pitch
        t5 = heave + roll + pitch
        t6 = heave - roll - pitch
        t7 = heave + roll - pitch

        raw_cmds = np.array([t0, t1, t2, t3, t4, t5, t6, t7])
        max_val = np.max(np.abs(raw_cmds))
        final_cmds = raw_cmds / max_val if max_val > 1.0 else raw_cmds

        for i, cmd in enumerate(final_cmds):
            self.data.ctrl[self.act_ids[i]] = cmd
            self.model.site_rgba[self.site_ids[i]] = (
                [1, 0, 0, 1] if cmd < 0 else [0, 0.5, 0.5, 1]
            )
        return final_cmds

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_body_state(self):
        """
        Returns
        -------
        vel_body : np.ndarray (6,) — body-frame velocity [u, v, w, p, q, r]
        rot_mat  : np.ndarray (3, 3) — world-to-body rotation matrix
        """
        rot_mat = self.data.xmat[self.body_id].reshape(3, 3)
        v_lin_world = self.data.qvel[0:3]
        v_ang_body = self.data.qvel[3:6]
        v_lin_body = rot_mat.T @ v_lin_world
        return np.concatenate([v_lin_body, v_ang_body]), rot_mat

    def get_world_pose(self):
        """
        Returns
        -------
        pos_com : np.ndarray (3,) — centre-of-mass world position
        pos_cob : np.ndarray (3,) — centre-of-buoyancy world position
        """
        return self.data.xipos[self.body_id], self.data.site_xpos[self.cob_site_id]
