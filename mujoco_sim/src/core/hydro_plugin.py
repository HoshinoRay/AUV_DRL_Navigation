import numpy as np
import torch
import joblib

from src.utils.kalman_filter import KalmanFilter6D
from src.core.models import DeepHydroMLP


class HydroInference:
    """
    Wraps DeepHydroMLP for online inference inside the MuJoCo step loop.

    Coordinate convention
    ---------------------
    The MLP was trained on Stonefish (NED) data; MuJoCo uses ENU.
    ``coord_mask`` flips the axes that differ between the two frames
    before feeding the network and after reading its output.
    """

    # ENU <-> NED sign flip for [Surge, Sway, Heave, Roll, Pitch, Yaw]
    COORD_MASK = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])

    # Input clipping bounds (physical saturation limits)
    MAX_LIN_VEL = 1.3   # m/s
    MAX_ANG_VEL = 1.0   # rad/s

    def __init__(self, model_path: str, scaler_x_path: str, scaler_y_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.model = DeepHydroMLP(input_dim=12, output_dim=6).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, velocity: np.ndarray, acceleration: np.ndarray, gain: float = -1.0) -> np.ndarray:
        """
        Predict 6-DOF hydrodynamic force in the MuJoCo (ENU) body frame.

        Parameters
        ----------
        velocity     : 6-DOF body-frame velocity [u, v, w, p, q, r] in ENU
        acceleration : 6-DOF body-frame acceleration from the Kalman filter
        gain         : scalar multiplier applied to the raw network output
                       (use -1.0 to convert from ``force on fluid'' to
                       ``force on body'' convention)

        Returns
        -------
        np.ndarray of shape (6,), clipped to [-180, 180] N / Nm
        """
        vel_ned = velocity * self.COORD_MASK
        acc_ned = acceleration * self.COORD_MASK
        vel_ned[:3] = np.clip(vel_ned[:3], -self.MAX_LIN_VEL, self.MAX_LIN_VEL)
        vel_ned[3:] = np.clip(vel_ned[3:], -self.MAX_ANG_VEL, self.MAX_ANG_VEL)

        input_vector = np.concatenate([vel_ned, acc_ned]).reshape(1, -1)
        input_scaled = self.scaler_x.transform(input_vector)
        with torch.no_grad():
            tensor_out = self.model(torch.tensor(input_scaled, dtype=torch.float32).to(self.device))
        output_force = self.scaler_y.inverse_transform(tensor_out.cpu().numpy())

        force_enu = output_force * self.COORD_MASK
        return np.clip(force_enu * gain, -180.0, 180.0).flatten()


class HydroDynamicsPlugin:
    """
    Applies hydrodynamic forces (drag, added mass, buoyancy) to the AUV
    body at every simulation step.

    Two operating modes are available for comparative experiments:

    simplified_mode=True  (default)
        Only added-mass forces are applied; the MLP drag prediction is
        zeroed out.  Use this as a physics-only baseline.

    simplified_mode=False
        Full MLP-predicted drag + added-mass forces.  Surge drag is
        overridden with a tuned quadratic model to account for the
        known sway-decoupled layout of the horizontal thrusters.
    """

    # Added-mass coefficients [Surge, Sway, Heave, Roll, Pitch, Yaw]
    MA = np.array([0.0, 102.31, 121.68, 0.0, 0.0, 0.0])

    BUOYANCY_MAGNITUDE = 1790.0  # N
    VEL_DEADBAND = 0.05          # m/s — suppress drag noise at near-zero velocity

    def __init__(
        self,
        model_path: str,
        scaler_x: str,
        scaler_y: str,
        dt: float,
        simplified_mode: bool = True,
    ):
        self.dt = dt
        self.simplified_mode = simplified_mode
        self.predictor = HydroInference(model_path, scaler_x, scaler_y)
        self.kf = None
        self.reset()

    def reset(self):
        """Reset the Kalman filter and surge-noise state between episodes."""
        self.kf = KalmanFilter6D(dt=self.dt, process_noise=10.0, measure_noise=0.001)
        self.surge_noise_state = 0.0
        self._noise_smoothing = 0.1

    def apply_hydrodynamics(self, robot):
        """
        Compute and apply all hydrodynamic forces to the robot body.

        Returns
        -------
        current_vel_body : np.ndarray (6,)  — body-frame velocity
        hydro_force_body : np.ndarray (6,)  — MLP drag term (body frame)
        total_hydro_body : np.ndarray (6,)  — drag + added mass (body frame)
        """
        data = robot.data
        current_vel_body, rot_mat = robot.get_body_state()
        accel_body = self.kf.update(current_vel_body)

        hydro_force_body = np.zeros(6)

        if self.simplified_mode:
            # Baseline: added-mass only; MLP drag disabled
            total_hydro_body = -self.MA * accel_body
        else:
            # Full mode: MLP drag + added mass
            hydro_force_body = self.predictor.predict(current_vel_body, accel_body, gain=-1.0)

            # Override Surge (index 0) with a tuned quadratic drag model to
            # account for the decoupled horizontal-thruster geometry.
            u = current_vel_body[0]
            base_drag = 30.0 * abs(u) + 12.0 * u ** 2
            raw_noise = np.random.uniform(-35.0, 35.0)
            self.surge_noise_state = (
                (1 - self._noise_smoothing) * self.surge_noise_state
                + self._noise_smoothing * raw_noise
            )
            total_surge = np.clip(base_drag + self.surge_noise_state, 0, 200.0)
            hydro_force_body[0] = -np.sign(u) * total_surge if abs(u) > 1e-3 else 0.0

            # Velocity dead-band: suppress near-zero drag noise
            for i in range(6):
                vel_abs = abs(current_vel_body[i])
                if vel_abs < self.VEL_DEADBAND:
                    hydro_force_body[i] *= vel_abs / self.VEL_DEADBAND

            added_mass = -self.MA * accel_body
            total_hydro_body = 0.7 * hydro_force_body + added_mass

        # Transform to world frame
        f_hydro_world = rot_mat @ total_hydro_body[:3]
        t_hydro_world = rot_mat @ total_hydro_body[3:]

        # Buoyancy (always applied)
        pos_com, pos_cob = robot.get_world_pose()
        f_buoyancy = np.array([0.0, 0.0, self.BUOYANCY_MAGNITUDE])
        t_buoyancy = np.cross(pos_cob - pos_com, f_buoyancy)

        data.xfrc_applied[robot.body_id] = np.concatenate([
            f_buoyancy + f_hydro_world,
            t_buoyancy + t_hydro_world,
        ])

        return current_vel_body, hydro_force_body, total_hydro_body
