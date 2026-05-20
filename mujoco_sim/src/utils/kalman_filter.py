import numpy as np


class KalmanFilter6D:
    """
    Constant-acceleration Kalman filter for 6-DOF velocity signals.

    Each degree of freedom is tracked independently with a 2-state model
    [velocity, acceleration]. The filter is used to estimate body-frame
    accelerations from noisy MuJoCo velocity readings, which are then
    passed to the hydrodynamics plugin.

    Parameters
    ----------
    dt            : simulation timestep (seconds)
    process_noise : Q scale — higher values allow faster acceleration changes
    measure_noise : R scale — higher values trust the velocity measurement less
    """

    def __init__(self, dt: float, process_noise: float = 0.1, measure_noise: float = 0.01):
        self.dt = dt

        # State: [vel, accel] for each of 6 DOFs
        self.x = np.zeros((6, 2))

        # Covariance matrices (one per DOF)
        self.P = np.array([np.eye(2) for _ in range(6)])

        # State transition: constant-acceleration model
        self.F = np.array([[1.0, dt], [0.0, 1.0]])

        # Observation: we only measure velocity
        self.H = np.array([1.0, 0.0])

        # Process noise covariance
        self.Q = np.array([
            [0.25 * dt ** 4, 0.5 * dt ** 3],
            [0.5  * dt ** 3, dt ** 2],
        ]) * process_noise

        # Measurement noise variance
        self.R = measure_noise

    def update(self, v_meas: np.ndarray) -> np.ndarray:
        """
        Run one predict-update cycle and return the estimated accelerations.

        Parameters
        ----------
        v_meas : (6,) array of measured body-frame velocities

        Returns
        -------
        estimated_accel : (6,) array of filtered accelerations
        """
        estimated_accel = np.zeros(6)
        for i in range(6):
            # Predict
            x_pred = self.F @ self.x[i]
            P_pred = self.F @ self.P[i] @ self.F.T + self.Q

            # Update (scalar measurement)
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T / S
            y = v_meas[i] - self.H @ x_pred        # innovation
            self.x[i] = x_pred + K * y
            self.P[i] = (np.eye(2) - np.outer(K, self.H)) @ P_pred

            estimated_accel[i] = self.x[i][1]

        return estimated_accel
