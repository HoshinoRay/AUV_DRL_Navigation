import csv
import os
from datetime import datetime


class DataLogger:
    """
    Writes per-step simulation data (time, velocity, hydrodynamic forces)
    to a timestamped CSV file for offline analysis.
    """

    HEADER = [
        "Time",
        "u", "v", "w", "p", "q", "r",
        "Fx_H", "Fy_H", "Fz_H", "Tx_H", "Ty_H", "Tz_H",
    ]

    def __init__(self, log_dir: str = "./logs", prefix: str = "hydro_debug"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(log_dir, f"{prefix}_{timestamp}.csv")
        self._file = open(self.filename, mode='w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        print(f"[DataLogger] Logging to {self.filename}")

    def log(self, time: float, velocity: list, forces: list):
        """Append one row: [time, 6 velocities, 6 forces]."""
        self._writer.writerow([time] + list(velocity) + list(forces))

    def close(self):
        self._file.close()
        print(f"[DataLogger] Log saved: {self.filename}")
