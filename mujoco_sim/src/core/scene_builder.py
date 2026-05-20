import mujoco
import numpy as np


class SceneBuilder:
    """
    Manages dynamic obstacle placement for curriculum training.

    Obstacles are implemented as MuJoCo mocap bodies pooled in the XML.
    ``reset_scene`` repositions them each episode according to the
    current curriculum stage; unused obstacles are moved off-screen.

    Stage layout
    ------------
    0 — no obstacles (basic navigation)
    1 — single offset pillar (forces yaw correction)
    2 — single head-on pillar (forces active avoidance)
    3 — 3 random pillars (generalisation)
    4+ — fixed 8-pillar slalom course
    """

    HIDE_POS = np.array([999.0, 999.0, -999.0])
    FIXED_Z = 7.5   # half-height of pillars; centres them in the 0–15 m water column

    def __init__(self, model, data, max_obstacles: int = 8):
        self.model = model
        self.data = data
        self.max_obstacles = max_obstacles

        self._mocap_ids = []
        self._geom_ids = []

        for i in range(max_obstacles):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{i}")
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,  f"obs_geom_{i}")
            if body_id != -1 and geom_id != -1:
                self._mocap_ids.append(model.body_mocapid[body_id])
                self._geom_ids.append(geom_id)
            else:
                print(f"[SceneBuilder] Warning: obs_{i} not found in XML.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset_scene(self, stage: int, start_pos: np.ndarray, target_pos: np.ndarray):
        self._hide_all()
        mid_x = (start_pos[0] + target_pos[0]) / 2.0
        mid_y = (start_pos[1] + target_pos[1]) / 2.0

        if stage == 0:
            pass
        elif stage == 1:
            self._place(0, [mid_x, mid_y + 1.5], radius=0.8)
        elif stage == 2:
            self._place(0, [mid_x, mid_y], radius=1.2)
        elif stage == 3:
            self._random_layout(start_pos, target_pos, num_obs=3, radius_range=(0.6, 1.2))
        else:
            self._fixed_slalom()

    def get_active_obstacles(self) -> list:
        """
        Return a list of active obstacle dicts for the A* planner:
            [{"pos": [x, y], "radius": r}, ...]
        """
        active = []
        for i, mocap_id in enumerate(self._mocap_ids):
            pos = self.data.mocap_pos[mocap_id]
            if pos[2] != self.HIDE_POS[2]:
                active.append({
                    "pos": [pos[0], pos[1]],
                    "radius": self.model.geom_size[self._geom_ids[i]][0],
                })
        return active

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hide_all(self):
        for mocap_id in self._mocap_ids:
            self.data.mocap_pos[mocap_id] = self.HIDE_POS

    def _place(self, index: int, pos: list, radius: float):
        if index >= len(self._mocap_ids):
            return
        self.model.geom_size[self._geom_ids[index]][0] = radius
        self.data.mocap_pos[self._mocap_ids[index]] = np.array([pos[0], pos[1], self.FIXED_Z])

    def _fixed_slalom(self):
        """
        Hand-designed 8-pillar slalom from start (-13, 0) to goal (18, 0).
        Provides a visually compelling and reproducible evaluation course.
        """
        layout = [
            {"pos": [-5.6,  3.3], "radius": 1.2},
            {"pos": [-3.0, -3.1], "radius": 1.5},
            {"pos": [ 1.7,  0.75],"radius": 1.0},
            {"pos": [ 4.0, -1.2], "radius": 1.8},
            {"pos": [ 9.0,  6.9], "radius": 1.5},
            {"pos": [ 9.0, -5.3], "radius": 1.5},
            {"pos": [13.0,  1.3], "radius": 1.2},
            {"pos": [16.0,  3.1], "radius": 0.8},
        ]
        for i, obs in enumerate(layout):
            if i < len(self._mocap_ids):
                self._place(i, obs["pos"], obs["radius"])

    def _random_layout(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        num_obs: int,
        radius_range: tuple,
    ):
        """
        Place obstacles randomly with rejection sampling to keep start/goal
        regions clear.
        """
        SAFE_SPAWN  = 4.0   # min distance from start
        SAFE_TARGET = 4.0   # min distance from goal
        placed = []

        for i in range(min(num_obs, len(self._mocap_ids))):
            radius = np.random.uniform(*radius_range)
            for _ in range(50):
                rand_x = np.random.uniform(start_pos[0] + 5, target_pos[0] - 5)
                rand_y = np.random.uniform(-10.0, 10.0)
                test_xy = np.array([rand_x, rand_y])

                if (np.linalg.norm(test_xy - start_pos[:2])  < SAFE_SPAWN  + radius or
                        np.linalg.norm(test_xy - target_pos[:2]) < SAFE_TARGET + radius):
                    continue
                if any(np.linalg.norm(test_xy - p) < radius * 2 for p in placed):
                    continue

                self._place(i, [rand_x, rand_y], radius)
                placed.append(test_xy)
                break
