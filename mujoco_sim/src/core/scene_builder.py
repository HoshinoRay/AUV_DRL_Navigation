import numpy as np
import mujoco

class SceneBuilder:
    """
    环境场景生成器：负责在不同训练阶段 (Curriculum Stage) 动态配置障碍物。
    基于 MuJoCo 的 Mocap Object Pooling 技术。
    """
    def __init__(self, model, data, max_obstacles=8):
        self.model = model
        self.data = data
        self.max_obstacles = max_obstacles
        
        # 缓存障碍物的 Mocap ID 和 Geom ID 以提高运行效率
        self.obs_mocap_ids = []
        self.obs_geom_ids = []
        
        for i in range(self.max_obstacles):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{i}")
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_geom_{i}")
            
            if body_id != -1 and geom_id != -1:
                # MuJoCo 中 mocap body 有专门的数组索引
                mocap_id = self.model.body_mocapid[body_id]
                self.obs_mocap_ids.append(mocap_id)
                self.obs_geom_ids.append(geom_id)
            else:
                print(f"[SceneBuilder] Warning: obs_{i} not found in XML!")

        # 隐藏坐标 (垃圾回收区)
        self.HIDE_POS = np.array([999.0, 999.0, -999.0])
        # Z 坐标固定为 7.5 (由于半高是7.5，加上7.5刚好贯穿 0~15m 的水体)
        self.FIXED_Z = 7.5

    def reset_scene(self, stage, start_pos, target_pos):
        self._hide_all_obstacles()
        mid_x = (start_pos[0] + target_pos[0]) / 2.0
        mid_y = (start_pos[1] + target_pos[1]) / 2.0

        if stage == 0:
            # Stage 0: 毫无障碍，只需学会基础向目标导航
            pass
            
        elif stage == 1:
            # Stage 1: 轻微的偏移障碍，擦边就能过，引导出“偏航”的动作
            # 偏离 Y 轴 1.5 米，半径较小
            self._place_obstacle(index=0, pos=[mid_x, mid_y + 1.5], radius=0.8)

        elif stage == 2:
            # Stage 2: 正面拦截，逼迫 AUV 必须主动变道躲避
            self._place_obstacle(index=0, pos=[mid_x, mid_y], radius=1.2)

        elif stage == 3:
            # Stage 3: 随机散布 3 个障碍物 (泛化训练)
            self._generate_random_layout(start_pos, target_pos, num_obs=3, radius_range=(0.6, 1.2))

        elif stage >= 4:
            # Stage 4: 终极固定 8 柱阵
            self._generate_fixed_layout()

    def _hide_all_obstacles(self):
        """将所有障碍物传送到不可见的远方"""
        for mocap_id in self.obs_mocap_ids:
            self.data.mocap_pos[mocap_id] = self.HIDE_POS

    def _place_obstacle(self, index, pos, radius):
        """
        放置单个障碍物并动态修改其粗细 (Radius)
        pos: [x, y]
        """
        if index >= len(self.obs_mocap_ids):
            return
            
        mocap_id = self.obs_mocap_ids[index]
        geom_id = self.obs_geom_ids[index]
        
        # 1. 动态修改圆柱体半径 (MuJoCo 中 size 数组的第0个元素是半径，第1个是半高)
        self.model.geom_size[geom_id][0] = radius
        
        # 2. 传送位置
        self.data.mocap_pos[mocap_id] = np.array([pos[0], pos[1], self.FIXED_Z])

    def _generate_fixed_layout(self):
        """
        【新增方法】固定赛道生成器：
        针对起点 (-13, 0) 到 终点 (18, 0) 精心设计的 8 柱阵。
        不仅适合稳定训练，拍出的 Demo 也具有极强的观赏性（S弯 + 穿越窄门）。
        """
        # 定义 8 个障碍物的绝对坐标 [x, y] 和 半径 radius
        fixed_obstacles = [
            {"pos": [-5.6,  3.3], "radius": 1.2},  
            {"pos": [-3.0, -3.1], "radius": 1.5},  
            {"pos":[ 1.0,  1.3], "radius": 1.0},  
            {"pos": [ 4.0, -1.3], "radius": 1.8},  
            {"pos":[ 9.0,  6.9], "radius": 1.5},  
            {"pos": [ 9.0, -5.3], "radius": 1.5},  
            
            {"pos":[13.0,  0.0], "radius": 1.2},  
            {"pos":[16.0,  3.1], "radius": 0.8},  
        ]
        
        # 遍历放置
        for i, obs in enumerate(fixed_obstacles):
            if i < len(self.obs_mocap_ids):
                self._place_obstacle(index=i, pos=obs["pos"], radius=obs["radius"])

    def _generate_random_layout(self, start_pos, target_pos, num_obs, radius_range):
        """
        带拒绝采样 (Rejection Sampling) 的随机生成器
        确保障碍物不会堵死起点和终点。
        """
        placed_positions = []
        safe_radius_around_auv = 4.0   # 离出生点至少 4m 远
        safe_radius_around_target = 4.0# 离目标点至少 4m 远

        for i in range(min(num_obs, len(self.obs_mocap_ids))):
            radius = np.random.uniform(radius_range[0], radius_range[1])
            valid_pos = False
            
            # 尝试 50 次找一个合法位置，找不到就妥协 (防止死循环)
            for _ in range(50):
                # 假设起点 X=0, 终点 X=30. 我们在 X: [5, 25], Y: [-10, 10] 范围内生成
                rand_x = np.random.uniform(start_pos[0] + 5, target_pos[0] - 5)
                rand_y = np.random.uniform(-10.0, 10.0)
                test_pos = np.array([rand_x, rand_y, self.FIXED_Z])
                
                # 检查距离
                dist_to_start = np.linalg.norm(test_pos[:2] - start_pos[:2])
                dist_to_target = np.linalg.norm(test_pos[:2] - target_pos[:2])
                
                if dist_to_start > (safe_radius_around_auv + radius) and \
                   dist_to_target > (safe_radius_around_target + radius):
                    # 检查障碍物之间不要互相完全重叠
                    overlap = False
                    for p in placed_positions:
                        if np.linalg.norm(test_pos[:2] - p) < radius * 2:
                            overlap = True
                            break
                    
                    if not overlap:
                        valid_pos = test_pos
                        break
            
            if valid_pos is not False:
                self._place_obstacle(index=i, pos=[valid_pos[0], valid_pos[1]], radius=radius)
                placed_positions.append(valid_pos[:2])
    def get_active_obstacles(self):
        """
        向外界(例如A*规划器)暴露当前场景中激活的障碍物位置和半径
        返回格式: [{"pos": [x, y], "radius": r}, ...]
        """
        active_obs =[]
        for i, mocap_id in enumerate(self.obs_mocap_ids):
            # 获取 MuJoCo 中的坐标
            pos = self.data.mocap_pos[mocap_id]
            # 如果 Z 坐标不是隐藏坐标，说明是激活状态的
            if pos[2] != self.HIDE_POS[2]: 
                geom_id = self.obs_geom_ids[i]
                radius = self.model.geom_size[geom_id][0]
                active_obs.append({
                    "pos": [pos[0], pos[1]],
                    "radius": radius
                })
        return active_obs