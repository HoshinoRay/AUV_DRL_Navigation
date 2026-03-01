import torch
import joblib
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from src.core.scene_builder import SceneBuilder
from src.core.robot import YuyuanRobot
from src.core.hydro_plugin import HydroDynamicsPlugin
from src.core.sensors import SensorManager

# 任务注册表，用于动态加载
from src.envs.tasks import TASK_REGISTRY

class AUVGymEnv(gym.Env):
    """
    AUV 仿真环境容器 (Container)。
    1. 管理 MuJoCo 物理引擎和渲染。
    2. 管理底层组件 (Robot, Hydro, Sensors)。
    3. 将逻辑委托给 Task 实例 (Reward, Done, Obs)。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg_env, cfg_task):
        """
        Args:
            cfg_env: 来自 configs/env/xxx.yaml 的配置 (物理参数)
            cfg_task: 来自 configs/task/xxx.yaml 的配置 (任务逻辑)
        """
        super().__init__()
        
        # 1. 保存配置
        self.cfg_env = cfg_env
        self.cfg_task = cfg_task
        
        # 2. 物理引擎初始化
        # 这里的 xml_path 从 config 读取，不要硬编码
        self.model = mujoco.MjModel.from_xml_path(self.cfg_env.xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.render_mode = self.cfg_env.render_mode

        # 3. 核心组件实例化
        # 机器人本体封装
        self.robot = YuyuanRobot(self.model, self.data)
        
        # 传感器管理器
        self.sensors = SensorManager(self.model, self.data)
        self.WATER_SURFACE_Z = self.sensors.WATER_SURFACE_Z # 方便外部访问

        self.scene_builder = SceneBuilder(self.model, self.data, max_obstacles=10)
        
        # 动态读取初始 Stage，如果配置里没有，就默认给 0 (兼容旧任务)
        if hasattr(self.cfg_task, 'curriculum'):
            self.current_stage = self.cfg_task.curriculum.initial_stage
        else:
            self.current_stage = 0
            
        print(f"🌍 环境初始化完毕，当前出生在 Stage: {self.current_stage}")

        # 水动力插件 (权重路径建议也在 config 里管理，这里暂且假设 env config 里有 path 或直接传参)
        # 为简化，这里假设 model_weights 已经通过某种方式传入，或者在 config 中指定了路径加载
        # 在实际工程中，通常是在 train.py 里加载好 weights 传进来，或者在这里 load
        # 这里为了保持接口简洁，我们假设 weights 已被加载到 cfg_env 中，或者硬编码路径 (暂且保留原逻辑的变体)
        # *注意*：为了让代码跑通，建议在 cfg_env 中增加 model_weights 字段，或者在外部加载
        self.hydro = HydroDynamicsPlugin(
            # 这里需要你确保在 cfg_env 或外部加载了这些权重
            # 如果不想太复杂，暂时可以保留原有的加载逻辑，但路径从 cfg 读取
            self.cfg_env.weights.mlp,      # 直接传字符串路径
            self.cfg_env.weights.scaler_x, # 直接传字符串路径
            self.cfg_env.weights.scaler_y, # 直接传字符串路径
            self.dt
        )

        # 4. 【核心】动态加载任务
        if self.cfg_task.name not in TASK_REGISTRY:
            raise ValueError(f"Task '{self.cfg_task.name}' not found in registry: {list(TASK_REGISTRY.keys())}")
        
        TaskClass = TASK_REGISTRY[self.cfg_task.name]
        print(f"🔄 Loading Task: {self.cfg_task.name}")
        self.task = TaskClass(self.cfg_task)

        # 5. 定义 Gym 空间
        # Action Space: 6DOF 推力 [-1, 1]
        self.n_actions = 6
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation Space: 由 Task 决定维度
        obs_dim = self.task.get_obs_dim()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 6. 其他参数
        self.max_steps = self.cfg_env.max_steps
        self.current_step = 0
        
        # 缓存 Body ID 和质量
        self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "yuyuan")
        self.base_mass = self.model.body_mass[self.robot_body_id]

        # --- 抓取目标小球的 Mocap ID ---
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        # MuJoCo 中 mocap 有专门的 ID 索引池
        self.target_mocap_id = self.model.body_mocapid[target_body_id] if target_body_id != -1 else -1
        
        # --- 设置基础目标位置 (水平 Z=10) ---
        self.fixed_start_pos = np.array(self.cfg_task.goals.start_pos)
        self.base_target_pos = np.array(self.cfg_task.goals.target_pos)
        self.fixed_target_pos = self.base_target_pos.copy()
        self.target_pos = self.fixed_target_pos

    def set_stage(self, stage: int):
        """外部接口：供训练脚本调用的升级函数"""
        print(f"🌊 [Curriculum] Environment upgrading to Stage {stage}")
        self.current_stage = stage

    def _load_weight(self, path, pickle=False):
        """辅助函数：加载权重"""
        if not os.path.exists(path):
            # 为了防止报错，这里可以写一个 fallback 或者 raise error
            # 假设路径是相对于项目根目录
            raise FileNotFoundError(f"Weight file not found: {path}")
            
        if pickle:
            return joblib.load(path)
        else:
            return torch.load(path, map_location='cpu')

    def reset(self, seed=None, options=None):
        """
        重置环境：物理重置 + 随机化 + 任务重置
        """
        super().reset(seed=seed)
        
        # 1. 物理重置
        mujoco.mj_resetData(self.model, self.data)
        self.hydro.reset()
        self.current_step = 0
        self.model.body_mass[self.robot_body_id] = self.base_mass

        # 2. 域随机化 (Domain Randomization) - 从 Config 读取参数
        # cfg_env.randomization.pos_noise -> [1.0, 1.0, 0.5]
        rand_cfg = self.cfg_env.randomization
        
        # 位置噪声
        pos_noise_range = np.array(rand_cfg.pos_noise) # [x, y, z]
        pos_noise = np.random.uniform(-pos_noise_range, pos_noise_range)
        start_pos = self.fixed_start_pos + pos_noise
        
        # 边界保护
        if start_pos[2] > 14.0: start_pos[2] = 14.0
        self.data.qpos[0:3] = start_pos

        # 姿态噪声 (Roll, Pitch, Yaw)
        angle_noise_range = np.array(rand_cfg.angle_noise) # [r, p, y]
        # 随机生成欧拉角
        rand_euler = np.random.uniform(-angle_noise_range, angle_noise_range)
        
        # 转换为四元数
        quat = self._euler_to_quat(rand_euler[0], rand_euler[1], rand_euler[2])
        self.data.qpos[3:7] = quat

        # ==========================================
        # 在 Config 中加上 pos_noise 后，这里生成带噪声的真实 target
        rand_cfg = self.cfg_env.randomization
        pos_noise_range = np.array(rand_cfg.pos_noise)
        target_noise = np.random.uniform(-pos_noise_range, pos_noise_range)
        self.fixed_target_pos = self.base_target_pos + target_noise
        self.target_pos = self.fixed_target_pos

        # 根据当前 Stage 布置障碍物
        self.scene_builder.reset_scene(
            stage=self.current_stage, 
            start_pos=start_pos, 
            target_pos=self.target_pos
        )
        # 强制 MuJoCo 更新一次正向运动学，让障碍物的位置瞬间生效
        mujoco.mj_forward(self.model, self.data)
        # ==========================================

        # 3. 预热 (Warmup) - 让水动力和滤波器稳定
        for _ in range(20):
            self.hydro.apply_hydrodynamics(self.robot)
            mujoco.mj_step(self.model, self.data)

        # ==========================================
        # 4. 生成这一局的真实目标 (加入随机扰动)
        # ==========================================
        rand_cfg = self.cfg_env.randomization
        pos_noise_range = np.array(rand_cfg.pos_noise) # 例如 [1.5, 1.5, 1.0]
        
        # 生成真正的随机偏差
        target_noise = np.random.uniform(-pos_noise_range, pos_noise_range)
        
        # 覆盖 fixed_target_pos，这是你 Agent 真正要去的坐标！
        self.fixed_target_pos = self.base_target_pos + target_noise

        # ==========================================
        # 5. 任务重置 (你的 BaseTask 会把 env.target_pos 刷新为上面的真实坐标)
        # ==========================================
        self.task.reset(self)

        # ==========================================
        # 6. 💥 强制同步视觉小球 (直接写底层 Data，绝不延迟！)
        # ==========================================
        if self.target_mocap_id != -1:
            # 直接改 data，不改 model。viewer.sync() 瞬间就能捕捉到位置变化
            self.data.mocap_pos[self.target_mocap_id] = self.target_pos

        # 7. 获取观测
        obs = self._get_obs()
        
        return obs, {}

    def step(self, action):
        self.current_step += 1
        clamped_action = np.clip(action, -1.0, 1.0)
        
        # 1. 应用动作
        self.robot.set_thrusters_6dof(clamped_action)
        self.hydro.apply_hydrodynamics(self.robot)

        applied_force_cache = self.data.xfrc_applied.copy()
        
        # 2. 物理步进 (Frame Skip 从 Config 读取)
        for _ in range(self.cfg_env.frame_skip):
        # 每次底层 step 前，重新填入外部力
            self.data.xfrc_applied = applied_force_cache
            mujoco.mj_step(self.model, self.data)
        
        # 3. 获取观测 (委托给 Task)
        obs = self._get_obs()
        
        # 4. 计算奖励与终止 (委托给 Task)
        reward, is_success, reward_info = self.task.compute_reward(self, action, obs)
        terminated, reason = self.task.is_done(self, self.current_step, self.max_steps)
        truncated = (reason == "timeout")

        # 5. 组装最终 Info
        # [修改处]: 将 reward_info 合并进来
        info = {
            # --- 环境自带信息 ---
            "termination_reason": reason, # 为什么停了(超时/撞击/成功)
            "stats/mean_thrust": np.mean(np.abs(clamped_action)), # 平均推力(需确保 robot 有此属性或计算 action)
            
            # --- 合并 Task 的详细信息 ---
            **reward_info  # Python 解包语法，把 task 返回的字典合并进来
        }
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        获取观测：不再自己拼装，而是问 Task 要。
        Env 负责提供原始数据，Task 负责挑选和处理。
        """
        return self.task.get_obs(self)

    def render(self):
        if self.render_mode == "human":
            # 这里可以添加简单的 viewer 代码，或者依赖外部 wrapper
            pass

    def _euler_to_quat(self, roll, pitch, yaw):
        """辅助函数：欧拉角转四元数"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])