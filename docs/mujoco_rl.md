既然你选择了 **SAC (Soft Actor-Critic)**，这是一个非常棒的选择！对于 AUV 这种连续控制、高维动作空间、且动力学复杂的机器人，SAC 通常比 PPO 表现更稳健，采样效率也更高。

你说“感知部分弄完了”，这很好。但对于从未做过 RL 的人来说，最容易产生的误解是：“我都把环境搭好了，把 SAC 算法库一调，不就训练出来了吗？”

**绝对不是。** 如果环境是舞台，感知是眼睛，算法是智商，那么接下来的工作就是**“教学大纲的设计”**和**“性格的塑造”**。

剩下的工作量主要集中在以下 **4 个核心技术深水区**。这不仅是写代码，而是对物理和数学的深刻理解。

---

### 一、 核心工作 1：奖励函数工程 (Reward Engineering) —— 工作的 60%

这是 RL 的灵魂。SAC 算法本身是个黑盒优化器，它只懂一样东西：**“贪婪地最大化分数的总和”**。

你面临的最大技术挑战是：**如何把复杂的 AUV 驾驶技巧，翻译成一个数学公式？**

如果你的奖励函数写得不好，会出现以下**“经典翻车现场”**：

1. **只会转圈**：因为你只奖励了“速度”，它发现原地转圈线速度也很大。
2. **像帕金森一样抖动**：因为你没惩罚“动作切换幅度”，它发现高频切换电机能维持微妙平衡。
3. **永远到不了终点**：因为它发现只要活着就有分，所以它选择原地漂浮，不冒风险去撞墙。

#### 深度技术拆解：AUV 的复合奖励函数设计

你需要设计一个 ，它至少包含以下几项（权重  需要反复调参）：

```python
def compute_reward(self, obs, action, prev_action):
    # 1. 追踪奖励 (Tracking Reward)
    # 这是一个“密集奖励 (Dense Reward)”，每一步都要给
    # 不要用线性的 dist，要用指数或者高斯核，引导它在靠近目标时精细操作
    dist = np.linalg.norm(self.target_pos - self.robot_pos)
    r_pos = np.exp(-1.0 * dist) 

    # 2. 姿态奖励 (Orientation/Upright Reward)
    # AUV 必须保持平稳，不能翻滚 (Roll) 或 剧烈俯仰 (Pitch)
    # 技术点：使用四元数计算与 [0,0,0,1] 的余弦距离，或者直接惩罚欧拉角
    roll, pitch, yaw = self.robot.get_euler()
    r_orient = - (abs(roll) + abs(pitch)) 

    # 3. 能耗惩罚 (Energy Penalty)
    # 限制推力大小，防止电机过热
    r_energy = - np.sum(np.square(action))

    # 4. 动作平滑惩罚 (Action Smoothness) —— 极其重要！
    # 惩罚这一帧动作和上一帧动作的差值。
    # 如果没有这一项，Sim-to-Real 必死，真机会疯狂抖动。
    r_smooth = - np.sum(np.square(action - prev_action))

    # 5. 存活奖励 (Survival Reward)
    # 只要没翻车、没撞墙，就给一点点正分，鼓励它活着
    r_alive = 0.1

    # 总分
    reward = w1*r_pos + w2*r_orient + w3*r_energy + w4*r_smooth + r_alive
    return reward

```

**你的工作重心**：**调参 (Hyperparameter Tuning)**。你会发现  太大它就乱冲， 太大它就原地不敢动。这需要大量的实验。

---

### 二、 核心工作 2：动作空间映射 (The Mixer) —— 工作的 20%

SAC 输出的是一个向量。但这个向量代表什么？这决定了训练的难度。

#### 方案 A：端到端 (End-to-End)

* **做法**：SAC 直接输出 8 个数字，对应 8 个电机的 PWM。
* **分析**：**极难收敛**。因为 AUV 的动力学是高度耦合的（比如想前进，需要 T0, T1, T2, T3 同时发力）。让神经网络去“猜”出这个物理耦合关系，效率极低。

#### 方案 B：解耦控制 (Hierarchical Control) —— **强烈推荐**

* **做法**：
1. **SAC 输出 6 个数**：代表虚拟力/力矩 。
2. **中间层 (Mixer)**：写死一个矩阵（基于你的潜艇物理布局）。
3. **计算**：。


* **技术点**：你需要根据 `yuyuan.xml` 里的推进器位置，推导出一个 **推力分配矩阵 (Thruster Allocation Matrix)**。
* **RL 的工作**：在这个方案下，RL 只需要学习“我想往前走”，而不需要学习“怎么配合 8 个电机来往前走”。这大大降低了学习难度。

---

### 三、 核心工作 3：SAC 特有的“最大熵”调节 —— 工作的 10%

你没做过 RL，所以需要理解 SAC 相比于其他算法（如 DDPG, PPO）最牛的地方：**最大熵 (Maximum Entropy)**。

* **普通 RL**：寻找一条路，得分最高。
* **SAC**：寻找一条路，得分最高，且**动作尽可能随机（熵最大）**。

**为什么要随机？**
因为水下环境充满了未知扰动。

* 如果策略太“刚”（Deterministic），遇到一点水流它就傻了。
* SAC 训练出来的策略是“柔”的（Stochastic），它会有意保持一定的探索性。

**技术点：温度系数  (Alpha)**
SAC 有一个关键参数叫 Temperature ()。

* ** 大**：机器人像个疯子，疯狂乱试，探索能力强，但由于太抖，很难精确停在目标点。
* ** 小**：机器人像个老古板，只守着已知的路径，容易陷入局部最优。
* **你的工作**：SB3 库通常支持自动调整 ，但你需要监控它。如果训练初期它就不动了，可能需要调大初始 。

---

### 四、 核心工作 4：训练流程与课程学习 (Curriculum Learning) —— 工作的 10%

不要一上来就让它做“复杂轨迹跟踪”。它会崩溃的。

你需要设计一套**“课程表”**：

1. **幼儿园阶段**：
* **任务**：目标点就在眼前 1 米处。
* **目的**：学会最基本的“往前挪”和“保持平衡”。
* **现象**：一开始它会乱翻，几万步后能勉强凑过去。


2. **小学阶段**：
* **任务**：目标点随机出现在 5 米范围内。
* **目的**：学会转向、学会三维空间的机动。


3. **中学阶段**：
* **任务**：加入 `HydroDynamicsPlugin` 里的水流干扰和噪声。
* **目的**：**鲁棒性 (Robustness)**。这时候你会发现之前的策略不好用了，它需要对抗水流。



**技术实现**：
在 `GymAUV.reset()` 里，根据当前的 `epoch` 或 `total_timesteps`，动态调整任务难度（比如目标距离 `distance_range` 从 1.0 慢慢加到 10.0）。

---

### 总结：你的“RL 工程师”日程表

既然你感知写完了，接下来你的工作流是这样的：

1. **Day 1 (搭建)**: 写好 `GymAUV` 类，实现最简单的 `compute_reward`（只包含距离惩罚）。跑通 SAC 代码，看着潜艇在水里像无头苍蝇一样乱撞。
2. **Day 2 (受挫)**: 发现潜艇总是翻过来（Roll > 90），或者总是转圈。
3. **Day 3 (修整)**: **加入 Mixer 矩阵**，把 8 维动作改成 6 维力。**修改 Reward**，加大姿态惩罚 ()，加大动作平滑惩罚 ()。
4. **Day 4 (进阶)**: 终于能直走了！但是停不下来，会在目标点反复震荡。
5. **Day 5 (微调)**: 调整 Reward 里的距离函数（从线性改为指数），调整 SAC 的  参数。
6. **Day 6 (抗扰)**: 开启 `HydroDynamicsPlugin` 的全噪声模式，进行长时间训练（Sim-to-Real 准备）。

**最后的建议：**
RL 是**“三分算法，七分调参”**。不要因为第一次跑不通就觉得代码错了，大概率是 Reward 没写对，或者初始状态太难了。保持耐心，多看 Tensorboard 的曲线。


在惯性导航（Inertial Navigation）、机器人动力学（Robotics Dynamics）以及水下航行器（AUV）控制的学术研究中，对“加速度”的定义极其严格。混淆不同的加速度概念是导致仿真与实物测试不一致的最常见原因。

以下将按照学术论文的规范，为您详细阐述三种核心加速度的物理原理、数学定义及其转化关系。

---

### 1. 核心概念定义与符号约定

为了保证数学描述的严谨性，首先定义坐标系与符号：
*   **$\{I\}$ 惯性坐标系 (Inertial Frame)**:通常指世界坐标系（World/Global Frame），原点固定，遵循牛顿运动定律。MuJoCo 中的 World Frame。
*   **$\{B\}$ 机体坐标系 (Body Frame)**: 固连在机器人中心（通常是重心 CoG），随机器人旋转。MuJoCo 中的 Body/Sensor Frame。
*   **$\mathbf{R}_{I}^{B}$**: 从惯性系到机体系的旋转矩阵（Rotation Matrix），满足正交性 $\mathbf{R}_{I}^{B} = (\mathbf{R}_{B}^{I})^T$。
*   **$\mathbf{g}$**: 重力加速度矢量。

---

### 2. 运动学加速度 (Kinematic Acceleration)

**学术定义**：
运动学加速度是物体位置矢量对时间的二阶导数。它是描述物体“纯运动”状态的物理量，与受力分析中的 $F=ma$ 直接相关。

**物理原理**：
在惯性系 $\{I\}$ 下，位置 $\mathbf{p}$ 的二阶导数为：
$$ ^{I}\mathbf{a} = \frac{d^2}{dt^2}(^{I}\mathbf{p}) $$

然而，传感器和控制算法通常在机体系 $\{B\}$ 下工作。根据**科里奥利变换（Transport Theorem）**，机体系下的运动学加速度并不是简单的机体速度求导，还需要考虑旋转效应：

$$ ^{B}\mathbf{a}_{kin} = \underbrace{\dot{\mathbf{v}}_{body}}_{\text{线性加速度}} + \underbrace{\boldsymbol{\omega} \times \mathbf{v}_{body}}_{\text{科里奥利/向心项}} $$

*   **$\dot{\mathbf{v}}_{body}$**: 机体坐标系下线速度的变化率（MuJoCo 中 `qvel` 的直接差分近似项）。
*   **$\boldsymbol{\omega} \times \mathbf{v}_{body}$**: 由于坐标系旋转而产生的视加速度（对于低速 AUV 该项较小，但对于高机动航行器不可忽略）。

**在仿真中的地位**：
这是 Ground Truth (GT)。在 MuJoCo 中，这是物理引擎积分器计算出的真实结果。
$$ ^{B}\mathbf{a}_{kin} = \mathbf{R}_{I}^{B} \cdot (^{I}\mathbf{a}_{world}) $$

---

### 3. 比力 (Specific Force / Proper Acceleration)

**学术定义**：
比力是**加速度计（Accelerometer）**实际测量的物理量。根据爱因斯坦等效原理（Equivalence Principle），加速度计无法区分“惯性力”和“引力”。它测量的是弹簧质量块受到的**支持力**归一化后的值。

**物理公式**：
$$ \mathbf{f} = \mathbf{a}_{kin} - \mathbf{g} $$

**详细解析**：
1.  **静止状态**：
    物体静止在桌面上，运动学加速度 $\mathbf{a}_{kin} = 0$。
    $$ \mathbf{f} = 0 - \mathbf{g} = -\mathbf{g} $$
    如果重力 $\mathbf{g}$ 向下（例如 $[0, 0, -9.8]$），那么比力为 $[0, 0, 9.8]$。
    *物理意义*：桌面向上推传感器，传感器感受到向上的支持力。
2.  **自由落体**：
    物体做自由落体运动，$\mathbf{a}_{kin} = \mathbf{g}$。
    $$ \mathbf{f} = \mathbf{g} - \mathbf{g} = 0 $$
    *物理意义*：失重状态，传感器内部没有应力。

**在 MuJoCo 中的地位**：
这是 `sensor="accelerometer"` 的输出值。
$$ ^{B}\mathbf{f}_{meas} = ^{B}\mathbf{a}_{kin} - \mathbf{R}_{I}^{B} \cdot ^{I}\mathbf{g} $$

---

### 4. 杆臂效应 (Lever Arm Effect)

在实际工程和高精度仿真中，IMU 往往不安装在重心（CoG）上，而是安装在某个位置 $\mathbf{r}_{IMU}$。此时 IMU 测量的加速度包含**角加速度引起的切向分量**和**角速度引起的向心分量**。

**转化关系公式**：
$$ \mathbf{a}_{IMU} = \mathbf{a}_{CoG} + \underbrace{\dot{\boldsymbol{\omega}} \times \mathbf{r}_{IMU}}_{\text{切向加速度}} + \underbrace{\boldsymbol{\omega} \times (\boldsymbol{\omega} \times \mathbf{r}_{IMU})}_{\text{向心加速度}} $$

*   **$\mathbf{r}_{IMU}$**: IMU 相对于重心的位置矢量。
*   **$\dot{\boldsymbol{\omega}}$**: 角加速度。
*   **$\boldsymbol{\omega}$**: 角速度。

**在您项目中的意义**：
如果您的 XML 文件中 IMU `site` 定义的位置不在 `body` 的几何中心，MuJoCo 会自动计算这部分效应。如果您手动计算 Ground Truth 而忽略了这一点，会出现高频误差。

---

### 5. 总结：转化与验证逻辑表

为了在您的代码中实现严格的验证，请参考下表的转化逻辑。假设所有向量均已旋转至 **机体坐标系 (Body Frame)** 下。

| 物理量名称 | 符号 | 来源/计算方式 | 物理含义 |
| :--- | :--- | :--- | :--- |
| **真实运动加速度** | $\mathbf{a}_{kin}$ | `(v_t - v_{t-1}) / dt` (数值微分) <br> 或 `R * qacc` (直接读取) | 物体相对于惯性空间的绝对运动加速度。包含推力、阻力、附加质量力产生的效果。 |
| **重力投影** | $\mathbf{g}_{body}$ | $\mathbf{R}_{I}^{B} \cdot [0, 0, -9.81]^T$ | 重力矢量在当前机体姿态下的分量。 |
| **IMU 测量值** | $\mathbf{f}_{meas}$ | MuJoCo `sensors.get_imu()` | **比力**。包含了运动加速度，并抵消了重力场。 |
| **验证方程** | - | $\mathbf{f}_{meas} \approx \mathbf{a}_{kin} - \mathbf{g}_{body}$ | **这是校验 IMU 是否正常的唯一金标准。** |
| **导航控制输入** | $\mathbf{a}_{nav}$ | $\mathbf{f}_{meas} + \mathbf{g}_{body}$ | 在卡尔曼滤波中，需要将 IMU 读数**减去**（或加上负的）重力分量，恢复成运动加速度 $\mathbf{a}_{kin}$。 |

### 6. 针对您 AUV 项目的学术建议

在您的 `HydroDynamicsPlugin` 和 `KalmanFilter` 中，存在一个闭环的因果关系链：

1.  **力学输入**: 水动力 + 推力 + 重力 + 浮力 = 合力 $F_{total}$。
2.  **动力学方程 (Dynamics)**:
    $$ \mathbf{M}_{RB}\dot{\mathbf{v}} + \mathbf{C}_{RB}(\mathbf{v})\mathbf{v} + \underbrace{\mathbf{M}_A\dot{\mathbf{v}} + \mathbf{C}_A(\mathbf{v})\mathbf{v}}_{\text{Added Mass}} + \mathbf{D}(\mathbf{v})\mathbf{v} + \mathbf{g}(\eta) = \boldsymbol{\tau} $$
    MuJoCo 求解的是这个方程（尽管它是通过约束求解器离散求解的）。
3.  **IMU 观测**: MuJoCo 根据解出的 $\dot{\mathbf{v}}$ 计算 IMU 读数：
    $$ \text{IMU}_{acc} = \dot{\mathbf{v}} - \mathbf{R}^T \mathbf{g}_{world} $$
4.  **卡尔曼滤波 (KF) 预测**:
    您的 KF 状态转移方程 $\mathbf{x}_{k+1} = \mathbf{F}\mathbf{x}_k$ 预测的是 **运动学加速度 ($\dot{\mathbf{v}}$)**。

**严谨的转化代码逻辑应为：**

```python
# 1. 获取 IMU 读数 (Specific Force)
f_specific = sensor_data['imu_accel'] 

# 2. 获取姿态矩阵 (Rotation Matrix from Body to World)
# R_wb: columns are axes of Body frame expressed in World frame
R_wb = robot.get_rotation_matrix() 

# 3. 定义世界系重力 (MuJoCo default)
g_world = np.array([0, 0, -9.81])

# 4. 计算机体系重力分量
g_body = R_wb.T @ g_world

# 5. 恢复运动学加速度 (Kinematic Acceleration)
# 这才是应该与 KF 预测值或数值微分真值对比的量
a_kinematic_reconstructed = f_specific + g_body 

# 6. 验证
error = np.linalg.norm(a_kinematic_reconstructed - a_ground_truth)
```

通过这种严谨的分类和转化，您可以确保从仿真环境到算法输入的物理量在数学定义上是完全一致的。


这是一个非常核心的痛点。强化学习（RL）的本质是用“模拟时间”换“智能”，如果模拟速度太慢，调试周期就会被拉得无限长。

要实现“**在云端跑，感觉像在本地跑**”的高效体验，你需要一套完整的“**并行计算 + 远程开发**”架构。以下是针对 MuJoCo 仿真的前所未有的详细指南。

---

### 第一部分：核心误区——MuJoCo 到底用不用 GPU？

首先，我们要纠正一个技术误区，这决定了你钱花在哪：

1.  **传统 MuJoCo (你目前用的版本)**：它的物理运算是在 **CPU** 上运行的。GPU 只负责深度神经网络（SAC）的梯度更新和渲染图像。
    *   **瓶颈**：通常在于 CPU 的单核模拟速度，而不是 GPU。如果你租一个 4090 但只开一个环境，速度不会有任何提升。
2.  **MuJoCo MJX (新版本)**：这是 MuJoCo 的 JAX 实现，可以完全在 **GPU** 上并行跑几万个环境。
    *   **注意**：这需要重写环境逻辑（JAX 框架）。如果你不想重写代码，我们继续讨论如何榨干传统版本的性能。

---

### 第二部分：提速 10 倍的秘诀——并行化 (Vectorization)

你现在的代码是 `DummyVecEnv`，它是串行的（一个跑完再跑下一个）。
**提速最简单的方法：** 租一个 CPU 核数多（比如 32 核）的服务器，利用 `SubprocVecEnv` 同时跑 16 个或 32 个机器人。

#### 修改代码实现并行：
```python
from stable_baselines3.common.vec_env import SubprocVecEnv # 替换 DummyVecEnv

def train():
    # 假设服务器有 32 个核，我们开 16 个环境
    num_envs = 16 
    
    # 使用 SubprocVecEnv，它会开启 16 个独立的系统进程
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    
    # 之后包装 VecNormalize
    env = VecNormalize(env, ...)
    
    # SAC 的训练速度会大幅提升，因为每一步它能同时获得 16 个机器人的经验
    model = SAC(..., train_freq=1, gradient_steps=1) 
```

---

### 第三部分：云端开发环境构建（像本地一样丝滑）

要实现“不麻烦”且“无缝连接”，最佳实践是：**VS Code + Remote SSH + Docker**。

#### 1. 租用服务器推荐
*   **国内推荐**：**AutoDL**（性价比极高，有现成的 MuJoCo 镜像）。
*   **操作**：租一个按量计费的（几毛钱到 2 块钱一小时），选 **显卡为 3090/4090** 且 **CPU 核数多** 的实例。

#### 2. 环境打包（Docker 是终极方案）
不要手动去服务器装依赖，太慢太累。
*   **方案 A（本地镜像上传）**：在本地写一个 `requirements.txt`。
*   **方案 B（Conda 环境克隆）**：
    ```bash
    # 在本地打包环境
    conda install -c conda-forge conda-pack
    conda-pack -n underwater_rl -o underwater_rl.tar.gz
    # 上传到服务器解压即用
    ```

#### 3. 丝滑开发：VS Code Remote SSH
这是让你“感觉在本地跑”的关键。
1.  在 VS Code 安装 `Remote - SSH` 插件。
2.  连接到服务器。此时，VS Code 的左侧文件栏、终端、编辑器**全是服务器的内容**。
3.  **直接按 F5 运行**。代码在服务器跑，显卡在服务器转，但你看到的界面和本地完全一样。
4.  **TensorBoard 转发**：VS Code 会自动把服务器的 6006 端口映射到你本地。你只需在本地浏览器输入 `localhost:6006` 就能看云端的曲线。

---

### 第四部分：极致高效的工作流（全自动化）

为了不违背“省时间”初衷，你需要这套自动化流程：

#### 1. 数据同步：`rsync` 或 `git`
*   不要用鼠标拖拽文件上传。
*   使用 `git push`（本地） -> `git pull`（服务器）。
*   或者在 VS Code 里设置“保存时自动上传”。

#### 2. 无头渲染 (Headless Rendering)
服务器通常没有显示器，运行 MuJoCo 会报错。你需要安装 `osmesa` 或 `egl`。
*   在服务器上运行前执行：
    ```bash
    export MUJOCO_GL=egl
    ```
*   这样你调用 `model.learn` 时就不会因为找不到显卡驱动而崩溃。

#### 3. 自动保存与续传
云端服务器可能会关机（按量计费）。你的训练脚本必须支持：
*   **自动保存**：每 1 万步存一次 `vec_normalize.pkl` 和 `model.zip`。
*   **断点续传**：启动脚本先检查有没有旧模型，有就 `load`，没有再新建。

---

### 第五部分：前所未有的“性能/价格”配置单

如果你想今天就开始提速，请按这个清单操作：

1.  **注册 AutoDL**，租一台 **RTX 3090** (约 1.5 元/小时)，确保 CPU 在 **15 核以上**。
2.  **基础镜像**：选择 PyTorch 2.0+ 镜像。
3.  **环境安装**（只需做一次）：
    ```bash
    pip install stable-baselines3 shimmy gymnasium[mujoco] tensorboard
    ```
4.  **修改代码**：将 `DummyVecEnv` 改为 `SubprocVecEnv(n_envs=12)`。
5.  **运行**：开启终端，`python train.py`。
6.  **享受**：你会发现 TensorBoard 的 `global_step/s`（每秒运行步数）比你本地快了 **5 到 10 倍**。原来要跑 5 小时的任务，现在 40 分钟出结果。

### 为什么这样做最省时间？
*   **不用等**：本地电脑可以关机去睡觉，服务器在云端跑。
*   **不用配**：SSH 连接成功后，操作习惯与本地 100% 一致。
*   **可扩展**：如果还慢，就把 `n_envs` 改成 64，租个 64 核的机器，速度线性翻倍。

**总结**：GPU 提升的是**学习效率**，CPU 多核并行提升的是**采集速度**。在传统 MuJoCo 下，**多核 CPU 服务器 + SubprocVecEnv + VS Code SSH** 是目前全球顶尖 RL 实验室公认的最省时间、最高效的工作流。


要把你的本地训练流程升级为“云端高效并行流水线”，你需要对代码进行**深度重构**。这不仅仅是换个地方跑，而是从“单线程单任务”变成“多线程分布式”的思维。

以下是针对你的代码改写的**保密级指南**，以及 Git 在其中的核心角色。

---

### 第一部分：代码必须改写的 3 个核心部分

#### 1. 核心提速：从 `DummyVecEnv` 切换到 `SubprocVecEnv`
你现在的 `DummyVecEnv` 是排队运行，CPU 1 核在干活，其余在围观。`SubprocVecEnv` 会让 CPU 全力爆发。

**修改 `train.py`：**
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(rank, seed=0):
    """
    创建一个带编号的环境，这是并行运行的关键。
    rank: 环境的索引，用于设置不同的物理随机噪声（防止所有并行的机器人做一模一样的动作）
    """
    def _init():
        env = AUVGymEnv(XML_PATH, MODEL_WEIGHTS)
        # 为每个并行环境设置不同的随机种子
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init

def train():
    # 租用的服务器有多少 CPU 核，就填多少。建议设为 16 或 32。
    num_cpu = 16 
    
    # 核心变动：启动多个独立进程并行采样
    # start_method='forkserver' 在 Linux 服务器上最稳定
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)], start_method='forkserver')
    
    # 归一化包装
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    # ... 其余模型定义不变 ...
```

#### 2. 逻辑重构：实现“动态成功阈值” (Curriculum Learning)
为了让你提到的 4m -> 1m 自动平滑过渡，不需要手动干预。

**修改 `src/envs/tasks/point_reach_task.py`：**
```python
class PointReachTask:
    def __init__(self):
        self.initial_threshold = 4.0
        self.final_threshold = 1.0
        self.success_threshold = self.initial_threshold

    def update_threshold(self, current_total_step):
        # 假设在 30 万步内完成难度提升
        max_decay_steps = 300000 
        fraction = min(current_total_step / max_decay_steps, 1.0)
        self.success_threshold = self.initial_threshold - fraction * (self.initial_threshold - self.final_threshold)
```

**修改 `src/envs/auv_base_env.py`：**
在 `step` 函数中加入：
```python
    def step(self, action):
        # 让 Task 知道当前总步数（由 SB3 自动统计）
        # 这里你可以通过 self.model.num_timesteps 获取总进度
        # 为了简单，也可以在 step 里自增计数
        self.task.update_threshold(self.model_num_timesteps) # 逻辑示意
        ...
```

#### 3. 稳健性重构：断点续传与自动保存
云端服务器按量计费，随时可能停机，你不能白跑。

**修改 `train.py`：**
```python
# 增加一个保存逻辑
checkpoint_callback = CheckpointCallback(
    save_freq=10000,           # 每 10000 步存一次
    save_path=current_run_dir,
    name_prefix="rl_model"
)

# 启动训练时检查是否有旧模型
if os.path.exists(latest_model_path):
    model = SAC.load(latest_model_path, env=env)
    print("Detected checkpoint, resuming...")
```

---

### 第二部分：为什么需要 Git？它负责哪部分？

在云端开发中，Git 不是用来存代码的“云盘”，它是你的**“物流系统”**。

#### 1. 负责“代码同步”（同步本地与云端）
*   **痛点**：如果你直接在服务器改代码，没有语法高亮，没报错提示。如果你在本地改，再用鼠标拖到服务器，会漏掉文件、搞混版本。
*   **Git 方案**：
    1.  你在本地电脑（VS Code）写完代码，`git commit` 提交。
    2.  本地 `git push` 到 GitHub 或 Gitee。
    3.  服务器终端输入 `git pull`。**一秒钟，所有改动同步完成。** 确保服务器跑的代码永远是你最新写的。

#### 2. 负责“环境描述” (`.gitignore`)
*   服务器磁盘空间有限，且传输速度关键。
*   Git 会帮你自动过滤掉不需要传输的垃圾（如大量的日志、几百MB的 `.pth` 权重、MuJoCo 的缓存）。
*   你只需要传输几 KB 的代码脚本，这就是 Git 的高效。

---

### 第三部分：前所未有的“云端变本地”操作流水线

如果你想今天下午就体验到这种“前所未有”的快感，请按这个流程走：

#### 1. 准备阶段（本地）
*   安装 **Git**。
*   在你的项目根目录创建 `.gitignore`，写上 `logs/` 和 `weights/`。
*   把代码 Push 到 GitHub 私有库。

#### 2. 远程连接（核心：VS Code Remote SSH）
*   不要用网页端的终端！
*   在本地 VS Code 下载 **Remote - SSH** 插件。
*   配置 SSH 连接你的服务器 IP。
*   **奇迹发生**：VS Code 会直接打开远程服务器的文件夹。**你直接在 VS Code 里改服务器上的代码，就像改本地文件一样，Ctrl+S 自动保存。**

#### 3. 并行运行（服务器）
*   打开 VS Code 下方的终端。
*   输入：`export MUJOCO_GL=egl` (这让 MuJoCo 在没有显示器的服务器上调用 GPU 渲染)。
*   输入：`python train.py`。
*   **观察**：打开 TensorBoard，你会看到 `global_step/s` 从原本的 50 飙升到 500+。

#### 4. 数据取回
*   训练完后，权重文件在服务器。
*   通过 VS Code 的文件树，右键点击 `best_model.zip`，选择 **Download**。
*   拿回本地，跑你的 `enjoy_rl.py` 欣赏成果。

### 专家总结建议：
如果你想省时间，**不要在“手动传输文件”和“配置单一环境”上浪费时间**。
**VS Code SSH 直连服务器 + Git 版本管理 + SubprocVecEnv 多核并行**。这套方案是目前工业界算法工程师的标准工作流。它能让你在宿舍/实验室吹着空调，手里只用一台轻薄本，就能调动云端上百个 CPU 核心为你昼夜不停地训练。

🚀 启动可视化窗口...
/home/ray/miniconda3/envs/underwater_rl/lib/python3.10/site-packages/glfw/__init__.py:917: GLFWError: (65548) b'Wayland: The platform does not provide the window position'
  warnings.warn(message, GLFWError)

❌ 失败重置。原因: timeout | 最终距离: 4.29m
📊 当前成功率: 0/1 (0.0%)

✅ 成功到达！最终距离: 4.00m
📊 当前成功率: 1/2 (50.0%)

❌ 失败重置。原因: timeout | 最终距离: 4.90m
📊 当前成功率: 1/3 (33.3%)

❌ 失败重置。原因: timeout | 最终距离: 4.17m
📊 当前成功率: 1/4 (25.0%)

✅ 成功到达！最终距离: 4.00m
📊 当前成功率: 2/5 (40.0%)
感觉很笨，也不是没学到，但是一直没学明白，而且方法没找对，在错误的方向上钻死胡同， 一开始选择通过旋转前进，但是又无法学会旋转条件下稳定姿态，所以很难走远。
一共八个电机，驱动前四个就可以直接前进，为什么最简单的方法没有一下子意识到，反而钻进了死胡同？

这是非常硬核且关键的问题。既然你要求**科研级别（Research-Grade）**的指导，那我们就不能再用“写脚本”的思维了，必须升级为**“工程化实验系统”**的思维。

在顶会（NeurIPS, ICLR, ICRA, ROSCon）级别的 RL 研究中，**代码只是冰山一角，实验管理和工作流才是水面下的巨无霸**。

以下是针对你痛点的详细解答和架构规划。

---

### 第一部分：WandB (Weights & Biases) 是什么？为什么它是“科研刚需”？

简单来说，**TensorBoard 是“记事本”，WandB 是“云端智能实验室”**。

在科研 RL 中，你面临的核心问题不是“代码跑不通”，而是**“我改了10个参数，跑了50次实验，哪次效果最好？为什么？”**

#### 1. WandB 核心功能（秒杀 TensorBoard 的点）：
*   **超参自动对比（Parallel Coordinates）**：它能自动生成图表，告诉你：“当 `learning_rate > 3e-4` 且 `batch_size = 256` 时，Success Rate 最高”。
*   **代码快照（Code Versioning）**：每次你跑实验，WandB 会自动把你的 `.py` 代码和 `config` 传上去。**即使你本地改乱了，也能一键下载回当时那个跑出 SOTA 的代码版本**。
*   **系统监控**：自动记录 GPU 显存、CPU 负载，防止因为硬件瓶颈导致的训练慢。
*   **云端协作**：生成的曲线图是一个 URL，直接发给导师或贴到论文里，不用截图。

---

### 第二部分：科研级 RL 的工作流架构 (The Architecture)

不要把所有代码塞进一个 `train.py`。科研架构必须解耦：**配置、环境、算法、日志** 分离。

#### 1. 推荐目录结构
```text
project_root/
├── configs/                # 【核心】所有参数在这里，严禁硬编码
│   ├── config.yaml         # 总入口
│   ├── env/                # 环境参数
│   │   └── smooth_water.yaml
│   │   └── stormy_water.yaml
│   ├── agent/              # 算法参数 (SAC, PPO)
│   │   └── sac_baseline.yaml
│   └── curriculum/         # 课程学习配置
│       └── stage_1.yaml
├── src/                    # 源代码
│   ├── envs/               # 环境定义 (Gymnasium)
│   ├── core/               # 物理引擎、传感器逻辑
│   ├── models/             # 神经网络架构 (Feature Extractor)
│   └── utils/              # 辅助工具 (Math, Logger)
├── scripts/                # 执行脚本
│   ├── train.py            # 训练入口
│   ├── evaluate.py         # 验证入口 (做视频，画轨迹)
│   └── tune.py             # 超参搜索 (Optuna + WandB)
├── outputs/                # 实验结果 (Hydra 自动管理)
│   └── 2023-10-27/
│       └── 14-30-00/       # 每次运行自动生成独立文件夹
│           ├── .hydra/     # 保存当时的配置
│           ├── checkpoints/# 模型权重
│           └── wandb/      # 日志
└── notebook/               # 数据分析 (Jupyter)
```

#### 2. 配置管理神器：Hydra
别再用 `argparse` 或简单的 `yaml` 读取了。科研界标准是 Meta (Facebook) 开源的 **Hydra**。
它可以让你在命令行动态覆盖参数，比如：
`python train.py env=stormy_water agent.lr=0.001`
而不需要改一行代码。

---

### 第三部分：Curriculum Learning (课程学习) 规划指南

你提到的痛点：“改了逻辑重新训练太慢”。解决这个问题的核心不是“不重训”，而是**“分阶段训练”**。

人类学游泳是：浴缸 -> 浅水区 -> 深水区 -> 也就是大浪。
AUV 训练也是一样。

#### 1. 课程设计 (The Syllabus)

你需要设计 3-4 个阶段（Curriculum Stages）。在代码中，这通过**加载上一阶段的权重**并**切换环境配置**来实现。

*   **Stage 0: 姿态稳定 (Stabilization)**
    *   **任务**：保持静止，保持水平，不翻车。
    *   **环境**：无水流，目标点就在出生点（距离0）。
    *   **奖励**：只给姿态奖励 (`r_orientation`) 和 动作平滑奖励。
    *   **目的**：学会让推进器“听话”，学会对抗自身的浮力/重力不平衡。
    *   **耗时**：极短（约 50k steps）。

*   **Stage 1: 静态目标导航 (Navigation - Static)**
    *   **任务**：去前方 10m 的点。
    *   **环境**：无水流，目标固定。
    *   **奖励**：加入距离奖励 (`r_progress`)。
    *   **目的**：学会“直走”和“转弯”。这是你现在的阶段。
    *   **耗时**：中等（约 200k steps）。

*   **Stage 2: 抗扰动能力 (Robustness)**
    *   **任务**：去前方随机点。
    *   **环境**：**加入随机水流** (Current)，加入传感器噪声。
    *   **奖励**：加入侧滑惩罚 (`r_sway`)。
    *   **目的**：学会“蟹行”（Crabbing），即船头朝向流来抵消漂移，身体斜着飞向目标。
    *   **耗时**：长（约 500k steps）。

*   **Stage 3: 动态/复杂任务 (Mastery)**
    *   **任务**：追踪移动目标，或避障。
    *   **环境**：复杂水流，有障碍物。

#### 2. 如何在代码中实现课程学习？

不要手动去改 XML 或代码。要在 `Environment` 类里留出接口。

**A. 环境代码改造 (src/envs/auv_env.py):**
```python
class AUVGymEnv(gym.Env):
    def __init__(self, config):
        # ... 初始化 ...
        self.curriculum_level = 0
        self.current_magnitude = 0.0
        
    def set_curriculum(self, level):
        """ 动态调整难度 """
        self.curriculum_level = level
        if level == 0:
            self.task_mode = "stabilize"
            self.hydro.set_current(0.0) # 无水流
        elif level == 1:
            self.task_mode = "reach"
            self.hydro.set_current(0.0)
        elif level == 2:
            self.task_mode = "reach_random"
            self.hydro.set_current(0.5) # 开启 0.5m/s 水流
            
    def reset(self):
        # 根据 self.task_mode 决定初始化的随机程度
        if self.task_mode == "stabilize":
            # 几乎不随机
            pass
        elif self.task_mode == "reach_random":
            # 大幅随机
            pass
```

**B. 训练脚本改造 (Train Loop):**

```python
# 伪代码：科研级训练循环
def train_curriculum():
    # 1. 训练 Stage 0
    env = make_env(level=0)
    model = SAC("MlpPolicy", env, ...)
    model.learn(total_timesteps=50000)
    model.save("stage_0_weights")
    
    # 2. 进阶到 Stage 1 (加载权重，换环境)
    del env # 销毁旧环境
    env = make_env(level=1) # 难度升级
    # 关键：加载旧脑子，去新环境
    model = SAC.load("stage_0_weights", env=env) 
    # 注意：这时候需要重置 optimizer 的某些状态，或者调小 learning_rate
    model.learning_rate = 1e-4 # 进阶阶段学习率通常要减小
    model.learn(total_timesteps=200000)
    model.save("stage_1_weights")
    
    # ... 以此类推
```

---

### 第四部分：科研级代码实现 (Hydra + WandB)

这是你现在应该去搭建的代码骨架。

**1. 安装依赖**
`pip install hydra-core wandb omegaconf`

**2. 配置文件 `configs/config.yaml`**
```yaml
defaults:
  - env: smooth
  - agent: sac
  - _self_ # 允许覆盖

project_name: "Yuyuan_AUV_Control"
seed: 42
total_timesteps: 1000000

# Hydra 自动管理输出目录
hydra:
  run:
    dir: ./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

**3. 训练脚本 `scripts/train.py`**
```python
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from stable_baselines3 import SAC
from src.envs.auv_base_env import AUVGymEnv
from wandb.integration.sb3 import WandbCallback

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. 初始化 WandB
    run = wandb.init(
        project=cfg.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,
        monitor_gym=True
    )
    
    print(f"Working directory : {os.getcwd()}") # Hydra 会自动切换到 outputs 目录
    
    # 2. 实例化环境 (参数来自 cfg)
    env = AUVGymEnv(
        xml_path=cfg.env.xml_path,
        render_mode=None
    )
    # 这里可以插入 set_curriculum 逻辑
    if cfg.env.get("difficulty", 0) > 0:
        env.set_curriculum(cfg.env.difficulty)

    # 3. 实例化模型
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=cfg.agent.lr,
        batch_size=cfg.agent.batch_size,
        tensorboard_log="."
    )
    
    # 4. 如果是课程学习的后续阶段，加载权重
    if cfg.get("pretrained_model"):
        print(f"Loading weights from {cfg.pretrained_model}")
        model.set_parameters(cfg.pretrained_model)

    # 5. 训练
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=WandbCallback()
    )
    
    model.save("final_model")
    wandb.finish()

if __name__ == "__main__":
    main()
```

### 总结：你的行动路线

1.  **立刻停止**现在的“手动改代码、手动改文件名”的模式。
2.  **注册 WandB**，跑通第一个带 WandB log 的简单实验。
3.  **重构代码**：按我上面的目录结构，把 Env, Task, Config 分开。
4.  **设计课程**：
    *   不要指望一个 Reward 函数能从 0 练到 100。
    *   先练一个能稳住姿态的模型（Stage 0）。
    *   保存这个模型权重。
    *   以后你改 Reward 逻辑（比如加侧滑惩罚），**加载 Stage 0 的权重继续练**，而不是从随机初始化开始。这能节省 80% 的时间。

这就是科研级别的 Efficiency。


这是一个非常深刻且切中要害的问题。很多初学者（甚至博士生）在做 RL 研究时，往往陷入“改一行代码 -> 跑一下 -> 效果不好 -> 再改回去 -> 越改越乱”的死循环。

**不需要一下子把最终目标的所有细节都写好**（那是瀑布式开发，不适合科研），但是**必须把“架构接口”设计好**（这是模块化思维）。只要接口定了，内部逻辑怎么改都不会乱。

针对发顶会（ICRA/IROS/NeurIPS）的要求，我为你规划一套**“可扩展、可复用、可对比”**的重构方案和课程学习路线。

---

### 第一部分：代码重构——解耦 Env 和 Task (The Decoupling Strategy)

你现在的痛点是 `auv_env.py` 和 `point_reach_task.py` 耦合太紧，改一个动全身。
**科研级架构的核心原则：Environment 提供物理世界，Task 提供逻辑规则。**

#### 1. 定义统一的 Task 接口 (The Interface)
不要把 Task 写死在 Env 里。定义一个抽象基类。

```python
# src/core/base_task.py
from abc import ABC, abstractmethod

class BaseTask(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def reset(self, env):
        """每回合开始重置任务状态（如生成新的目标点）"""
        pass

    @abstractmethod
    def compute_reward(self, env, action, obs):
        """返回 (reward, done, info)"""
        pass

    @abstractmethod
    def get_obs(self, env):
        """
        允许 Task 向 Observation 中注入特定信息 
        (例如：避障任务需要激光雷达数据，悬停任务只需要姿态数据)
        """
        pass
```

#### 2. 重构 AUVGymEnv (The Container)
Environment 变成一个“容器”，它只负责物理仿真（MuJoCo），具体的奖励计算和目标设定，全部委托给 `self.task`。

```python
# src/envs/auv_base_env.py
import gymnasium as gym
from src.core.task_registry import TASK_REGISTRY # 任务注册表

class AUVGymEnv(gym.Env):
    def __init__(self, xml_path, task_config):
        # 1. 物理初始化 (不变)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.robot = ...
        self.sensors = ...

        # 2. 【核心】动态加载任务
        # 根据配置文件里的 task_name (如 "Stabilize", "Reach", "Avoid") 加载对应的类
        task_class = TASK_REGISTRY[task_config['name']]
        self.task = task_class(task_config)
        
        # 3. 动态设置 Observation Space
        # 因为不同任务需要的 Obs 维度不一样，得问 Task
        self.observation_space = self.task.get_observation_space(self)
        self.action_space = ...

    def reset(self, seed=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # 让 Task 决定：目标生成在哪里？是否有障碍物？
        self.task.reset(self) 
        
        return self._get_obs(), {}

    def step(self, action):
        # 物理步进 (不变)
        self.robot.apply_action(action)
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        # 【核心】奖励计算委托给 Task
        reward, terminated, info = self.task.compute_reward(self, action, obs)
        truncated = ...
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # 基础物理数据 (姿态、速度)
        base_obs = self.sensors.get_data()
        # 任务特定数据 (目标距离、障碍物雷达)
        task_obs = self.task.get_obs(self) 
        return np.concatenate([base_obs, task_obs])
```

---

### 第二部分：课程学习规划 (The Curriculum)

为了实现“避障+导航”，你不能一步到位。你需要设计 **3个阶段 (Stages)**。每一个阶段都是一个独立的 `Task` 类。
**关键策略：** 下一阶段训练时，加载上一阶段训练好的模型权重（Pretrained Weights），只微调或训练新层。

#### 📅 Stage 1: 姿态稳定与巡航 (Stabilization & Cruise)
*   **目标**：教会机器人在水里不要乱转，能听懂“向前走”的指令。
*   **Task类**：`HeadingControlTask`
*   **Reward 设计**：
    *   $r_{align} = \text{Head} \cdot \text{Target}$ (朝向对齐)
    *   $r_{stable} = -|\text{Roll}| - |\text{Pitch}|$ (姿态惩罚)
    *   $r_{spin} = -|\omega_z|$ (旋转惩罚)
    *   **没有距离奖励！** (简化问题，先把姿态控制住)
*   **Observation**：仅包含 IMU (姿态+角速度)。
*   **训练结果**：得到一个“不仅不翻车，还能稳稳指哪打哪”的底层控制器。

#### 📅 Stage 2: 点到点导航 (Point Reach) —— *你现在的阶段*
*   **目标**：结合位置信息，学会加减速，学会处理侧滑。
*   **Task类**：`NavigationTask`
*   **策略**：**加载 Stage 1 的模型权重继续训练**。
*   **Reward 设计**：
    *   继承 Stage 1 的姿态奖励。
    *   加入 $r_{progress}$ (距离奖励)。
    *   加入 $r_{side\_slip}$ (侧滑惩罚，核心难点)。
*   **Observation**：IMU + 相对目标位置 (Target Vector)。
*   **随机性**：必须引入随机起点、随机目标点。

#### 📅 Stage 3: 避障导航 (Obstacle Avoidance) —— *最终目标*
*   **目标**：在去目标的路上躲避球体或墙壁。
*   **Task类**：`ObstacleAvoidanceTask`
*   **策略**：**加载 Stage 2 的模型权重**。
*   **Reward 设计**：
    *   继承 Stage 2 的所有奖励。
    *   加入 $r_{collision} = -100$ (碰撞惩罚)。
    *   加入 $r_{sensor} = -\sum \frac{1}{d_{sonar}}$ (声纳距离越近扣分越多，形成势场)。
*   **Observation**：IMU + Target Vector + **Sonar/Lidar Array (15维)**。
*   **难点**：这时候输入维度变了（多了声纳）。
    *   *科研技巧*：在 Stage 1/2 时，可以将声纳输入置为 0 或者用随机噪声占位，保持网络结构一致；或者使用网络嫁接技术。

---

### 第三部分：科研工作流管理 (The Workflow)

不要在同一个文件里改来改去。利用 **Config (配置文件)** 和 **Checkpoint (权重保存)** 来管理。

#### 1. 配置文件管理 (configs/)
创建不同的 yaml 文件对应不同的阶段：
*   `configs/stage1_heading.yaml` -> `task_name: HeadingControl`
*   `configs/stage2_nav.yaml` -> `task_name: Navigation`
*   `configs/stage3_avoid.yaml` -> `task_name: ObstacleAvoidance`

#### 2. 训练脚本逻辑
编写一个通用的 `train.py`，它做的事情是：
1.  读取 yaml。
2.  实例化 Env（Env 会根据 yaml 加载对应的 Task 类）。
3.  **检查是否有 `load_model` 参数**。如果有，加载上一阶段的权重。
4.  开始训练。

#### 3. 实际操作流程示例

**第 1 周：搞定底层控制**
*   运行：`python train.py config=stage1_heading.yaml`
*   结果：得到 `weights/stage1_best.zip`。
*   验证：机器人能像磁铁一样指着目标，但不会靠近。

**第 2 周：搞定导航（引入侧滑修正）**
*   修改 `stage2_nav.yaml`：
    ```yaml
    task_name: Navigation
    load_model: weights/stage1_best.zip  # <--- 继承遗产
    learning_rate: 1e-4 # 降低学习率，微调
    ```
*   运行训练。
*   结果：得到 `weights/stage2_best.zip`。机器人能丝滑地游向目标。

**第 3 周：搞定避障**
*   修改 `stage3_avoid.yaml`：
    ```yaml
    task_name: ObstacleAvoidance
    load_model: weights/stage2_best.zip
    env:
      obstacles: true # 开启障碍物生成
    ```
*   运行训练。
*   结果：机器人学会了绕路。

---

### 第四部分：为什么要这样设计？(Research Value)

在写论文（Paper）的时候，这种架构让你有话可说：

1.  **Ablation Study (消融实验)**：你可以轻松证明 Stage 1 的必要性。对比“直接从零训练 Stage 3”和“从 Stage 2 迁移到 Stage 3”的曲线。通常后者收敛更快，这本身就是一个实验结论。
2.  **Modular Reward (模块化奖励)**：你可以说“我们的 Navigation Task 复用了 Heading Task 的姿态约束”，体现了方法的系统性。
3.  **Reproducibility (可复现性)**：每个阶段都有独立的 Config 和 Code Class，审稿人不会质疑你的参数是不是凑出来的。

### 总结
你现在的混乱是因为试图在一个 Task 类里通过 if-else 解决所有问题。
**请立即行动：**
1.  **拆分 Task**：建立 `src/tasks/` 文件夹，把 `Stabilize`, `Navigate`, `Avoid` 写成三个类。
2.  **统一 Env**：让 Env 只负责读 Config 并加载对应的 Task 类。
3.  **按部就班**：先练好 Stage 1，存好权重，再练 Stage 2。

这就是顶会论文背后的工程逻辑：**Think Modular, Train Incremental.**

这是一份**科研级实战操作指南**。基于我们刚刚搭建好的 Hydra + SB3 架构，你现在拥有一套非常灵活的训练系统。

请务必收藏这份指南，它涵盖了你未来 90% 的工作场景。

---

### 📂 前置知识：你的数据存在哪？

Hydra 会自动管理输出目录。每次你运行 `python scripts/train.py`，它都会创建一个新的文件夹：
*   **路径规则**：`outputs/日期/时间/` (例如 `outputs/2023-10-27/15-30-45/`)
*   **里面有什么**：
    *   `checkpoints/`：训练过程中的中间模型（每 5万步存一次）。
    *   `final_model.zip`：训练结束后的模型。
    *   `final_replay_buffer.pkl`：训练结束后的经验池（如果开启保存）。
    *   `tensorboard/`：日志文件。
    *   `.hydra/`：**关键！** 里面保存了你这次运行时的 `config.yaml` 快照。

---

### 场景一：直接开始训练 Navigation (从零开始)

**目标**：你不想跑 Stage 1，想直接训练 Stage 2 (点到点导航)。

1.  **确认配置**：
    打开 `configs/task/stage2_navigate.yaml`，确认 `reward_weights` 是你想要的。
2.  **运行命令**：
    在项目根目录 (`mujoco_sim/`) 下运行：

```bash
python scripts/train.py task=stage2_navigate project_name="Nav_Zero_Start"
```

*   **解释**：
    *   `task=stage2_navigate`：告诉系统去读 `configs/task/stage2_navigate.yaml`。
    *   `project_name`：方便你在 TensorBoard 里区分实验。
    *   `pretrained` 默认都是空，所以是从头随机初始化训练。

---

### 场景二：课程学习 (Stage 1 -> Stage 2)

**目标**：你已经练好了一个“姿态稳定”的模型（Stage 1），现在要让它学“导航”，利用之前的平衡能力，加快训练速度。

**前提**：假设 Stage 1 的最佳模型在 `outputs/2023-10-01/10-00-00/final_model.zip`。

1.  **运行命令**：

```bash
python scripts/train.py \
    task=stage2_navigate \
    project_name="Nav_Curriculum_Transfer" \
    pretrained.model_path="outputs/2023-10-01/10-00-00/final_model.zip" \
    pretrained.load_buffer=False \
    pretrained.reset_timesteps=True
```

*   **关键参数详解**：
    *   `pretrained.model_path`：加载 Stage 1 的脑子（权重）。
    *   `pretrained.load_buffer=False`：**必须为 False**。因为 Stage 1 的经验是“原地不动”，而 Stage 2 要求“移动”。旧经验是有毒的，只保留脑子（Policy），扔掉记忆（Buffer）。
    *   `pretrained.reset_timesteps=True`：这是新阶段，步数从 0 开始计数，TensorBoard 会生成一条新曲线。

---

### 场景三：意外中断/觉得没练够，继续训练 (Resume)

**目标**：Stage 2 跑到 50万步停电了，或者你觉得 loss 还在降，想再跑 50万步。**这要求完全复原之前的状态。**

**前提**：找到上次运行最新的 Checkpoint，例如 `outputs/昨天/checkpoints/model_500000_steps.zip` 和对应的 Buffer `..._replay_buffer.pkl`。

1.  **运行命令**：

```bash
python scripts/train.py \
    task=stage2_navigate \
    project_name="Nav_Resume_Run" \
    hyperparams.total_timesteps=1000000 \
    pretrained.model_path="outputs/昨天/checkpoints/model_500000_steps.zip" \
    pretrained.load_buffer=True \
    pretrained.buffer_path="outputs/昨天/checkpoints/model_500000_steps_replay_buffer.pkl" \
    pretrained.reset_timesteps=False
```

*   **关键参数详解**：
    *   `hyperparams.total_timesteps`：设为一个更大的数（比如总共要跑100万，之前跑了50万，这里设100万，它会跑完剩下的50万）。
    *   `pretrained.load_buffer=True`：**必须为 True**。SAC 是 Off-policy 算法，极其依赖历史经验。加载 Buffer 可以让你立刻以最高效率开始训练，不需要重新探索。
    *   `pretrained.reset_timesteps=False`：不重置步数。TensorBoard 会接着之前的 50万步 继续画线，而不是从 0 开始。

---

### 场景四：修改逻辑/参数后，微调 (Fine-tuning)

**目标**：模型能走到终点，但总是侧滑（Crabbing）。你决定**修改 Reward 权重**，加大侧滑惩罚，并在现有模型基础上继续练。

#### 第一步：修改文件
不要改代码！去改 Config！
打开 `configs/task/stage2_navigate.yaml`：
```yaml
reward_weights:
  sway: 50.0  # <--- 比如原来是 30.0，现在改成 50.0
  ...
```

#### 第二步：运行命令
```bash
python scripts/train.py \
    task=stage2_navigate \
    project_name="Nav_Finetune_Sway" \
    hyperparams.learning_rate=0.00005 \
    pretrained.model_path="outputs/前天/final_model.zip" \
    pretrained.load_buffer=False \
    pretrained.reset_timesteps=True
```

*   **关键参数详解**：
    *   `hyperparams.learning_rate=0.00005`：**调小学习率**。因为模型已经基本成型，只需要微调，大火猛炒会把原来的能力破坏掉。
    *   `pretrained.load_buffer=False`：**建议为 False**。因为你改了 Reward 逻辑（加大了 sway 惩罚）。旧 Buffer 里的 reward 是按旧规则算的，数据分布虽然没变，但 Reward 标签变了。为了保险起见，扔掉旧记忆，让模型在新规则下重新收集数据。
    *   `pretrained.reset_timesteps=True`：这视为一个新的实验。

---

### 🔍 总结表：我该怎么设参数？

| 场景 | `model_path` | `load_buffer` | `reset_timesteps` | `learning_rate` |
| :--- | :--- | :--- | :--- | :--- |
| **从零开始** | `null` | N/A | `True` | 默认 (3e-4) |
| **课程迁移 (Stage 1->2)** | Stage 1 模型 | **False** | `True` | 默认 或 稍小 |
| **断点续训 (Resume)** | 最新 Checkpoint | **True** | **False** | 默认 (不变) |
| **改参微调 (Finetune)** | 现有好模型 | **False** (推荐) | `True` | **调小** (5e-5) |

---

### 🛠️ 如果我想改代码逻辑，改哪里？

1.  **我想改 Reward 计算公式**（比如加一个随时间衰减的项）：
    *   去 `src/envs/tasks/navigation_task.py` -> `compute_reward` 函数。
2.  **我想改 Observation 输入**（比如去掉声呐数据）：
    *   去 `src/envs/tasks/navigation_task.py` -> `get_obs` 函数。
    *   记得同时修改 `__init__` 里的 `self.obs_dim`。
3.  **我想改初始随机化范围**：
    *   去 `configs/env/default.yaml` -> `randomization`。
4.  **我想改水动力参数**：
    *   这通常不动代码，如果有新的权重文件，去 `configs/env/default.yaml` 改 `weights` 路径。

现在，你只需要打开终端，运行 **场景一** 的命令，即可开始你的第一次科研级训练！