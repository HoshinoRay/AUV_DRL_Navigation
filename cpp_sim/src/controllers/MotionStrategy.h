#ifndef MOTION_STRATEGY_H
#define MOTION_STRATEGY_H

#include "../utils/DataCollector.h" 
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <map>

// 6自由度归一化力/力矩 (-1.0 到 1.0)
struct ControlOutput {
    double surge = 0.0;double sway = 0.0;double heave = 0.0;double roll = 0.0;double pitch = 0.0;double yaw = 0.0;
};

class MotionStrategy {
public:
    virtual ~MotionStrategy() {}
    
    // 核心函数：输入当前状态和时间步长，输出6自由度控制指令?
    virtual ControlOutput calculate(const AUVState& state, double dt) = 0;
    
    // 可选：重置内部状态（如积分项）?
    virtual void reset() {}
};

class IdleStrategy : public MotionStrategy {
public:
    ControlOutput calculate(const AUVState& state, double dt) override {
        return ControlOutput(); // 全0
    }
};

// 3. PID 定点/定深策略 
class PIDStationKeepingStrategy : public MotionStrategy {
private:
    // PID parameter
    double Kp_roll = 2.0, Kd_roll = 0.5;
    double Kp_pitch = 3.0, Kd_pitch = 0.8;
    double Kp_depth = 1.5, Kd_depth = 0.5;
    double Kp_yaw = 1.0, Kd_yaw = 0.3;
    double Kp_surge = 1.0, Kd_surge = 0.0;
    double Kp_sway = 1.0, Kd_sway = 0.2;

    // target
    double targetDepth = 15.0;
    double targetYaw = 0.0; // rad
    double targetRoll = 0.0;
    double targetPitch = 0.0;
    double targetSurge = 0.5; // m/s

    // 辅助函数：角度归一化
    double normalizeAngle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

    // 辅助函数：限幅
    double clamp(double val, double limit) {
        if (val > limit) return limit;
        if (val < -limit) return -limit;
        return val;
    }

public:
    // 可以添加构造函数来动态设置目标
    PIDStationKeepingStrategy(double depth, double surgeVel) 
        : targetDepth(depth), targetSurge(surgeVel) {}

    ControlOutput calculate(const AUVState& state, double dt) override {
        ControlOutput cmd;

        // --- 1. 姿态稳定 ---
        double err_roll = normalizeAngle(targetRoll - state.roll);
        cmd.roll = Kp_roll * err_roll - Kd_roll * state.p;

        double err_pitch = normalizeAngle(targetPitch - state.pitch);
        cmd.pitch = Kp_pitch * err_pitch - Kd_pitch * state.q;

        // --- 2. 深度控制 ---
        double err_depth = targetDepth - state.z;
        // 防翻船保护
        double depth_gain_scale = std::max(0.0, 1.0 - std::abs(state.pitch) / (M_PI/4.0));
        cmd.heave = (Kp_depth * err_depth - Kd_depth * state.w) * depth_gain_scale;

        // --- 3. 航向控制 ---
        double err_yaw = normalizeAngle(targetYaw - state.yaw);
        cmd.yaw = Kp_yaw * err_yaw - Kd_yaw * state.r;

        // --- 4. 平动控制 ---
        double err_surge = targetSurge - state.u;
        cmd.surge = Kp_surge * err_surge - Kd_surge * 0.0;

        double err_sway = 0.0 - state.v;
        cmd.sway = Kp_sway * err_sway - Kd_sway * 0.0;

        // --- 5. 安全限幅 ---
        cmd.surge = clamp(cmd.surge, 0.6);
        cmd.sway  = clamp(cmd.sway,  0.5);
        cmd.heave = clamp(cmd.heave, 0.8);
        cmd.roll  = clamp(cmd.roll,  0.5);
        cmd.pitch = clamp(cmd.pitch, 1.0);
        cmd.yaw   = clamp(cmd.yaw,   0.5);

        return cmd;
    }
};

// 阶梯式全速测试策略：用于系统辨识和数据采集
class MaxThrustStrategy : public MotionStrategy {
private:
    // --- 状态机变量 ---
    double timer = 0.0;           // 当前阶梯的计时器
    int stepIndex = 0;            // 当前执行到第几步
    double currentSurgeCmd = 0.0; // 当前实际发送的指令（用于平滑斜坡处理）

    // --- 配置参数 ---
    const double STEP_DURATION = 8.0; // 每个速度保持 8 秒（给足时间让水流稳定）
    const double RAMP_RATE = 0.2;     // 加速率：每秒增加 0.2 (即从0到1.0需要5秒，很柔和)
    const double TARGET_DEPTH = 15.0; // 锁定深度

    // --- 速度阶梯定义 ---
    std::vector<double> SPEED_STEPS = {
        0.0, 0.25, 0.50, 0.75, 1.00,  // 正向加速阶段
        0.75, 0.50, 0.25, 0.00,       // 正向减速阶段
        -0.25, -0.50, -0.75, -1.00,   // 反向加速阶段
        -0.75, -0.50, -0.25, 0.00     // 反向减速阶段
    };

    // --- PID 参数 (定深 + 姿态维持) ---
    struct PIDParams {
        double kp_z = 2.0, ki_z = 0.05, kd_z = 1.5; // 深度
        double kp_p = 3.0, kd_p = 1.0;              // 俯仰 (Pitch)
        double kp_r = 2.0, kd_r = 0.5;              // 横滚 (Roll)
        double integral_z = 0.0;                    // 深度积分
    } pid;

public:
    void reset() override {
        timer = 0.0;
        stepIndex = 0;
        currentSurgeCmd = 0.0;
        pid.integral_z = 0.0;
        std::cout << "[MaxThrust] Reset. Starting Step-Velocity Data Collection." << std::endl;
    }

    ControlOutput calculate(const AUVState& state, double dt) override {
        ControlOutput cmd;
        timer += dt;

        // ================= 1. 速度阶梯逻辑 =================
        // 检查是否需要切换到下一个速度阶梯
        if (timer >= STEP_DURATION) {
            timer = 0.0;
            stepIndex++;
            if (stepIndex >= (int)SPEED_STEPS.size()) {
                stepIndex = 0; // 循环
            }
            std::cout << "[DataCollection] Step " << stepIndex 
                      << ": Target Surge = " << SPEED_STEPS[stepIndex] << std::endl;
        }

        // 获取当前目标速度
        double targetSurge = SPEED_STEPS[stepIndex];

        // --- 斜坡平滑处理 (Ramp Generator) ---
        // 即使目标是从 0.5 跳到 0.75，我们也要慢慢加上去
        double diff = targetSurge - currentSurgeCmd;
        double max_change = RAMP_RATE * dt;

        // 如果差距很小，直接到位；否则按最大斜率逼近
        if (std::abs(diff) <= max_change) {
            currentSurgeCmd = targetSurge; 
        } else {
            currentSurgeCmd += (diff > 0 ? max_change : -max_change);
        }

        cmd.surge = currentSurgeCmd;

        // ================= 2. 深度控制 (PID) =================
        double err_depth = TARGET_DEPTH - state.z;

        // 积分抗饱和
        if (std::abs(err_depth) < 1.0) { 
            pid.integral_z += err_depth * dt;
            // 积分限幅
            if (pid.integral_z > 2.0) pid.integral_z = 2.0;
            if (pid.integral_z < -2.0) pid.integral_z = -2.0;
        } else {
            pid.integral_z = 0.0; // 误差大时清空积分
        }

        // 深度输出 = P项 + I项 - D项
        // --- 修正点在此 ---
        cmd.heave = pid.kp_z * err_depth + pid.ki_z * pid.integral_z - pid.kd_z * state.w;


        // ================= 3. 姿态稳定 (Active Stabilization) =================
        // Pitch Control (防止抬头/低头)
        double err_pitch = 0.0 - state.pitch;
        cmd.pitch = pid.kp_p * err_pitch - pid.kd_p * state.q;

        // Roll Control (防止侧翻)
        double err_roll = 0.0 - state.roll;
        cmd.roll = pid.kp_r * err_roll - pid.kd_r * state.p;

        // ================= 4. 辅助修正 =================
        cmd.sway = 0.0; 
        cmd.yaw  = 0.0; 

        // 限幅保护
        cmd.heave = std::max(-1.0, std::min(1.0, cmd.heave));
        cmd.pitch = std::max(-1.0, std::min(1.0, cmd.pitch));
        cmd.roll  = std::max(-1.0, std::min(1.0, cmd.roll));
        // Surge 不需要在这里限幅，因为 SPEED_STEPS 里的值已经是归一化的

        return cmd;
    }
};

// 4. Zero-G 优化的数据采集策略
class DataCollectionStrategy : public MotionStrategy {
private:
    double timer = 0.0;
    int currentPhase = 0;
    
    // 安全深度和目标
    const double SAFE_DEPTH = 15.0; 
    const double STABILIZE_TIME = 10.0; // 每次动作后，花10秒钟重新稳定
    const double PHASE_DURATION = 20.0; // 动作持续时间

    // 内部简易 PID 参数 (针对无重力环境微调)
    // 因为没有重力，Heave不需要很大就能拉回来
    struct PIDConfig {
        double kp_z = 2.0, kd_z = 1.0;
        double kp_w = 0.0, kd_w = 0.0; // w的控制包含在z的微分里，这里简化
        double kp_att = 1.5, kd_att = 0.5; // 姿态
        double kp_vel = 2.0; // 速度阻尼
    } pid;

    // 三角波生成器 (保持不变)
    double getTriangleWave(double t, double period) {
        double u = t / period;
        if (u > 1.0) u = 1.0; 
        if (u < 0.25) return u * 4.0;
        else if (u < 0.75) return 1.0 - (u - 0.25) * 4.0;
        else return -1.0 + (u - 0.75) * 4.0;
    }

    // --- 核心：通用 PID 稳定函数 ---
    // 用于在动作间隙把机器人拉回 15m，并摆正姿态
    ControlOutput runStabilize(const AUVState& state, double targetYawRate = 0.0) {
        ControlOutput cmd;

        // 1. 深度锁定 (最重要，防止飞天)
        double err_depth = SAFE_DEPTH - state.z;
        cmd.heave = pid.kp_z * err_depth - pid.kd_z * state.w;

        // 2. 姿态摆正 (Roll/Pitch 归零)
        cmd.roll  = 0.0 - pid.kp_att * state.roll  - pid.kd_att * state.p;
        cmd.pitch = 0.0 - pid.kp_att * state.pitch - pid.kd_att * state.q;

        // 3. 水平速度阻尼 (刹车)
        cmd.surge = -pid.kp_vel * state.u; 
        cmd.sway  = -pid.kp_vel * state.v;

        // 4. 航向控制 (要么锁死，要么缓慢旋转)
        // 如果 targetYawRate 为 0，则是阻尼模式（锁死航向）
        if (std::abs(targetYawRate) < 0.001) {
            cmd.yaw = -1.0 * state.r; // 纯阻尼，不回正北方，只求不转
        } else {
            // 开环给一点力让它慢转，或者闭环控制转速
            // 这里简单给一个反向阻尼 + 目标转速前馈
            cmd.yaw = -0.5 * (state.r - targetYawRate);
        }

        // 限幅保护
        cmd.heave = std::max(-1.0, std::min(1.0, cmd.heave));
        cmd.surge = std::max(-0.5, std::min(0.5, cmd.surge));
        cmd.sway  = std::max(-0.5, std::min(0.5, cmd.sway));
        cmd.roll  = std::max(-0.5, std::min(0.5, cmd.roll));
        cmd.pitch = std::max(-0.5, std::min(0.5, cmd.pitch));

        return cmd;
    }

public:
    void reset() override {
        timer = 0.0;
        currentPhase = 0;
        std::cout << "[DataCollection] Strategy Reset. Starting Safe Sequence." << std::endl;
    }

    ControlOutput calculate(const AUVState& state, double dt) override {
        ControlOutput cmd; // 默认为0
        timer += dt;

        // 紧急安全网：如果在任何测试阶段（除了Stabilize），深度小于 2米 (快飞出水面了)
        // 强制接管，立刻下潜
        bool safetyTrigger = (currentPhase % 2 == 0) && (state.z < 2.0); 
        if (safetyTrigger) {
            std::cout << "[WARNING] Near Surface! Emergency Dive!" << std::endl;
            cmd = runStabilize(state);
            cmd.heave = -1.0; // 强制最大力下潜
            return cmd;
        }

        // 偶数阶段：执行测试动作 (0, 2, 4...)
        // 奇数阶段：执行 PID 归位 (1, 3, 5...)
        
        switch (currentPhase) {
            // ================= PHASE 0: Surge Test =================
            case 0: 
                cmd.surge = getTriangleWave(timer, PHASE_DURATION);
                // 此时没有重力，cmd.heave = 0 理论上应该不动，但为了保险，可以微弱开启深度保持
                // cmd.heave = (SAFE_DEPTH - state.z) * 0.5; 
                if (timer >= PHASE_DURATION) nextPhase();
                break;

            // ================= PHASE 1: Stabilize (归位) =================
            case 1:
                cmd = runStabilize(state);
                if (timer >= STABILIZE_TIME) nextPhase(); // 10秒后进入下一项
                break;

            // ================= PHASE 2: Heave Test =================
            case 2:
                cmd.heave = getTriangleWave(timer, PHASE_DURATION);
                // 只有这里最危险，容易飞天。
                // 三角波前半段是向上的力(0->1)，如果此时在5m深度，可能会冲出去。
                // 所以我们在上面加了 safetyTrigger 保护。
                if (timer >= PHASE_DURATION) nextPhase();
                break;

            // ================= PHASE 3: Stabilize (归位) =================
            case 3:
                cmd = runStabilize(state);
                if (timer >= STABILIZE_TIME) nextPhase();
                break;

            // ================= PHASE 4: Yaw Test =================
            case 4:
                cmd.yaw = getTriangleWave(timer, PHASE_DURATION);
                // 锁住深度
                cmd.heave = (SAFE_DEPTH - state.z) * 1.0 - 0.5 * state.w;
                if (timer >= PHASE_DURATION) nextPhase();
                break;

            // ================= PHASE 5: Stabilize (归位) =================
            case 5:
                cmd = runStabilize(state);
                if (timer >= STABILIZE_TIME) nextPhase();
                break;
            
            // ================= PHASE 6: Roll Test =================
            case 6:
                cmd.roll = getTriangleWave(timer, PHASE_DURATION);
                // 锁住深度
                cmd.heave = (SAFE_DEPTH - state.z) * 1.0 - 0.5 * state.w;
                if (timer >= PHASE_DURATION) nextPhase();
                break;

            // ================= PHASE 7: Stabilize (归位) =================
            case 7:
                cmd = runStabilize(state);
                if (timer >= STABILIZE_TIME) nextPhase();
                break;

            // ================= DEFAULT: 结束待机 =================
            default:
                // 这里按你的要求：定深，原地不动 或 极慢速旋转
                // runStabilize 默认会把速度杀到0，且拉回 15m
                
                // 方式A: 完全不动 (Station Keeping)
                cmd = runStabilize(state, 0.0); 

                // 方式B: 如果你想要“很慢的原地打转”(比如为了展示它还活着)
                // cmd = runStabilize(state, 0.1); // 目标 YawRate = 0.1 rad/s
                
                break;
        }

        return cmd;
    }

private:
    void nextPhase() {
        currentPhase++;
        timer = 0.0;
        std::cout << "[DataCollection] Phase Transition -> " << currentPhase << std::endl;
    }
};

// ==========================================
// 辅助工具：OU 噪声生成器 (用于随机阶段)
// ==========================================
class OUNoise {
private:
    double val, theta, sigma, mean, dt;
    std::mt19937 gen;
    std::normal_distribution<double> dist;
public:
    OUNoise(double _theta, double _sigma, double _mean, double _dt) 
        : val(_mean), theta(_theta), sigma(_sigma), mean(_mean), dt(_dt), dist(0.0, 1.0) {
        std::random_device rd;
        gen.seed(rd());
    }
    double sample() {
        double noise = dist(gen);
        val += theta * (mean - val) * dt + sigma * std::sqrt(dt) * noise;
        return std::max(-1.0, std::min(1.0, val));
    }
    void reset() { val = mean; }
};

// ==========================================
// 核心策略：混合数据采集 (Hybrid Data Collection)
// ==========================================
class HybridDataCollectionStrategy : public MotionStrategy {
public:
    // --- 阶段定义 ---
    enum Phase {
        INIT_STABILIZE = 0, // 初始稳定
        TEST_SURGE,         // 1. 前后平滑运动
        //TEST_SWAY,          // 2. 左右平滑运动
        TEST_HEAVE,         // 3. 升降平滑运动
        TEST_ROLL,          // 4. 横滚测试
        TEST_PITCH,         // 5. 俯仰测试
        TEST_YAW,           // 6. 偏航测试
        TEST_CIRCLE_CW,     // 7. 顺时针盘旋
        TEST_CIRCLE_CCW,    // 8. 逆时针盘旋

        // === 新增耦合测试阶段 ===
        TEST_WAVE_VERTICAL,  // 1. 垂直波浪 (Surge + Heave): 模拟起伏地形跟踪
        TEST_CORKSCREW,      // 2. 螺旋升降 (Heave + Yaw): 模拟环绕目标上升/下降
        TEST_DRIFT_TURN,     // 3. 漂移转弯 (Surge + Sway + Yaw): 模拟侧滑过弯
        // ======================

        STOCHASTIC_EXPLORE, // 9. 随机全覆盖 (最终阶段)
        _PHASE_COUNT
    };

private:
    // --- 时间配置 (秒) ---
    const double T_STABILIZE = 5.0;   // 每次切换动作间的稳定时间
    const double T_AXIS_TEST = 180.0;   // 单轴测试持续时间
    const double T_CIRCLE    = 180.0;   // 转圈测试持续时间
    // 【新增】耦合测试时间 (Wave, Corkscrew, Drift)
    const double T_COUPLED   = 270.0;  
    // 随机阶段持续时间：设置为极大值，实际上就是一直跑直到你手动停止
    const double T_STOCHASTIC = 99999.0; 

    // --- 安全阈值 (Safety Limits) ---
    const double DEPTH_MIN = 7.0;      // 浅于5米触发保护
    const double DEPTH_MAX = 44.0;     // 深于45米触发保护
    const double SAFE_DEPTH = 20.0;    // 保护触发后的目标回归深度
    const double MAX_TILT  = M_PI / 3.0; // 60度倾角限制

    // --- 内部状态 ---
    double timer = 0.0;
    double phaseTimer = 0.0;
    Phase currentPhase = INIT_STABILIZE;
    bool isStabilizing = true; // 动作间歇标志
    std::vector<OUNoise> noises; // 6轴噪声器

    // --- PID 参数 ---
    struct PIDConfig {
        double kp_z = 2.0, kd_z = 1.0;     // 深度
        double kp_att = 2.5, kd_att = 0.8; // 姿态 (Roll/Pitch)
        double kp_vel = 1.5;               // 速度阻尼
    } pid;

public:
    // 构造函数：初始化噪声发生器
    HybridDataCollectionStrategy(double dt = 0.01) {
        // 配置OU噪声：(theta回弹速度, sigma波动幅度, mean, dt)
        noises.emplace_back(0.15, 0.3, 0.0, dt); // Surge
        noises.emplace_back(0.20, 0.2, 0.0, dt); // Sway
        noises.emplace_back(0.20, 0.3, 0.0, dt); // Heave
        noises.emplace_back(0.50, 0.4, 0.0, dt); // Roll
        noises.emplace_back(0.50, 0.4, 0.0, dt); // Pitch
        noises.emplace_back(0.30, 0.4, 0.0, dt); // Yaw
    }

    void reset() override {
        timer = 0.0;
        phaseTimer = 0.0;
        currentPhase = INIT_STABILIZE;
        isStabilizing = true;
        for(auto& n : noises) n.reset();
        std::cout << "[HybridStrategy] Reset. Starting Sequence." << std::endl;
    }

    ControlOutput calculate(const AUVState& state, double dt) override {
        timer += dt;
        phaseTimer += dt;
        ControlOutput cmd; // 默认为0

        // ==========================================
        // 1. 安全检查 (Safety Check) - 最高优先级
        // ==========================================
        double safetyFactor = calculateSafetyFactor(state);
        
        // 如果处于极度危险状态 (safetyFactor > 0.8)，强制进入 PID 稳态模式
        // 并忽略当前的测试指令
        if (safetyFactor > 0.01) {
            // 计算安全恢复指令 (Target: Depth=20m, Roll=0, Pitch=0, Vel=0)
            ControlOutput safeCmd = calculateStabilizeCmd(state, SAFE_DEPTH);
            
            // 获取当前的动作指令 (如果完全危险则不计算，节省资源)
            ControlOutput actionCmd = (safetyFactor >= 0.99) ? ControlOutput() : runPhaseLogic(state, dt);

            // 线性混合：随着危险程度增加，逐渐从 actionCmd 过渡到 safeCmd
            // Cmd = (1-alpha) * Action + alpha * Safety
            return blendCommands(actionCmd, safeCmd, safetyFactor);
        }

        // ==========================================
        // 2. 状态机逻辑 (State Machine)
        // ==========================================
        
        // 检查当前阶段时间是否结束
        double currentDuration = getPhaseDuration(currentPhase);
        double actualTimeLimit = isStabilizing ? T_STABILIZE : currentDuration;

        if (phaseTimer >= actualTimeLimit) {
            // 切换逻辑
            if (isStabilizing) {
                isStabilizing = false;
                std::cout << ">>> Phase Started: " << getPhaseName(currentPhase) << std::endl;
            } else {
                // 干活完了，休息一下，准备下一阶段
                // 如果是随机阶段，永远不结束
                if (currentPhase != STOCHASTIC_EXPLORE) {
                    currentPhase = static_cast<Phase>(currentPhase + 1);
                    isStabilizing = true;
                    std::cout << "--- Stabilizing (Resting) ---" << std::endl;
                }
            }
            phaseTimer = 0.0; // 重置阶段计时器
        }

        // ==========================================
        // 3. 执行指令
        // ==========================================
        if (isStabilizing) {
            // 间歇期：保持在安全深度，归零姿态，等待水流平稳
            cmd = calculateStabilizeCmd(state, SAFE_DEPTH);
        } else {
            // 动作期
            cmd = runPhaseLogic(state, dt);
        }

        return cmd;
    }

private:
    // --- 核心动作逻辑 ---
    ControlOutput runPhaseLogic(const AUVState& state, double dt) {
        ControlOutput cmd;
        // 正弦波生成：平滑的正弦波，周期20秒
        // output = 0.8 * sin(2*pi*t/20)
        double sineWave = std::sin(2.0 * M_PI * phaseTimer / 20.0);
        
        // 辅助保持力：在测试水平运动时，需要微弱的力保持深度和姿态，否则机器会飘走
        ControlOutput holdCmd = calculateStabilizeCmd(state, SAFE_DEPTH);
        // 减弱保持力，避免干扰测试，但不能完全没有
        holdCmd.heave *= 0.3; 
        holdCmd.pitch *= 0.3;
        holdCmd.roll *= 0.3;

        switch (currentPhase) {
            case TEST_SURGE: // 前后
                cmd = holdCmd; // 保持深度
                cmd.surge = sineWave; 
                cmd.surge = clamp(cmd.surge, 1.0);
                break;

            // case TEST_SWAY: // 左右
            //     cmd = holdCmd;
            //     cmd.sway = sineWave;
            //     break;

            case TEST_HEAVE: // 上下
                cmd = holdCmd;
                // 在保持深度的PID基础上叠加正弦波
                // 注意：这里可能会有些冲突，但为了测试动力学，必须施加动态力
                // 我们直接开环给 Heave 力，但保留姿态控制(Pitch/Roll)
                cmd.heave = sineWave; 
                break;

            case TEST_ROLL:
                cmd = holdCmd;
                cmd.roll = sineWave;
                break;

            case TEST_PITCH:
                cmd = holdCmd;
                cmd.pitch = sineWave;
                break;

            case TEST_YAW:
                cmd = holdCmd;
                cmd.yaw = sineWave;
                break;

            case TEST_CIRCLE_CW: // 顺时针螺旋 (覆盖多种差速比)
            {
                // 归一化进度 (0.0 -> 1.0)
                double progress = phaseTimer / T_CIRCLE;

                // 1. Yaw (转向) 线性递增：从 0.1 (微弯) -> 1.0 (急转)
                // 模拟从“切入弯道”到“死锁急转”的过程
                double targetYaw = 0.1 + 0.9 * progress;

                // 2. Surge (前进) 正弦变化：在 0.3 到 1.0 之间波动
                // 使用 sin(PI * progress) 让它在测试中间达到最高速，两头慢
                // 这样能覆盖 (高速+中弯) 和 (低速+急弯) 等各种组合
                double targetSurge = 0.3 + 0.7 * std::sin(M_PI * progress);

                cmd = holdCmd; // 保持定深/稳姿态
                
                // 转弯时离心力大，稍微加强一点深度保持的权重，防止掉高或飞天
                // (holdCmd 之前被 *0.5 了，这里我们手动补偿回来一点)
                cmd.heave *= 1.5; 
                cmd.roll  *= 1.5; // 防止侧倾过大

                cmd.surge = targetSurge;
                cmd.yaw   = targetYaw;   // 正值 = 顺时针/右转
                
                break;
            }
            
            case TEST_CIRCLE_CCW: // 逆时针螺旋
            {
                double progress = phaseTimer / T_CIRCLE;

                // 逻辑同上，只是 Yaw 取反
                double targetYaw = 0.1 + 0.9 * progress;
                double targetSurge = 0.3 + 0.7 * std::sin(M_PI * progress);

                cmd = holdCmd;
                cmd.heave *= 1.5; 
                cmd.roll  *= 1.5;

                cmd.surge = targetSurge;
                cmd.yaw   = -targetYaw;  // 负值 = 逆时针/左转

                break;
            }

             // 1. 垂直波浪 (Surge + Heave)
            // 场景：机器人在快速前进的同时，大幅度上下潜行。
            // 目的：捕捉前进速度对垂直阻力（升力效应）的影响，以及 Pitch 的耦合力矩。
            case TEST_WAVE_VERTICAL: 
            {
                // Surge: 0.2 到 0.8 的慢速变化 (周期 50s)
                // Heave: -0.8 到 +0.8 的快速变化 (周期 10s)
                // 这种频率差异能保证我们在不同的前进速度下，都测试到了上浮和下潜
                double surgeInput = 0.5 + 0.3 * std::sin(2.0 * M_PI * phaseTimer / 50.0); 
                double heaveInput = 0.8 * std::sin(2.0 * M_PI * phaseTimer / 10.0);

                cmd = holdCmd; 
                
                // 关键：因为我们要主动测试 Heave，必须覆盖掉 holdCmd 里的定深指令
                // 但保留 Pitch/Roll 的稳定控制，防止翻车
                cmd.heave = heaveInput; 
                cmd.surge = surgeInput;
                
                // 稍微加强姿态保持，因为快速升降会带来巨大的 Pitch 力矩
                cmd.pitch *= 2.0; 
                break;
            }

            // 2. 螺旋升降 / 拔塞钻 (Heave + Yaw)
            // 场景：原地旋转的同时下潜或上浮。
            // 目的：测试旋转坐标系下的垂直阻力和科氏力。
            case TEST_CORKSCREW:
            {
                // Heave: 正弦波 (上下往复)
                double heaveInput = 0.8 * std::sin(2.0 * M_PI * phaseTimer / 20.0);
                
                // Yaw: 恒定旋转 (保持一个中等速度旋转)
                // 我们让它转得比较快，把水搅浑
                double yawInput = 0.5; 

                // 如果你想覆盖左右旋，可以让 yawInput 变成方波或者慢速正弦
                // 这里我们让 Yaw 方向随 Heave 变化：下潜时顺时针，上浮时逆时针
                if (heaveInput < 0) yawInput = 0.5; else yawInput = -0.5;

                cmd = holdCmd;
                cmd.surge = 0.0; // 原地
                cmd.heave = heaveInput;
                cmd.yaw   = yawInput;
                
                // 螺旋运动容易晕头转向(Roll不稳定)，加强Roll控制
                cmd.roll *= 2.0; 
                cmd.pitch *= 2.0;
                break;
            }

            // 3. 漂移转弯 (Surge + Sway + Yaw)
            // 场景：模拟侧向推力介入的复杂机动，或者“蟹行”。
            // 目的：即使你不单独测 Sway，这种组合能训练网络理解“斜着走”的动力学。
            case TEST_DRIFT_TURN:
            {
                // Surge: 恒定前进
                double surgeInput = 0.6;
                
                // Yaw: 慢速正弦摆动 (蛇形走位)
                double yawInput = 0.4 * std::sin(2.0 * M_PI * phaseTimer / 15.0);
                
                // Sway: 与 Yaw 相位差 90度 (使用 cos)
                // 这样能产生完美的“圆周漂移”覆盖
                double swayInput = 0.5 * std::cos(2.0 * M_PI * phaseTimer / 15.0);

                cmd = holdCmd;
                cmd.surge = surgeInput;
                cmd.yaw   = yawInput;
                cmd.sway  = swayInput; // 主动输入 Sway
                
                // 深度保持加强
                cmd.heave *= 1.5;
                break;
            }

            case STOCHASTIC_EXPLORE: // 随机全覆盖 (你的最终策略)
                {
                    // 1. 获取6轴随机噪声
                    cmd.surge = noises[0].sample();
                    cmd.sway  = noises[1].sample();
                    cmd.heave = noises[2].sample();
                    cmd.roll  = noises[3].sample();
                    cmd.pitch = noises[4].sample();
                    cmd.yaw   = noises[5].sample();
                    
                    // 2. 即使在随机阶段，如果偏离安全深度太远（比如虽然没触发Safety，但在25m了）
                    // 我们可以施加一个微弱的“引力”把它的中心拉回20m，但不强行锁定
                    // 这样可以避免长时间随机游走到水面
                    double depth_bias = (SAFE_DEPTH - state.z) * 0.05; // 极弱的P
                    cmd.heave += depth_bias;
                }
                break;

            default:
                break;
        }

        return cmd;
    }

    // --- 安全系数计算 (0.0 安全 ~ 1.0 危险) ---
    double calculateSafetyFactor(const AUVState& state) {
        double factor = 0.0;

        // 1. 深度危险
        if (state.z < DEPTH_MIN) { // < 5m
            // 离水面越近，危险系数越大。2m时系数为1.0
            factor = std::max(factor, (DEPTH_MIN - state.z) / 3.0);
        } else if (state.z > DEPTH_MAX) { // > 45m
            factor = std::max(factor, (state.z - DEPTH_MAX) / 3.0);
        }

        // 2. 姿态危险 (Roll/Pitch)
        // double tilt = std::max(std::abs(state.roll), std::abs(state.pitch));
        // if (tilt > MAX_TILT) { // > 60 deg
        //     factor = 1.0; // 直接接管
        // } else if (tilt > M_PI/4.0) { // > 45 deg
        //     // 45~60度之间线性增加
        //     factor = std::max(factor, (tilt - M_PI/4.0) / (MAX_TILT - M_PI/4.0));
        // }

        return std::min(1.0, std::max(0.0, factor));
    }

    // --- 混合指令 ---
    ControlOutput blendCommands(const ControlOutput& action, const ControlOutput& safe, double alpha) {
        ControlOutput finalCmd;
        finalCmd.surge = (1.0 - alpha) * action.surge + alpha * safe.surge;
        finalCmd.sway  = (1.0 - alpha) * action.sway  + alpha * safe.sway;
        finalCmd.heave = (1.0 - alpha) * action.heave + alpha * safe.heave;
        finalCmd.roll  = (1.0 - alpha) * action.roll  + alpha * safe.roll;
        finalCmd.pitch = (1.0 - alpha) * action.pitch + alpha * safe.pitch;
        finalCmd.yaw   = (1.0 - alpha) * action.yaw   + alpha * safe.yaw;
        
        // 全局限幅
        finalCmd.surge = clamp(finalCmd.surge, 1.0);
        finalCmd.sway = clamp(finalCmd.sway, 1.0);
        finalCmd.heave = clamp(finalCmd.heave, 1.0);
        finalCmd.roll = clamp(finalCmd.roll, 1.0);
        finalCmd.pitch = clamp(finalCmd.pitch, 1.0);
        finalCmd.yaw = clamp(finalCmd.yaw, 1.0);
        
        return finalCmd;
    }

    // --- PID 稳定控制器 (用于安全接管) ---
    ControlOutput calculateStabilizeCmd(const AUVState& state, double targetZ) {
        ControlOutput cmd;
        // 深度 P-D
        double err_depth = targetZ - state.z;
        cmd.heave = pid.kp_z * err_depth - pid.kd_z * state.w;
        
        // 姿态 P-D (目标全0)
        cmd.roll  = -pid.kp_att * state.roll  - pid.kd_att * state.p;
        cmd.pitch = -pid.kp_att * state.pitch - pid.kd_att * state.q;
        
        // 水平速度阻尼 (刹车)
        cmd.surge = -pid.kp_vel * state.u;
        cmd.sway  = -pid.kp_vel * state.v;
        cmd.yaw   = -pid.kp_vel * state.r; // 阻止旋转

        // 这里的输出不限幅，交给blend函数去限幅
        return cmd;
    }

    // --- 工具函数 ---
    double clamp(double val, double limit) {
        return std::max(-limit, std::min(limit, val));
    }

    double getPhaseDuration(Phase p) {
        if (p >= TEST_SURGE && p <= TEST_YAW) return T_AXIS_TEST;
        if (p == TEST_CIRCLE_CW || p == TEST_CIRCLE_CCW) return T_CIRCLE;
        // 【修改这里】耦合测试：使用 T_COUPLED 变量，不再写死 50.0
        if (p == TEST_WAVE_VERTICAL || p == TEST_CORKSCREW || p == TEST_DRIFT_TURN) {
            return T_COUPLED;
        }
        if (p == STOCHASTIC_EXPLORE) return T_STOCHASTIC;
        return 5.0; 
    }

    std::string getPhaseName(Phase p) {
        static std::map<Phase, std::string> names = {
            {INIT_STABILIZE, "Init Stabilize"},
            {TEST_SURGE, "Surge Test (Sine)"}, //{TEST_SWAY, "Sway Test (Sine)"},
            {TEST_HEAVE, "Heave Test (Sine)"}, {TEST_ROLL, "Roll Test (Sine)"},
            {TEST_PITCH, "Pitch Test (Sine)"}, {TEST_YAW, "Yaw Test (Sine)"},
            {TEST_CIRCLE_CW, "Circle CW"}, {TEST_CIRCLE_CCW, "Circle CCW"},
            // === 新增名称 ===
            {TEST_WAVE_VERTICAL, "Coupled: Vertical Wave (Surge+Heave)"},
            {TEST_CORKSCREW,     "Coupled: Corkscrew (Heave+Yaw)"},
            {TEST_DRIFT_TURN,    "Coupled: Drift Turn (Surge+Sway+Yaw)"},

            {STOCHASTIC_EXPLORE, ">>> STOCHASTIC EXPLORATION <<<"}
        };
        return names[p];
    }
};

#endif