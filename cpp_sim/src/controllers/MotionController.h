#ifndef MOTION_CONTROLLER_H
#define MOTION_CONTROLLER_H

#include "MotionStrategy.h"
#include <core/Robot.h>
#include <actuators/Thruster.h>
#include <vector>
#include <string>
#include <memory>

class MotionController {
public:
    MotionController() {
        // 默认初始化为空闲策略，防止空指针
        currentStrategy = std::make_shared<IdleStrategy>();
    }

    // 切换策略 (使用 shared_ptr 自动管理内存)
    void setStrategy(std::shared_ptr<MotionStrategy> newStrategy) {
        if (newStrategy) {
            currentStrategy = newStrategy;
            currentStrategy->reset();
            std::cout << "[MotionController] Strategy switched." << std::endl;
        }
    }

    // 核心执行函数：获取指令 -> 混合 -> 发送给机器人
    void update(sf::Robot* robot, const AUVState& state, double dt) {
        if (!robot || !currentStrategy) return;

        // 1. 计算所需控制力 (6-DOF)
        ControlOutput cmd = currentStrategy->calculate(state, dt);

        // 2. 动力分配 (Mixing) - 将 6-DOF 转换为 8 个推进器信号
        applyThrusts(robot, cmd);
    }

private:
    std::shared_ptr<MotionStrategy> currentStrategy;

    // 推进器名称列表 (对应 MyAUVManager 中的定义)
    const std::vector<std::string> thrusterNames = {
        "HorzFrontLeft",  "HorzFrontRight", "HorzRearLeft",   "HorzRearRight",
        "VertFrontLeft",  "VertFrontRight", "VertRearLeft",   "VertRearRight"
    };

    // 动力分配逻辑 (从 MyAUVManager 迁移过来)
    void applyThrusts(sf::Robot* robot, const ControlOutput& cmd) {
        // --- 水平推进器混控 ---
        // FL, FR, RL, RR
        double t_fl = cmd.surge + cmd.sway + cmd.yaw;
        double t_fr = cmd.surge - cmd.sway - cmd.yaw;
        double t_rl = cmd.surge - cmd.sway + cmd.yaw;
        double t_rr = cmd.surge + cmd.sway - cmd.yaw;

        // --- 垂直推进器混控 ---
        // VFL, VFR, VRL, VRR
        double t_vfl = cmd.heave - cmd.pitch - cmd.roll;
        double t_vfr = cmd.heave - cmd.pitch + cmd.roll;
        double t_vrl = cmd.heave + cmd.pitch - cmd.roll;
        double t_vrr = cmd.heave + cmd.pitch + cmd.roll;

        // 归一化处理 (保持原逻辑)
        std::vector<double> cmds = {t_fl, t_fr, t_rl, t_rr, t_vfl, t_vfr, t_vrl, t_vrr};
        double max_val = 1.0;
        for (double v : cmds) {
            if (std::fabs(v) > max_val) max_val = std::fabs(v);
        }

        // 发送给机器人
        for (size_t i = 0; i < thrusterNames.size(); i++) {
            sf::Thruster* th = dynamic_cast<sf::Thruster*>(robot->getActuator(thrusterNames[i]));
            if (th) {
                th->setSetpoint(cmds[i] / max_val);
            }
        }
    }
};

#endif