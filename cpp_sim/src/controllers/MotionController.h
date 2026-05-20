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
        currentStrategy = std::make_shared<IdleStrategy>();
    }

    void setStrategy(std::shared_ptr<MotionStrategy> newStrategy) {
        if (newStrategy) {
            currentStrategy = newStrategy;
            currentStrategy->reset();
            std::cout << "[MotionController] Strategy switched." << std::endl;
        }
    }

    // Compute 6-DOF demand from the active strategy and distribute to thrusters.
    void update(sf::Robot* robot, const AUVState& state, double dt) {
        if (!robot || !currentStrategy) return;
        ControlOutput cmd = currentStrategy->calculate(state, dt);
        applyThrusts(robot, cmd);
    }

private:
    std::shared_ptr<MotionStrategy> currentStrategy;

    const std::vector<std::string> thrusterNames = {
        "HorzFrontLeft",  "HorzFrontRight", "HorzRearLeft",   "HorzRearRight",
        "VertFrontLeft",  "VertFrontRight", "VertRearLeft",   "VertRearRight"
    };

    // Mix 6-DOF demand to 8 individual thruster setpoints and normalise.
    void applyThrusts(sf::Robot* robot, const ControlOutput& cmd) {
        // Horizontal thrusters: FL, FR, RL, RR
        double t_fl = cmd.surge + cmd.sway + cmd.yaw;
        double t_fr = cmd.surge - cmd.sway - cmd.yaw;
        double t_rl = cmd.surge - cmd.sway + cmd.yaw;
        double t_rr = cmd.surge + cmd.sway - cmd.yaw;

        // Vertical thrusters: VFL, VFR, VRL, VRR
        double t_vfl = cmd.heave - cmd.pitch - cmd.roll;
        double t_vfr = cmd.heave - cmd.pitch + cmd.roll;
        double t_vrl = cmd.heave + cmd.pitch - cmd.roll;
        double t_vrr = cmd.heave + cmd.pitch + cmd.roll;

        std::vector<double> cmds = {t_fl, t_fr, t_rl, t_rr, t_vfl, t_vfr, t_vrl, t_vrr};

        double max_val = 1.0;
        for (double v : cmds) {
            if (std::fabs(v) > max_val) max_val = std::fabs(v);
        }

        for (size_t i = 0; i < thrusterNames.size(); i++) {
            sf::Thruster* th = dynamic_cast<sf::Thruster*>(robot->getActuator(thrusterNames[i]));
            if (th) th->setSetpoint(cmds[i] / max_val);
        }
    }
};

#endif
