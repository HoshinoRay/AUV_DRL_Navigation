#ifndef MOTION_STRATEGY_H
#define MOTION_STRATEGY_H

#include "../utils/DataCollector.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <map>

// Normalised 6-DOF force/torque demand in [-1.0, 1.0]
struct ControlOutput {
    double surge = 0.0;
    double sway  = 0.0;
    double heave = 0.0;
    double roll  = 0.0;
    double pitch = 0.0;
    double yaw   = 0.0;
};

class MotionStrategy {
public:
    virtual ~MotionStrategy() {}
    virtual ControlOutput calculate(const AUVState& state, double dt) = 0;
    virtual void reset() {}
};

class IdleStrategy : public MotionStrategy {
public:
    ControlOutput calculate(const AUVState& state, double dt) override {
        return ControlOutput{};
    }
};

// PID station-keeping strategy: maintains depth, attitude, and a constant surge speed.
class PIDStationKeepingStrategy : public MotionStrategy {
private:
    double Kp_roll  = 2.0, Kd_roll  = 0.5;
    double Kp_pitch = 3.0, Kd_pitch = 0.8;
    double Kp_depth = 1.5, Kd_depth = 0.5;
    double Kp_yaw   = 1.0, Kd_yaw   = 0.3;
    double Kp_surge = 1.0, Kd_surge = 0.0;
    double Kp_sway  = 1.0, Kd_sway  = 0.2;

    double targetDepth = 15.0;
    double targetYaw   = 0.0;
    double targetRoll  = 0.0;
    double targetPitch = 0.0;
    double targetSurge = 0.5;

    double normalizeAngle(double angle) {
        while (angle >  M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

    double clamp(double val, double limit) {
        if (val >  limit) return  limit;
        if (val < -limit) return -limit;
        return val;
    }

public:
    PIDStationKeepingStrategy(double depth, double surgeVel)
        : targetDepth(depth), targetSurge(surgeVel) {}

    ControlOutput calculate(const AUVState& state, double dt) override {
        ControlOutput cmd;

        // Attitude
        cmd.roll  = Kp_roll  * normalizeAngle(targetRoll  - state.roll)  - Kd_roll  * state.p;
        cmd.pitch = Kp_pitch * normalizeAngle(targetPitch - state.pitch) - Kd_pitch * state.q;

        // Depth — scale by pitch to protect against flip-over
        double err_depth       = targetDepth - state.z;
        double depth_gain_scale = std::max(0.0, 1.0 - std::abs(state.pitch) / (M_PI / 4.0));
        cmd.heave = (Kp_depth * err_depth - Kd_depth * state.w) * depth_gain_scale;

        // Heading
        cmd.yaw   = Kp_yaw * normalizeAngle(targetYaw - state.yaw) - Kd_yaw * state.r;

        // Surge / sway
        cmd.surge = Kp_surge * (targetSurge - state.u);
        cmd.sway  = Kp_sway  * (0.0 - state.v);

        cmd.surge = clamp(cmd.surge, 0.6);
        cmd.sway  = clamp(cmd.sway,  0.5);
        cmd.heave = clamp(cmd.heave, 0.8);
        cmd.roll  = clamp(cmd.roll,  0.5);
        cmd.pitch = clamp(cmd.pitch, 1.0);
        cmd.yaw   = clamp(cmd.yaw,   0.5);

        return cmd;
    }
};

// Step-velocity data collection strategy for system identification.
// Sweeps surge through a ramp sequence of speed steps and holds attitude with PID.
class MaxThrustStrategy : public MotionStrategy {
private:
    double timer           = 0.0;
    int    stepIndex       = 0;
    double currentSurgeCmd = 0.0;

    const double STEP_DURATION = 8.0;  // seconds per speed step
    const double RAMP_RATE     = 0.2;  // setpoint change per second
    const double TARGET_DEPTH  = 15.0;

    std::vector<double> SPEED_STEPS = {
        0.0, 0.25, 0.50, 0.75, 1.00,
        0.75, 0.50, 0.25, 0.00,
        -0.25, -0.50, -0.75, -1.00,
        -0.75, -0.50, -0.25, 0.00
    };

    struct PIDParams {
        double kp_z = 2.0, ki_z = 0.05, kd_z = 1.5;
        double kp_p = 3.0, kd_p = 1.0;
        double kp_r = 2.0, kd_r = 0.5;
        double integral_z = 0.0;
    } pid;

public:
    void reset() override {
        timer = 0.0; stepIndex = 0; currentSurgeCmd = 0.0; pid.integral_z = 0.0;
        std::cout << "[MaxThrust] Reset. Starting Step-Velocity Data Collection." << std::endl;
    }

    ControlOutput calculate(const AUVState& state, double dt) override {
        ControlOutput cmd;
        timer += dt;

        // Advance to the next speed step when the current one expires
        if (timer >= STEP_DURATION) {
            timer = 0.0;
            stepIndex = (stepIndex + 1) % (int)SPEED_STEPS.size();
            std::cout << "[DataCollection] Step " << stepIndex
                      << ": Target Surge = " << SPEED_STEPS[stepIndex] << std::endl;
        }

        // Ramp generator: slew currentSurgeCmd toward target at RAMP_RATE
        double diff       = SPEED_STEPS[stepIndex] - currentSurgeCmd;
        double max_change = RAMP_RATE * dt;
        if (std::abs(diff) <= max_change)
            currentSurgeCmd = SPEED_STEPS[stepIndex];
        else
            currentSurgeCmd += (diff > 0 ? max_change : -max_change);

        cmd.surge = currentSurgeCmd;

        // Depth PID with integral anti-windup
        double err_depth = TARGET_DEPTH - state.z;
        if (std::abs(err_depth) < 1.0) {
            pid.integral_z = std::max(-2.0, std::min(2.0, pid.integral_z + err_depth * dt));
        } else {
            pid.integral_z = 0.0;
        }
        cmd.heave = pid.kp_z * err_depth + pid.ki_z * pid.integral_z - pid.kd_z * state.w;

        // Attitude stabilisation
        cmd.pitch = pid.kp_p * (0.0 - state.pitch) - pid.kd_p * state.q;
        cmd.roll  = pid.kp_r * (0.0 - state.roll)  - pid.kd_r * state.p;
        cmd.sway  = 0.0;
        cmd.yaw   = 0.0;

        cmd.heave = std::max(-1.0, std::min(1.0, cmd.heave));
        cmd.pitch = std::max(-1.0, std::min(1.0, cmd.pitch));
        cmd.roll  = std::max(-1.0, std::min(1.0, cmd.roll));

        return cmd;
    }
};

// Zero-gravity optimised data collection strategy.
// Exercises each DOF in sequence with stabilisation intervals between manoeuvres.
class DataCollectionStrategy : public MotionStrategy {
private:
    double timer        = 0.0;
    int    currentPhase = 0;

    const double SAFE_DEPTH      = 15.0;
    const double STABILIZE_TIME  = 10.0;
    const double PHASE_DURATION  = 20.0;

    struct PIDConfig {
        double kp_z = 2.0, kd_z = 1.0;
        double kp_att = 1.5, kd_att = 0.5;
        double kp_vel = 2.0;
    } pid;

    double getTriangleWave(double t, double period) {
        double u = std::min(t / period, 1.0);
        if      (u < 0.25) return u * 4.0;
        else if (u < 0.75) return 1.0 - (u - 0.25) * 4.0;
        else               return -1.0 + (u - 0.75) * 4.0;
    }

    ControlOutput runStabilize(const AUVState& state, double targetYawRate = 0.0) {
        ControlOutput cmd;
        cmd.heave = pid.kp_z   * (SAFE_DEPTH - state.z) - pid.kd_z   * state.w;
        cmd.roll  = -pid.kp_att * state.roll  - pid.kd_att * state.p;
        cmd.pitch = -pid.kp_att * state.pitch - pid.kd_att * state.q;
        cmd.surge = -pid.kp_vel * state.u;
        cmd.sway  = -pid.kp_vel * state.v;
        cmd.yaw   = (std::abs(targetYawRate) < 0.001)
                  ? -1.0 * state.r
                  : -0.5 * (state.r - targetYawRate);

        cmd.heave = std::max(-1.0, std::min(1.0, cmd.heave));
        cmd.surge = std::max(-0.5, std::min(0.5, cmd.surge));
        cmd.sway  = std::max(-0.5, std::min(0.5, cmd.sway));
        cmd.roll  = std::max(-0.5, std::min(0.5, cmd.roll));
        cmd.pitch = std::max(-0.5, std::min(0.5, cmd.pitch));
        return cmd;
    }

public:
    void reset() override {
        timer = 0.0; currentPhase = 0;
        std::cout << "[DataCollection] Strategy Reset. Starting Safe Sequence." << std::endl;
    }

    ControlOutput calculate(const AUVState& state, double dt) override {
        ControlOutput cmd;
        timer += dt;

        // Emergency surface-proximity override
        bool safetyTrigger = (currentPhase % 2 == 0) && (state.z < 2.0);
        if (safetyTrigger) {
            std::cout << "[WARNING] Near Surface! Emergency Dive!" << std::endl;
            cmd = runStabilize(state);
            cmd.heave = -1.0;
            return cmd;
        }

        switch (currentPhase) {
            case 0: cmd.surge = getTriangleWave(timer, PHASE_DURATION); if (timer >= PHASE_DURATION) nextPhase(); break;
            case 1: cmd = runStabilize(state); if (timer >= STABILIZE_TIME) nextPhase(); break;
            case 2: cmd.heave = getTriangleWave(timer, PHASE_DURATION); if (timer >= PHASE_DURATION) nextPhase(); break;
            case 3: cmd = runStabilize(state); if (timer >= STABILIZE_TIME) nextPhase(); break;
            case 4: cmd.yaw   = getTriangleWave(timer, PHASE_DURATION);
                    cmd.heave = (SAFE_DEPTH - state.z) * 1.0 - 0.5 * state.w;
                    if (timer >= PHASE_DURATION) nextPhase(); break;
            case 5: cmd = runStabilize(state); if (timer >= STABILIZE_TIME) nextPhase(); break;
            case 6: cmd.roll  = getTriangleWave(timer, PHASE_DURATION);
                    cmd.heave = (SAFE_DEPTH - state.z) * 1.0 - 0.5 * state.w;
                    if (timer >= PHASE_DURATION) nextPhase(); break;
            case 7: cmd = runStabilize(state); if (timer >= STABILIZE_TIME) nextPhase(); break;
            default: cmd = runStabilize(state, 0.0); break;
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

// Ornstein-Uhlenbeck noise generator (bounded to [-1, 1]).
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
        val += theta * (mean - val) * dt + sigma * std::sqrt(dt) * dist(gen);
        return std::max(-1.0, std::min(1.0, val));
    }
    void reset() { val = mean; }
};

// Hybrid data collection strategy.
// Runs a series of deterministic single-axis and coupled-axis manoeuvres
// followed by open-ended stochastic exploration using OU noise.
class HybridDataCollectionStrategy : public MotionStrategy {
public:
    enum Phase {
        INIT_STABILIZE = 0,
        TEST_SURGE,
        TEST_HEAVE,
        TEST_ROLL,
        TEST_PITCH,
        TEST_YAW,
        TEST_CIRCLE_CW,
        TEST_CIRCLE_CCW,
        TEST_WAVE_VERTICAL,  // surge + heave coupling
        TEST_CORKSCREW,      // heave + yaw coupling
        TEST_DRIFT_TURN,     // surge + sway + yaw coupling
        STOCHASTIC_EXPLORE,
        _PHASE_COUNT
    };

private:
    const double T_STABILIZE  = 5.0;
    const double T_AXIS_TEST  = 180.0;
    const double T_CIRCLE     = 180.0;
    const double T_COUPLED    = 270.0;
    const double T_STOCHASTIC = 99999.0;

    const double DEPTH_MIN  = 7.0;
    const double DEPTH_MAX  = 44.0;
    const double SAFE_DEPTH = 20.0;
    const double MAX_TILT   = M_PI / 3.0;

    double timer      = 0.0;
    double phaseTimer = 0.0;
    Phase  currentPhase = INIT_STABILIZE;
    bool   isStabilizing = true;

    std::vector<OUNoise> noises;

    struct PIDConfig {
        double kp_z = 2.0, kd_z = 1.0;
        double kp_att = 2.5, kd_att = 0.8;
        double kp_vel = 1.5;
    } pid;

public:
    HybridDataCollectionStrategy(double dt = 0.01) {
        noises.emplace_back(0.15, 0.3, 0.0, dt); // surge
        noises.emplace_back(0.20, 0.2, 0.0, dt); // sway
        noises.emplace_back(0.20, 0.3, 0.0, dt); // heave
        noises.emplace_back(0.50, 0.4, 0.0, dt); // roll
        noises.emplace_back(0.50, 0.4, 0.0, dt); // pitch
        noises.emplace_back(0.30, 0.4, 0.0, dt); // yaw
    }

    void reset() override {
        timer = 0.0; phaseTimer = 0.0;
        currentPhase = INIT_STABILIZE; isStabilizing = true;
        for (auto& n : noises) n.reset();
        std::cout << "[HybridStrategy] Reset. Starting Sequence." << std::endl;
    }

    ControlOutput calculate(const AUVState& state, double dt) override {
        timer      += dt;
        phaseTimer += dt;

        // Safety blending: smoothly override action with stabilise command near limits
        double safetyFactor = calculateSafetyFactor(state);
        if (safetyFactor > 0.01) {
            ControlOutput safeCmd   = calculateStabilizeCmd(state, SAFE_DEPTH);
            ControlOutput actionCmd = (safetyFactor >= 0.99) ? ControlOutput() : runPhaseLogic(state, dt);
            return blendCommands(actionCmd, safeCmd, safetyFactor);
        }

        double currentDuration = getPhaseDuration(currentPhase);
        double timeLimit       = isStabilizing ? T_STABILIZE : currentDuration;

        if (phaseTimer >= timeLimit) {
            if (isStabilizing) {
                isStabilizing = false;
                std::cout << ">>> Phase Started: " << getPhaseName(currentPhase) << std::endl;
            } else {
                if (currentPhase != STOCHASTIC_EXPLORE) {
                    currentPhase  = static_cast<Phase>(currentPhase + 1);
                    isStabilizing = true;
                    std::cout << "--- Stabilizing (Resting) ---" << std::endl;
                }
            }
            phaseTimer = 0.0;
        }

        return isStabilizing ? calculateStabilizeCmd(state, SAFE_DEPTH) : runPhaseLogic(state, dt);
    }

private:
    ControlOutput runPhaseLogic(const AUVState& state, double dt) {
        ControlOutput cmd;
        double sineWave = std::sin(2.0 * M_PI * phaseTimer / 20.0);

        ControlOutput holdCmd = calculateStabilizeCmd(state, SAFE_DEPTH);
        holdCmd.heave *= 0.3;
        holdCmd.pitch *= 0.3;
        holdCmd.roll  *= 0.3;

        switch (currentPhase) {
            case TEST_SURGE:
                cmd = holdCmd;
                cmd.surge = clamp(sineWave, 1.0);
                break;

            case TEST_HEAVE:
                cmd = holdCmd;
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

            case TEST_CIRCLE_CW: {
                double progress = phaseTimer / T_CIRCLE;
                cmd = holdCmd;
                cmd.heave  *= 1.5;
                cmd.roll   *= 1.5;
                cmd.surge = 0.3 + 0.7 * std::sin(M_PI * progress);
                cmd.yaw   = 0.1 + 0.9 * progress;   // clockwise
                break;
            }

            case TEST_CIRCLE_CCW: {
                double progress = phaseTimer / T_CIRCLE;
                cmd = holdCmd;
                cmd.heave  *= 1.5;
                cmd.roll   *= 1.5;
                cmd.surge = 0.3 + 0.7 * std::sin(M_PI * progress);
                cmd.yaw   = -(0.1 + 0.9 * progress); // counter-clockwise
                break;
            }

            case TEST_WAVE_VERTICAL: {
                // Surge + heave coupling: slow surge variation, fast heave oscillation
                cmd = holdCmd;
                cmd.heave  = 0.8 * std::sin(2.0 * M_PI * phaseTimer / 10.0);
                cmd.surge  = 0.5 + 0.3 * std::sin(2.0 * M_PI * phaseTimer / 50.0);
                cmd.pitch *= 2.0;
                break;
            }

            case TEST_CORKSCREW: {
                // Heave + yaw coupling: vertical oscillation while rotating
                double heaveInput = 0.8 * std::sin(2.0 * M_PI * phaseTimer / 20.0);
                cmd = holdCmd;
                cmd.surge = 0.0;
                cmd.heave = heaveInput;
                cmd.yaw   = (heaveInput < 0) ? 0.5 : -0.5;
                cmd.roll  *= 2.0;
                cmd.pitch *= 2.0;
                break;
            }

            case TEST_DRIFT_TURN: {
                // Surge + sway + yaw: snake-drift pattern covering lateral dynamics
                cmd = holdCmd;
                cmd.heave *= 1.5;
                cmd.surge = 0.6;
                cmd.yaw   = 0.4 * std::sin(2.0 * M_PI * phaseTimer / 15.0);
                cmd.sway  = 0.5 * std::cos(2.0 * M_PI * phaseTimer / 15.0);
                break;
            }

            case STOCHASTIC_EXPLORE: {
                cmd.surge = noises[0].sample();
                cmd.sway  = noises[1].sample();
                cmd.heave = noises[2].sample();
                cmd.roll  = noises[3].sample();
                cmd.pitch = noises[4].sample();
                cmd.yaw   = noises[5].sample();
                // Weak depth bias to prevent unbounded drift
                cmd.heave += (SAFE_DEPTH - state.z) * 0.05;
                break;
            }

            default:
                break;
        }
        return cmd;
    }

    double calculateSafetyFactor(const AUVState& state) {
        double factor = 0.0;
        if      (state.z < DEPTH_MIN) factor = std::max(factor, (DEPTH_MIN - state.z) / 3.0);
        else if (state.z > DEPTH_MAX) factor = std::max(factor, (state.z - DEPTH_MAX) / 3.0);
        return std::min(1.0, std::max(0.0, factor));
    }

    ControlOutput blendCommands(const ControlOutput& action, const ControlOutput& safe, double alpha) {
        ControlOutput out;
        out.surge = clamp((1.0 - alpha) * action.surge + alpha * safe.surge, 1.0);
        out.sway  = clamp((1.0 - alpha) * action.sway  + alpha * safe.sway,  1.0);
        out.heave = clamp((1.0 - alpha) * action.heave + alpha * safe.heave, 1.0);
        out.roll  = clamp((1.0 - alpha) * action.roll  + alpha * safe.roll,  1.0);
        out.pitch = clamp((1.0 - alpha) * action.pitch + alpha * safe.pitch, 1.0);
        out.yaw   = clamp((1.0 - alpha) * action.yaw   + alpha * safe.yaw,   1.0);
        return out;
    }

    ControlOutput calculateStabilizeCmd(const AUVState& state, double targetZ) {
        ControlOutput cmd;
        cmd.heave = pid.kp_z   * (targetZ - state.z) - pid.kd_z   * state.w;
        cmd.roll  = -pid.kp_att * state.roll  - pid.kd_att * state.p;
        cmd.pitch = -pid.kp_att * state.pitch - pid.kd_att * state.q;
        cmd.surge = -pid.kp_vel * state.u;
        cmd.sway  = -pid.kp_vel * state.v;
        cmd.yaw   = -pid.kp_vel * state.r;
        return cmd;
    }

    double clamp(double val, double limit) {
        return std::max(-limit, std::min(limit, val));
    }

    double getPhaseDuration(Phase p) {
        if (p >= TEST_SURGE && p <= TEST_YAW)              return T_AXIS_TEST;
        if (p == TEST_CIRCLE_CW || p == TEST_CIRCLE_CCW)   return T_CIRCLE;
        if (p == TEST_WAVE_VERTICAL || p == TEST_CORKSCREW || p == TEST_DRIFT_TURN)
                                                            return T_COUPLED;
        if (p == STOCHASTIC_EXPLORE)                        return T_STOCHASTIC;
        return 5.0;
    }

    std::string getPhaseName(Phase p) {
        static std::map<Phase, std::string> names = {
            {INIT_STABILIZE,    "Init Stabilize"},
            {TEST_SURGE,        "Surge Test (Sine)"},
            {TEST_HEAVE,        "Heave Test (Sine)"},
            {TEST_ROLL,         "Roll Test (Sine)"},
            {TEST_PITCH,        "Pitch Test (Sine)"},
            {TEST_YAW,          "Yaw Test (Sine)"},
            {TEST_CIRCLE_CW,    "Circle CW"},
            {TEST_CIRCLE_CCW,   "Circle CCW"},
            {TEST_WAVE_VERTICAL,"Coupled: Vertical Wave (Surge+Heave)"},
            {TEST_CORKSCREW,    "Coupled: Corkscrew (Heave+Yaw)"},
            {TEST_DRIFT_TURN,   "Coupled: Drift Turn (Surge+Sway+Yaw)"},
            {STOCHASTIC_EXPLORE,">>> STOCHASTIC EXPLORATION <<<"},
        };
        return names[p];
    }
};

#endif
