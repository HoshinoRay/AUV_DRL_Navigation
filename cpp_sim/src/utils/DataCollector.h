#ifndef DATA_COLLECTOR_H
#define DATA_COLLECTOR_H

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>
#include <iomanip>
#include <sstream>

// AUV state snapshot: NED position, Euler angles, body-frame velocities, and motor setpoints.
struct AUVState {
    double x, y, z;           // NED position (m)
    double roll, pitch, yaw;  // Euler angles (rad)
    double u, v, w;           // Body-frame linear velocity: surge, sway, heave (m/s)
    double p, q, r;           // Body-frame angular velocity: roll, pitch, yaw rate (rad/s)
    std::vector<double> motors; // Normalised thruster setpoints [-1, 1], length 8
};

class DataCollector {
public:
    DataCollector() : is_open(false) {}
    ~DataCollector() { close(); }

    // Open a timestamped CSV file in the given directory.
    bool init(const std::string& directory, const std::string& prefix = "auv_log_") {
        std::string filename = directory + "/" + prefix + getCurrentTimestamp() + ".csv";
        outFile.open(filename);
        if (!outFile.is_open()) {
            std::cerr << "[DataCollector] Error: could not open " << filename << std::endl;
            return false;
        }
        is_open = true;
        std::cout << "[DataCollector] Logging to: " << filename << std::endl;

        outFile << "Time(s),"
                << "X(m),Y(m),Z(m),"
                << "Roll(rad),Pitch(rad),Yaw(rad),"
                << "u(m/s),v(m/s),w(m/s),"
                << "p(rad/s),q(rad/s),r(rad/s),"
                << "M_FL,M_FR,M_RL,M_RR,M_VFL,M_VFR,M_VRL,M_VRR"
                << "\n";
        return true;
    }

    void log(double time, const AUVState& state) {
        if (!is_open) return;
        outFile << std::fixed << std::setprecision(4)
                << time << ","
                << state.x << "," << state.y << "," << state.z << ","
                << state.roll << "," << state.pitch << "," << state.yaw << ","
                << state.u << "," << state.v << "," << state.w << ","
                << state.p << "," << state.q << "," << state.r;
        for (double m : state.motors) outFile << "," << m;
        outFile << "\n";
    }

    void close() {
        if (is_open) {
            outFile.close();
            is_open = false;
            std::cout << "[DataCollector] Log file closed." << std::endl;
        }
    }

private:
    std::ofstream outFile;
    bool is_open;

    std::string getCurrentTimestamp() {
        auto t  = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
        return oss.str();
    }
};

#endif // DATA_COLLECTOR_H
