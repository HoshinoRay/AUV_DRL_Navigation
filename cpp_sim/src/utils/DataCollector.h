#ifndef DATA_COLLECTOR_H
#define DATA_COLLECTOR_H

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>
#include <iomanip>
#include <sstream>

// 将 AUVState 定义移到这里，方便多处使用
struct AUVState {
    double x, y, z;       // NED 位置
    double roll, pitch, yaw; // 欧拉角 (rad)
    double u, v, w;       // 机体线速度 (Surge, Sway, Heave)
    double p, q, r;       // 机体角速度 (RollRate, PitchRate, YawRate)
    // 【新增】8个推进器的 PWM 值 (归一化 -1.0 到 1.0)
    std::vector<double> motors; 
};

class DataCollector {
public:
    DataCollector() : is_open(false) {}

    ~DataCollector() {
        close();
    }

    // 初始化日志文件，自动生成带时间戳的文件名
    bool init(const std::string& directory, const std::string& prefix = "auv_log_") {
        std::string filename = directory + "/" + prefix + getCurrentTimestamp() + ".csv";
        
        outFile.open(filename);
        if (!outFile.is_open()) {
            std::cerr << "[DataCollector] Error: Could not open file " << filename << std::endl;
            return false;
        }

        is_open = true;
        std::cout << "[DataCollector] Logging to: " << filename << std::endl;

        // 写入 CSV 表头
        outFile << "Time(s),"
                << "X(m),Y(m),Z(m),"
                << "Roll(rad),Pitch(rad),Yaw(rad),"
                << "u(m/s),v(m/s),w(m/s),"
                << "p(rad/s),q(rad/s),r(rad/s)," 
                << "M_FL,M_FR,M_RL,M_RR,M_VFL,M_VFR,M_VRL,M_VRR" // 8个电机
                << "\n";
        
        return true;
    }

    // 记录单帧数据
    void log(double time, const AUVState& state) {
        if (!is_open) return;

        // 使用高精度写入
        outFile << std::fixed << std::setprecision(4)
                << time << ","
                << state.x << "," << state.y << "," << state.z << ","
                << state.roll << "," << state.pitch << "," << state.yaw << ","
                << state.u << "," << state.v << "," << state.w << ","
                << state.p << "," << state.q << "," << state.r;
        // 【新增】遍历并写入 8 个电机的数据
        // 也就是把 motors 里的数据依次填入
        for (double m : state.motors) {
            outFile << "," << m;
        }

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

    // 获取当前时间字符串用于文件名
    std::string getCurrentTimestamp() {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
        return oss.str();
    }
};

#endif // DATA_COLLECTOR_H