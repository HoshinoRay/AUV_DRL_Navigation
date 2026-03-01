#ifndef MY_AUV_MANAGER_H
#define MY_AUV_MANAGER_H

#include <core/SimulationManager.h>
#include "utils/DataCollector.h" // 【新增】引入数据采集器
#include "controllers/MotionController.h" 

//#define PARSED_SCENARIO

class MyAUVManager : public sf::SimulationManager
{
public:
    MyAUVManager(sf::Scalar stepsPerSecond);
    
    // 析构函数（建议加上，用于确保文件关闭）
    ~MyAUVManager();

    void BuildScenario();
    void SimulationStepCompleted(sf::Scalar timeStep);

    private:
    // 恢复这个定义，用于遍历读取数据
    const std::vector<std::string> thrusterNames = {
        "HorzFrontLeft", "HorzFrontRight", "HorzRearLeft", "HorzRearRight",
        "VertFrontLeft", "VertFrontRight", "VertRearLeft", "VertRearRight"
    };
    
    // 声明一个辅助函数来获取状态
    AUVState getAUVState();
    
    // 【新增】数据采集相关变量
    DataCollector* dataCollector; 
    double currentTime; // 用于记录仿真流逝的时间
    // 【新增】运动控制器
    MotionController* motionController;

                       
};

#endif