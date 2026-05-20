#ifndef MY_AUV_MANAGER_H
#define MY_AUV_MANAGER_H

#include <core/SimulationManager.h>
#include "utils/DataCollector.h"
#include "controllers/MotionController.h"

class MyAUVManager : public sf::SimulationManager
{
public:
    MyAUVManager(sf::Scalar stepsPerSecond);
    ~MyAUVManager();

    void BuildScenario();
    void SimulationStepCompleted(sf::Scalar timeStep);

private:
    const std::vector<std::string> thrusterNames = {
        "HorzFrontLeft", "HorzFrontRight", "HorzRearLeft", "HorzRearRight",
        "VertFrontLeft", "VertFrontRight", "VertRearLeft", "VertRearRight"
    };

    AUVState getAUVState();

    DataCollector*    dataCollector;
    double            currentTime;
    MotionController* motionController;
};

#endif
