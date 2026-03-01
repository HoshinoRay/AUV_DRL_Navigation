#include "MyAUVApp.h" 
#include <actuators/Servo.h>
#include <actuators/Thruster.h>
#include <actuators/VariableBuoyancy.h>
#include <core/Robot.h>
#include <sensors/scalar/Accelerometer.h>
#include <sensors/scalar/IMU.h>
#include <sensors/scalar/DVL.h>
#include <sensors/vision/FLS.h>
#include <sensors/vision/SSS.h>
#include <graphics/IMGUI.h>
#include <utils/SystemUtil.hpp>
#include <comms/USBL.h>
#include <core/Console.h>
#include <cstdio>
#include <glm/glm.hpp> 

MyAUVApp::MyAUVApp(std::string dataDirPath, sf::RenderSettings s, sf::HelperSettings h, MyAUVManager *sim)
    : GraphicalSimulationApp("Underwater Test", dataDirPath, s, h, sim)
{
}

void MyAUVApp::InitializeGUI()
{
    largePrint = new sf::OpenGLPrinter(sf::GetShaderPath() + std::string(STANDARD_FONT_NAME), 64.0);
}

void MyAUVApp::DoHUD()
{
    // 调用父类 HUD
    GraphicalSimulationApp::DoHUD();

    // 获取机器人指针
    sf::Robot* robot = this->getSimulationManager()->getRobot("GIRONA500");
    
    if(robot)
    {
        sf::Transform transform = robot->getTransform();
        const sf::Vector3& pos = transform.getOrigin();

        btScalar yaw, pitch, roll;
        transform.getBasis().getEulerYPR(yaw, pitch, roll);
        sf::SolidEntity* vehicle = robot->getLink("Vehicle");
        sf::Vector3 bodyLinVel(0, 0, 0);

        if(vehicle)
        {
            sf::Vector3 worldLinVel = vehicle->getLinearVelocity();
            bodyLinVel = transform.getBasis().transpose() * worldLinVel;
        }

        glm::vec4 color(1.0f, 1.0f, 1.0f, 1.0f);
        float fontSize = 16.0f;
        char buffer[128]; 

        std::snprintf(buffer, sizeof(buffer), "POS X: %.2f m", pos.x());
        largePrint->Print(std::string(buffer), color, 10, 10, fontSize);

        std::snprintf(buffer, sizeof(buffer), "POS Y: %.2f m", pos.y());
        largePrint->Print(std::string(buffer), color, 10, 30, fontSize);

        std::snprintf(buffer, sizeof(buffer), "DEPTH: %.2f m", pos.z());
        largePrint->Print(std::string(buffer), color, 10, 50, fontSize);

        std::snprintf(buffer, sizeof(buffer), "Roll:  %.1f deg", roll * 180.0 / M_PI);
        largePrint->Print(std::string(buffer), color, 10, 80, fontSize);

        std::snprintf(buffer, sizeof(buffer), "Pitch: %.1f deg", pitch * 180.0 / M_PI);
        largePrint->Print(std::string(buffer), color, 10, 100, fontSize);

        std::snprintf(buffer, sizeof(buffer), "Yaw:   %.1f deg", yaw * 180.0 / M_PI);
        largePrint->Print(std::string(buffer), color, 10, 120, fontSize);

        std::snprintf(buffer, sizeof(buffer), "Surge(u): %.3f m/s", bodyLinVel.x());
        largePrint->Print(std::string(buffer), color, 250, 10, fontSize);

        std::snprintf(buffer, sizeof(buffer), "Sway(v):  %.3f m/s", bodyLinVel.y());
        largePrint->Print(std::string(buffer), color, 250, 30, fontSize);

        std::snprintf(buffer, sizeof(buffer), "Heave(w): %.3f m/s", bodyLinVel.z());
        largePrint->Print(std::string(buffer), color, 250, 50, fontSize);
    }
}


