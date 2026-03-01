
#include "MyAUVManager.h"
#include "MyAUVApp.h"
#include <core/FeatherstoneRobot.h>
#include <entities/statics/Plane.h>
#include <entities/statics/Obstacle.h>
#include <entities/solids/Polyhedron.h>
#include <entities/solids/Box.h>
#include <entities/solids/Sphere.h>
#include <entities/solids/Torus.h>
#include <entities/solids/Cylinder.h>
#include <entities/solids/Compound.h>
#include <entities/solids/Wing.h>
#include <graphics/OpenGLPointLight.h>
#include <graphics/OpenGLSpotLight.h>
#include <graphics/OpenGLTrackball.h>
#include <utils/SystemUtil.hpp>
#include <entities/statics/Obstacle.h>
#include <entities/statics/Terrain.h>
#include <actuators/Thruster.h>
#include <actuators/Servo.h>
#include <actuators/VariableBuoyancy.h>
#include <sensors/scalar/Pressure.h>
#include <sensors/scalar/Odometry.h>
#include <sensors/scalar/DVL.h>
#include <sensors/scalar/Compass.h>
#include <sensors/scalar/IMU.h>
#include <sensors/scalar/GPS.h>
#include <sensors/Contact.h>
#include <sensors/vision/ColorCamera.h>
#include <sensors/vision/DepthCamera.h>
#include <sensors/vision/Multibeam2.h>
#include <sensors/vision/FLS.h>
#include <sensors/vision/SSS.h>
#include <sensors/vision/MSIS.h>
#include <comms/AcousticModem.h>
#include <sensors/Sample.h>
#include <actuators/Light.h>
#include <sensors/scalar/RotaryEncoder.h>
#include <sensors/scalar/Accelerometer.h>
#include <entities/FeatherstoneEntity.h>
#include <entities/forcefields/Trigger.h>
#include <entities/forcefields/Pipe.h>
#include <entities/forcefields/Jet.h>
#include <entities/forcefields/Uniform.h>
#include <entities/AnimatedEntity.h>
#include <sensors/scalar/Profiler.h>
#include <sensors/scalar/Multibeam.h>
#include <utils/UnitSystem.h>
#include <core/ScenarioParser.h>
#include <core/NED.h>
#include <cmath>
#include <actuators/Thruster.h> 

MyAUVManager::MyAUVManager(sf::Scalar stepsPerSecond)
: SimulationManager(stepsPerSecond, sf::SolverType::SOLVER_SI, sf::CollisionFilteringType::COLLISION_EXCLUSIVE)
{
    currentTime = 0.0;
    dataCollector = new DataCollector();
    bool success = dataCollector->init("../logs", "GeneralMission_"); 
    // 初始化运动控制器
    motionController = new MotionController();

    // 设置初始PID 策略，设定深度 15m，前进速度 0.5m/s
    //auto pidStrategy = std::make_shared<PIDStationKeepingStrategy>(15.0, 0.5);
    //motionController->setStrategy(pidStrategy);

    //auto dataStrategy = std::make_shared<DataCollectionStrategy>();
    //motionController->setStrategy(dataStrategy);

    // auto maxThrustStrategy = std::make_shared<MaxThrustStrategy>();
    // motionController->setStrategy(maxThrustStrategy);

    auto hybridStrategy = std::make_shared<HybridDataCollectionStrategy>(1.0 / stepsPerSecond);
    motionController->setStrategy(hybridStrategy);
    std::cout << "[MyAUVManager] Hybrid Data Collection Strategy Activated." << std::endl;
    std::cout << "[MyAUVManager] Phase 1: Deterministic Tests -> Phase 2: Stochastic Exploration" << std::endl;
}

//析构函数：手动释放之前申请的内存，防止内存泄漏
MyAUVManager::~MyAUVManager() {
    if(dataCollector) delete dataCollector;
    if(motionController) delete motionController;
}

void MyAUVManager::BuildScenario()
{
    ///////MATERIALS////////
    CreateMaterial("Dummy", sf::UnitSystem::Density(sf::CGS, sf::MKS, 0.9), 0.3);
    CreateMaterial("Fiberglass", sf::UnitSystem::Density(sf::CGS, sf::MKS, 1.5), 0.9);
    CreateMaterial("Rock", sf::UnitSystem::Density(sf::CGS, sf::MKS, 3.0), 0.6);
    SetMaterialsInteraction("Dummy", "Dummy", 0.5, 0.2);
    SetMaterialsInteraction("Fiberglass", "Fiberglass", 0.5, 0.2);
    SetMaterialsInteraction("Rock", "Rock", 0.9, 0.7);
    SetMaterialsInteraction("Fiberglass", "Dummy", 0.5, 0.2);
    SetMaterialsInteraction("Rock", "Dummy", 0.6, 0.4);
    SetMaterialsInteraction("Rock", "Fiberglass", 0.6, 0.4);
    
    ///////LOOKS///////////
    CreateLook("yellow", sf::Color::RGB(1.f, 0.9f, 0.f), 0.3f, 0.f);
    CreateLook("grey", sf::Color::RGB(0.3f, 0.3f, 0.3f), 0.4f, 0.5f);
    CreateLook("seabed", sf::Color::RGB(0.7f, 0.7f, 0.5f), 0.9f, 0.f, 0.f, "", sf::GetDataPath() + "sand_normal.png");
    CreateLook("propeller", sf::Color::RGB(1.f, 1.f, 1.f), 0.3f, 0.f, 0.f, sf::GetDataPath() + "propeller_tex.png");
    CreateLook("black", sf::Color::RGB(0.1f, 0.1f, 0.1f), 0.4f, 0.5f);

    ////////OBJECTS    
    //Create environment
    EnableOcean(0.0);
    getOcean()->setWaterType(0.12);
    getOcean()->AddVelocityField(new sf::Jet(sf::Vector3(0,0,1.0), sf::VY(), 0.3, 5.0));
    //getOcean()->AddVelocityField(new sf::Uniform(sf::Vector3(1.0,0.0,0.0)));
    //getOcean()->EnableCurrents();  //洋流和喷射流
    getAtmosphere()->SetSunPosition(0.0, 60.0);
    getNED()->Init(41.77737, 3.03376, 0.0);
    
    sf::Terrain* seabed = new sf::Terrain("Seabed", sf::GetDataPath() + "terrain.png", 1.0, 1.0, 5.0, "Rock", "seabed", 5.f);
    AddStaticEntity(seabed, sf::Transform(sf::IQ(), sf::Vector3(0,0,50.0)));
    //sf::Obstacle* cyl = new sf::Obstacle("Cyl", 0.5, 5.0, sf::I4(), "Fiberglass", "seabed");
    //AddStaticEntity(cyl, sf::Transform(sf::Quaternion(0,M_PI_2,0), sf::Vector3(6.0,2.0,5.0)));
	sf::Light* spot = new sf::Light("Spot", 0.02, 50.0, sf::Color::BlackBody(5000.0), 100.0);
	spot->AttachToWorld(sf::Transform(sf::Quaternion(0,0,M_PI/3.0), sf::Vector3(0.0,0.0,1.0)));
	AddActuator(spot);
    
    sf::Light* omni = new sf::Light("Omni", 0.02, sf::Color::BlackBody(5000.0), 10000.0);
	omni->AttachToWorld(sf::Transform(sf::Quaternion(0,0,M_PI/3.0), sf::Vector3(2.0,2.0,0.5)));
	AddActuator(omni);

    //Create underwater vehicle body
    double inX = 0.33;  // 内圈前后距离
    double inY = 0.137;  // 内圈左右距离
    double outX = 0.45; 
    double outY = 0.60; 
    double zLevel = -0.0;     
    double zLevelVert = -0.0; 
    // 垂直旋转四元数 
    sf::Quaternion vertRot(0, -M_PI_2, 0); 
    // 视觉修正
    sf::Quaternion vertRotVisualFix(0, -M_PI_2, M_PI);

    //Externals
    sf::BodyPhysicsSettings phy;
    phy.mode = sf::BodyPhysicsMode::SUBMERGED;
    phy.collisions = true;
    
    phy.buoyancy = false;
    sf::Box* hullB = new sf::Box("RootDummy", phy, sf::Vector3(0.001, 0.001, 0.001), sf::I4(), "Dummy", "black");    //Root Link
    sf::Polyhedron* hullP = new sf::Polyhedron("HullPort", phy, sf::GetDataPath() + "hull_hydro.obj", sf::Scalar(1), sf::I4(), "Fiberglass", "yellow", sf::Scalar(0.003));
    sf::Polyhedron* hullS = new sf::Polyhedron("HullStarboard", phy, sf::GetDataPath() + "hull_hydro.obj", sf::Scalar(1), sf::I4(), "Fiberglass", "yellow", sf::Scalar(0.003));
    sf::Polyhedron* vBarStern = new sf::Polyhedron("VBarStern", phy, sf::GetDataPath() + "vbar_hydro.obj", sf::Scalar(1), sf::I4(), "Dummy", "grey", sf::Scalar(0.003));
    sf::Polyhedron* vBarBow = new sf::Polyhedron("VBarBow", phy, sf::GetDataPath() + "vbar_hydro.obj", sf::Scalar(1), sf::I4(), "Dummy", "grey", sf::Scalar(0.003));
    this->setGravity(0.0);
    phy.buoyancy = false;
    std::vector<sf::Polyhedron*> vertDucts;
    for(int i=0; i<8; i++) {
        // 创建垂直导管
        vertDucts.push_back(new sf::Polyhedron("DuctVert" + std::to_string(i), phy, sf::GetDataPath() + "duct_hydro.obj", sf::Scalar(1), sf::I4(), "Dummy", "black"));
    }
    //Internals
    sf::Cylinder* batteryCyl = new sf::Cylinder("BatteryCylinder", phy, 0.13, 0.6, sf::I4(), "Dummy", "grey");
    batteryCyl->ScalePhysicalPropertiesToArbitraryMass(sf::Scalar(92.5));
    sf::Cylinder* portCyl = new sf::Cylinder("PortCylinder", phy, 0.13, 1.0, sf::I4(), "Dummy", "grey");
    portCyl->ScalePhysicalPropertiesToArbitraryMass(sf::Scalar(20));
    sf::Cylinder* starboardCyl = new sf::Cylinder("StarboardCylinder", phy, 0.13, 1.0, sf::I4(), "Dummy", "grey");
    starboardCyl->ScalePhysicalPropertiesToArbitraryMass(sf::Scalar(20));
    
    //Build whole body
    sf::Compound* vehicle = new sf::Compound("Vehicle", phy, hullB, sf::I4());
    vehicle->AddExternalPart(hullP, sf::Transform(sf::IQ(), sf::Vector3(0,-0.35,-0.0)));
    vehicle->AddExternalPart(hullS, sf::Transform(sf::IQ(), sf::Vector3(0,0.35,-0.0)));
    vehicle->AddExternalPart(vertDucts[0], sf::Transform(vertRotVisualFix, sf::Vector3(outX, -outY, zLevelVert)));
    vehicle->AddExternalPart(vertDucts[1], sf::Transform(vertRot, sf::Vector3(outX, outY, zLevelVert)));
    vehicle->AddExternalPart(vertDucts[2], sf::Transform(vertRotVisualFix, sf::Vector3(-outX, -outY, zLevelVert)));
    vehicle->AddExternalPart(vertDucts[3], sf::Transform(vertRot, sf::Vector3(-outX, outY, zLevelVert)));
    vehicle->AddInternalPart(batteryCyl, sf::Transform(sf::Quaternion(0,M_PI_2,0), sf::Vector3(0,0,0)));
    vehicle->AddInternalPart(portCyl, sf::Transform(sf::Quaternion(0,M_PI_2,0), sf::Vector3(0.0,-0.35,-0.0)));
    vehicle->AddInternalPart(starboardCyl, sf::Transform(sf::Quaternion(0,M_PI_2,0), sf::Vector3(0.0,0.35,-0.0)));
    
    vehicle->setDisplayInternalParts(false);
    
    //Create thrusters
    std::array<std::string, 8> thrusterNames = { 
    "HorzFrontLeft", "HorzFrontRight", "HorzRearLeft",  "HorzRearRight",  "VertFrontLeft",  "VertFrontRight", "VertRearLeft",  "VertRearRight"   
    };
    std::array<sf::Thruster*, 8> thrusters;
    std::shared_ptr<sf::Polyhedron> propeller = std::make_shared<sf::Polyhedron>("Propeller", phy, sf::GetDataPath() + "propeller.obj", sf::Scalar(1), sf::I4(), "Dummy", "propeller");
    for(size_t i=0; i<thrusterNames.size(); ++i)
    {
        std::shared_ptr<sf::MechanicalPI> rotorDynamics;
        rotorDynamics = std::make_shared<sf::MechanicalPI>(1.0, 10.0, 5.0, 5.0);
        std::shared_ptr<sf::FDThrust> thrustModel;
        thrustModel = std::make_shared<sf::FDThrust>(0.20, 0.48, 0.48, 0.05, true, getOcean()->getLiquid().density); 
        thrusters[i] = new sf::Thruster(thrusterNames[i], propeller, rotorDynamics, thrustModel, 0.18, true, 155.0, false, true);
    }
       
    //Create sensors
    sf::Odometry* odom = new sf::Odometry("Odom");
    sf::Pressure* press = new sf::Pressure("Pressure");
    press->setNoise(1.0);
    sf::DVL* dvl = new sf::DVL("DVL", 30.0, false);
    dvl->setNoise(0.0, 0.02, 0.05, 0.0, 0.02);
    sf::IMU* imu = new sf::IMU("IMU");
    imu->setNoise(sf::V0(), sf::Vector3(0.05, 0.05, 0.1), 0.0, sf::Vector3(0.01, 0.01, 0.02));
    sf::Compass* fog = new sf::Compass("FOG");
    fog->setNoise(0.01);
    sf::GPS* gps = new sf::GPS("GPS");
    gps->setNoise(0.5);
    sf::FLS* fls = new sf::FLS("FLS", 256, 500, 150.0, 30.0, 1.0, 20.0, sf::ColorMap::GREEN_BLUE);
    fls->setNoise(0.05, 0.05);
    fls->setDisplayOnScreen(false, 800, 250, 0.4f);
    sf::SSS* sss = new sf::SSS("SSS", 800, 400, 70.0, 1.5, 50.0, 1.0, 100.0, sf::ColorMap::GREEN_BLUE);
    sss->setDisplayOnScreen(false, 710, 5, 0.6f);
    sf::MSIS* msis = new sf::MSIS("MSIS", 1.5, 500, 2.0, 30.0, -50, 50, 1.0, 100.0, sf::ColorMap::GREEN_BLUE);
    msis->setDisplayOnScreen(false, 880, 455, 0.6f);
    //sf::ColorCamera* cam = new sf::ColorCamera("Cam", 300, 200, 60.0, 10.0);
    //cam->setDisplayOnScreen(true);
    //sf::ColorCamera* cam2 = new sf::ColorCamera("Cam", 300, 200, 60.0);
    
    //Create AUV
    sf::Robot* auv = new sf::FeatherstoneRobot("GIRONA500", false);
    
    // 创建一个空的列表 robot对象
    std::vector<sf::SolidEntity*> emptyList;
    auv->DefineLinks(vehicle, emptyList);
    auv->BuildKinematicStructure();

    auv->AddLinkActuator(thrusters[0], "Vehicle", sf::Transform(sf::Quaternion(0, 0, 0), sf::Vector3(inX, -inY, zLevel)));
    auv->AddLinkActuator(thrusters[1], "Vehicle", sf::Transform(sf::Quaternion(0, 0, 0), sf::Vector3(inX, inY, zLevel)));
    auv->AddLinkActuator(thrusters[2], "Vehicle", sf::Transform(sf::Quaternion(0, 0, 0), sf::Vector3(-inX, -inY, zLevel)));
    auv->AddLinkActuator(thrusters[3], "Vehicle", sf::Transform(sf::Quaternion(0, 0, 0), sf::Vector3(-inX, inY, zLevel)));
    auv->AddLinkActuator(thrusters[4], "Vehicle", sf::Transform(vertRot, sf::Vector3(outX, -outY, zLevelVert)));
    auv->AddLinkActuator(thrusters[5], "Vehicle", sf::Transform(vertRot, sf::Vector3(outX, outY, zLevelVert)));
    auv->AddLinkActuator(thrusters[6], "Vehicle", sf::Transform(vertRot, sf::Vector3(-outX, -outY, zLevelVert)));
    auv->AddLinkActuator(thrusters[7], "Vehicle", sf::Transform(vertRot, sf::Vector3(-outX, outY, zLevelVert)));

    //Sensors
    auv->AddLinkSensor(odom, "Vehicle", sf::Transform(sf::IQ(), sf::Vector3(0,0,0)));
    auv->AddLinkSensor(press, "Vehicle", sf::Transform(sf::IQ(), sf::Vector3(0.6,0,-0.7)));
    auv->AddLinkSensor(dvl, "Vehicle", sf::Transform(sf::Quaternion(-M_PI_4,0,M_PI), sf::Vector3(-0.5,0,0.1)));
    auv->AddLinkSensor(imu, "Vehicle", sf::Transform(sf::IQ(), sf::Vector3(0,0,-0.7)));
    auv->AddLinkSensor(fog, "Vehicle", sf::Transform(sf::IQ(), sf::Vector3(0.3,0,-0.7)));
    auv->AddLinkSensor(gps, "Vehicle", sf::Transform(sf::IQ(), sf::Vector3(-0.5,0,-0.9)));
    auv->AddVisionSensor(fls, "Vehicle", sf::Transform(sf::Quaternion(1.57, 0.0, 0.8), sf::Vector3(0.0,0.0,1.0)));
    auv->AddVisionSensor(sss, "Vehicle", sf::Transform(sf::Quaternion(1.57, 0.0, 0.0), sf::Vector3(0.0,0.0,0.0)));
    auv->AddVisionSensor(msis, "Vehicle", sf::Transform(sf::Quaternion(0.0, 0.0, 1.57), sf::Vector3(0.0,0.0,1.0)));
    //auv->AddVisionSensor(cam, "Vehicle", sf::Transform(sf::Quaternion(1.57, 0.0, 1.57), sf::Vector3(0.0,0.0,1.0)));
    //auv->AddVisionSensor(cam2, "Vehicle", sf::Transform(sf::Quaternion(1.57, 0.0, 1.57), sf::Vector3(0.0,0.0,2.0)));
    AddRobot(auv, sf::Transform(sf::Quaternion(0,0,0), sf::Vector3(0.0,0.0,20.0)));
} 

AUVState MyAUVManager::getAUVState()
{
    AUVState state = {0};
    sf::Robot* robot = this->getRobot("GIRONA500");
    if (!robot) return state;

    sf::SolidEntity* vehicle = robot->getLink("Vehicle");
    if (!vehicle) return state;

    // SolidEntity 获取全局坐标和机器位置
    sf::Transform transform = vehicle->getGTransform(); 
    const sf::Vector3& pos = transform.getOrigin();
    
    state.x = pos.x();state.y = pos.y();state.z = pos.z(); 

    btScalar y, p, r;
    transform.getBasis().getEulerYPR(y, p, r);
    state.yaw = y;state.pitch = p;state.roll = r;

    sf::Vector3 worldLinVel = vehicle->getLinearVelocity();
    sf::Vector3 worldAngVel = vehicle->getAngularVelocity();

    // 坐标系转换
    sf::Vector3 bodyLinVel = transform.getBasis().transpose() * worldLinVel;
    sf::Vector3 bodyAngVel = transform.getBasis().transpose() * worldAngVel;

    state.u = bodyLinVel.x(); state.v = bodyLinVel.y(); state.w = bodyLinVel.z(); 
    state.p = bodyAngVel.x();state.q = bodyAngVel.y();state.r = bodyAngVel.z(); 
    return state;
}

void MyAUVManager::SimulationStepCompleted(sf::Scalar timeStep)
{
    currentTime += timeStep;
    AUVState state = getAUVState();
    sf::Robot* robot = getRobot("GIRONA500");

    if(motionController && robot) {
        motionController->update(robot, state, timeStep);
    }

    state.motors.clear();
    
    if (robot) {
        for(const std::string& name : this->thrusterNames) {
            sf::Actuator* act = robot->getActuator(name);
            sf::Thruster* th = dynamic_cast<sf::Thruster*>(act);
            
            if(th) {
                state.motors.push_back(th->getSetpoint());
            } else {
                state.motors.push_back(0.0);
            }
        }
    }

    if(dataCollector) {
        dataCollector->log(currentTime, state);
    }
}