#ifndef MY_AUV_APP_H  // 修改点1
#define MY_AUV_APP_H

#include <core/GraphicalSimulationApp.h>
#include <graphics/OpenGLPrinter.h>
#include "MyAUVManager.h"

class MyAUVApp : public sf::GraphicalSimulationApp
{
public:
    MyAUVApp(std::string dataDirPath, sf::RenderSettings s, sf::HelperSettings h, MyAUVManager* sim);
    
    void DoHUD();
    void InitializeGUI();
    
private:
    sf::OpenGLPrinter* largePrint;
};

#endif