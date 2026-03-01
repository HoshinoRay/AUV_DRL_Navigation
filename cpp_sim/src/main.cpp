#include "MyAUVApp.h"
#include "MyAUVManager.h"
#include <cfenv>

int main(int argc, const char *argv[])
{
    // feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    // feenableexcept(FE_INVALID | FE_OVERFLOW);

    sf::RenderSettings s;
    s.windowW = 1200;
    s.windowH = 900;
    s.aa = sf::RenderQuality::HIGH;
    s.shadows = sf::RenderQuality::HIGH;
    s.ao = sf::RenderQuality::HIGH;
    s.atmosphere = sf::RenderQuality::MEDIUM;
    s.ocean = sf::RenderQuality::HIGH;
    s.ssr = sf::RenderQuality::HIGH;

    sf::HelperSettings h;
    h.showFluidDynamics = false;
    h.showCoordSys = true;
    h.showBulletDebugInfo = false;
    h.showSensors = false;
    h.showActuators = false;
    h.showForces = true;

    MyAUVManager simulationManager(200.0);
    simulationManager.setRealtimeFactor(1.0);
    MyAUVApp app(std::string(DATA_DIR_PATH), s, h, &simulationManager);
    app.Run();

    return 0;
}
