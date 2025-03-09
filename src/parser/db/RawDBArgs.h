#pragma once
#include "global.h"

namespace db {

class RawDBArgs {
public:
    int numThreads = 1;

    // 1. custom definition of gcell
    int customGcellTrackX = 30;
    int customGcellTrackY = 30;

    // 2. DB
    bool EdgeSpacing = true;
    bool EnableFence = true;
    bool EnablePG = true;
    bool EnableIOPin = true;
    bool liteMode = false;
    bool random_place = true;

    // 3. IO
    std::string Format = "";
    std::string Dataset = "";
    std::string DesignName = "";

    std::string BookshelfVariety = "";
    std::string BookshelfAux = "";
    std::string BookshelfPl = "";

    std::string DefFile = "";
    std::string LefFile = "";
    std::string LefCell = "";
    std::string LefTech = "";
    std::vector<std::string> LefFiles;
    std::vector<std::string> LibFiles;

    std::string Constraints = "";
    std::string Verilog = "";
    std::string Size = "";

    std::string ICCAD2022InputFile = "";

    std::string OutputFile = "";

    void reset();
};

extern RawDBArgs rawDBArgs;
}  //   namespace db
