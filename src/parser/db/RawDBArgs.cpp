#include "global.h"
#include "RawDBArgs.h"

namespace db {
void RawDBArgs::reset() {
    Format = "";
    Dataset = "";
    DesignName = "";
    BookshelfVariety = "";
    BookshelfAux = "";
    BookshelfPl = "";
    DefFile = "";
    LefFile = "";
    LefCell = "";
    LefTech = "";
    LefFiles.clear();
    LibFiles.clear();
    Constraints = "";
    Verilog = "";
    ICCAD2022InputFile = "";
    OutputFile = "";
    
    liteMode = false;
    random_place = true;
}

RawDBArgs rawDBArgs;

}  //   namespace db
