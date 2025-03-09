#include "Database.h"


using namespace fp;

/***** Database *****/
Database::Database() {
    _buffer = new char[_bufferCapacity];
}

Database::~Database() {
    delete[] _buffer;
    _buffer = nullptr;
    // for regions.push_back(new Region("default"));
} //END MODULE

//---------------------------------------------------------------------

void Database::init(double ratio) {
    // area
    // total_area = 0;
    // for (Macro* macro : macros) {
    //     int area = macro->height() * macro->width();
    //     total_area += area;
    // }

    //outline_height = int(sqrt(total_area * (1 + ratio)));
    //outline_width = int(sqrt(total_area * (1 + ratio)));

    logger.info("outline: %d x %d",outline_height,outline_width);
} //END MODULE

//---------------------------------------------------------------------

Cell* Database::addCell(const string& name) {
    Cell* cell = getCell(name);
    if (cell) {
        printlog(LOG_INFO, "cell re-defined: %s", name.c_str());
        return cell;
    }
    cell = new Cell(name);
    name_cells.emplace(name, cell);
    cells.push_back(cell);
    return cell;
} //END MODULE

//---------------------------------------------------------------------

Macro* Database::addMacro(const string& name) {
    Macro* macro = getMacro(name);
    if (macro) {
        printlog(LOG_INFO, "macro re-defined: %s", name.c_str());
        return macro;
    }
    macro = new Macro(name);
    name_macros.emplace(name, macro);
    macros.push_back(macro);
    return macro;
} //END MODULE

void Database::printMacros() {
    for (Macro* macro : macros) {
        std::cout << "print macro: " << macro->name() << ' ' << macro->width(0) << ' ' << macro->height(0) << endl;
    }
    return;
} //END MODULE

//---------------------------------------------------------------------

Net* Database::addNet(const string& name) {
    // Net* net = getNet(name);
    // if (net) {
    //     printlog(LOG_INFO, "Net re-defined: %s", name.c_str());
    //     return net;
    // }
    Net* net = new Net(name);
    name_nets[name] = net;
    nets.push_back(net);
    return net;
} //END MODULE

//---------------------------------------------------------------------

Cell* Database::getCell(const string& name) {
    unordered_map<string, Cell*>::iterator mi = name_cells.find(name);
    if (mi == name_cells.end()) {
        return nullptr;
    }
    return mi->second;
} //END MODULE

//---------------------------------------------------------------------

Macro* Database::getMacro(const string& name) {
    unordered_map<string, Macro*>::iterator mi = name_macros.find(name);
    if (mi == name_macros.end()) {
        return nullptr;
    }
    return mi->second;
} //END MODULE

Macro* Database::getMacroByID(const int macro_id) {
    return macros[macro_id];
} //END MODULE

//---------------------------------------------------------------------

Net* Database::getNet(const string& name) {
    unordered_map<string, Net*>::iterator mi = name_nets.find(name);
    if (mi == name_nets.end()) {
        return nullptr;
    }
    return mi->second;
} //END MODULE

//---------------------------------------------------------------------
/*
Output the design format of previous version
*/

void Database::recall_design(){
    ofstream outfile;
    outfile.open("/data/ssd/bqfu/hw/floorplanner/fixed-outline_floorplanning/testcases/"+designName+".blocks", ios::out);

    outfile << "Outline: " << outline_width << ' ' << outline_height << "\n";

    outfile << "NumBlocks: " << nMacros << endl;
    outfile << "NumTerminals: " << nMacros << endl;

    // for (Macro* macro : macros) {
    //     outfile << macro->name() << ' ' << macro->width() << ' ' << macro->height() << endl;
    // }

    for (Cell* cell : cells) {
        outfile << cell->name() << " terminal " << cell->pos->x() << ' ' << cell->pos->y() << endl;
    }

} //END MODULE

//---------------------------------------------------------------------