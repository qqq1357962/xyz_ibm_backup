#pragma once
#include "global.h"
//#include "placer/database.h"

namespace fp {
class Cell;
class Macro;
class Net;
class Point;
struct pin;
}  // namespace fp

#include "Cell.h"
#include "Net.h"
#include "Macro.h"
#include "Point.h"


namespace fp {

class Database {
public:
    string designName;

    int nCells;
    int nMacros;
    int nMacros1;
    int nMacros2;
    int nNets;
    int numRotatable;

    vector<int> macro_sequence_rotatable;

    unordered_map<string, Cell*> name_cells;
    unordered_map<string, Macro*> name_macros;
    unordered_map<string, Net*> name_nets;

    vector<Cell*> cells;
    vector<Macro*> macros;
    vector<Net*> nets;

    vector<vector<int>> nets_by_id;
    // vector<pair<fp::Point, fp::Point>> net_terminals;
    vector<pin> net_terminals;

    void init(double ratio);
    // int total_area;
    int outline_height;
    int outline_width;

    void recall_design();

    Cell* addCell(const string& name);
    Macro* addMacro(const string& name);
    Net* addNet(const string& name);
    void printMacros();

// buffer
private:
    static const size_t _bufferCapacity = 128 * 1024;
    size_t _bufferSize = 0;
    char* _buffer = nullptr;

public:
    Database();
    ~Database();
    void clear();

    Cell* getCell(const string& name);
    Macro* getMacro(const string& name);
    Macro* getMacroByID(const int macro_id);
    Net* getNet(const string& name);

    inline unsigned getNumCells() const { return cells.size(); }
    inline unsigned getNumNets() const { return nets.size(); }

public:
    /* defined in io/ */
    bool read(const std::string& cellFile, const std::string& netFile);
    //bool read(dp::DetailedPlaceData& db);
    bool read(const std::string& blockFile, const std::string& netFile, const std::string& plFile);
    bool readCells(const std::string& file);
    bool readNets(const std::string& file);

    bool readBSNets(const std::string& file);
    bool readBSCell(const std::string& file);
    bool readBSPl(const std::string& file);

    void write(ofstream& ofs);

};

} // namespace fp