#pragma once

#include "global.h"
#include "RawDBArgs.h"

namespace db {
class Rectangle;
class Geometry;
class GeoMap;
class Cell;
class CellType;
class Pin;
class PinType;
class IOPin;
class CellPin;
class Net;
class Row;
class RowSegment;
class Track;
class Layer;
class Bonding;
class Via;
class ViaType;
class Region;
class NDR;
class SNet;
class Site;
class PowerNet;
class EdgeTypes;
class GCell;
}  // namespace db

#include "Cell.h"
#include "DesignRule.h"
#include "Geometry.h"
#include "Layer.h"
#include "SiteMap.h"
#include "Net.h"
#include "Pin.h"
#include "Region.h"
#include "GCellGrid.h"
#include "Row.h"
#include "Site.h"
#include "SNet.h"
#include "TargetDensity.h"
#include "Via.h"
#include "Bonding.h"

namespace db {

#define CLEAR_POINTER_LIST(list) \
    {                            \
        for (auto obj : list) {  \
            delete obj;          \
        }                        \
        list.clear();            \
    }

#define CLEAR_POINTER_MAP(map) \
    {                          \
        for (auto p : map) {   \
            delete p.second;   \
        }                      \
        map.clear();           \
    }

class Database {
public:
    enum IssueType {
        E_ROW_EXCEED_DIE,
        E_OVERLAP_ROWS,
        W_NON_UNIFORM_SITE_WIDTH,
        W_NON_HORIZONTAL_ROW,
        E_MULTIPLE_NET_DRIVING_PIN,
        E_NO_NET_DRIVING_PIN
    };

    unordered_map<string, CellType*> name_celltypes;
    unordered_map<string, Cell*> name_cells;
    unordered_map<string, Net*> name_nets;
    unordered_map<string, IOPin*> name_iopins;
    unordered_map<string, ViaType*> name_viatypes;

    bool verilog_only = false;
    bool lef_read = false;
    vector<Layer> layers;
    vector<Site> sites;
    vector<ViaType*> viatypes;
    vector<CellType*> celltypes;

    vector<Cell*> cells;
    vector<IOPin*> iopins;
    vector<Net*> nets;
    vector<Row*> rows;
    vector<Region*> regions;
    map<string, NDR*> ndrs;
    vector<SNet*> snets;
    vector<Track*> tracks;

    vector<Geometry> routeBlockages;
    vector<Rectangle> placeBlockages;

    PowerNet powerNet;

private:
    static const size_t _bufferCapacity = 128 * 1024;
    size_t _bufferSize = 0;
    char* _buffer = nullptr;

public:
    unsigned siteW = 0;
    int siteH = 0;
    unsigned nSitesX = 0;
    unsigned nSitesY = 0;
    int site_width_keep = 0;
    int site_height_keep = 0;

    SiteMap siteMap;
    TDBins tdBins;  // local target density
    GRGrid grGrid;  // global routing grid

    EdgeTypes edgetypes;

    int dieLX, dieLY, dieHX, dieHY;
    int coreLX, coreLY, coreHX, coreHY;

    double bot_Max_util = 0.8;
    double top_Max_util = 0.8;

    double maxDensity = 0;
    double maxDisp = 0;

    int LefConvertFactor;
    double DBU_Micron;
    double version;
    string designName;

    vector<IssueType> dbIssues;

public:
    Database();
    ~Database();
    void clear();
    void clearTechnology();
    inline void clearLibrary() { CLEAR_POINTER_LIST(celltypes); }
    void clearDesign();

    Layer& addLayer(const string& name, const char type = 'x');
    Site& addSite(const string& name, const string& siteClassName, const int w, const int h);
    ViaType* addViaType(const string& name, bool isDef);
    inline ViaType* addViaType(const string& name) { return addViaType(name, false); }
    CellType* addCellType(const string& name, unsigned libcell);
    void reserveCells(const size_t n) { cells.reserve(n); }
    Cell* addCell(const string& name, CellType* type = nullptr);
    IOPin* addIOPin(const string& name = "", const string& netName = "", const char direction = 'x');
    void reserveNets(const size_t n) { nets.reserve(n); }
    Net* addNet(const string& name = "", const NDR* ndr = nullptr);
    Row* addRow(const string& name,
                const string& macro,
                const int x,
                const int y,
                const unsigned xNum = 0,
                const unsigned yNum = 0,
                const bool flip = false,
                const unsigned xStep = 0,
                const unsigned yStep = 0);
    Track* addTrack(char direction, double start, double num, double step);
    Region* addRegion(const string& name = "", const char type = 'x');
    NDR* addNDR(const string& name, const bool hardSpacing);
    void reserveSNets(const size_t n) { snets.reserve(n); }
    SNet* addSNet(const string& name);

    Layer* getRLayer(const int index);
    const Layer* getCLayer(const unsigned index) const;
    Layer* getLayer(const string& name);
    CellType* getCellType(const string& name);
    Cell* getCell(const string& name);
    Net* getNet(const string& name);
    Region* getRegion(const string& name);
    Region* getRegion(const unsigned char id);
    NDR* getNDR(const string& name) const;
    IOPin* getIOPin(const string& name) const;
    ViaType* getViaType(const string& name) const;
    SNet* getSNet(const string& name);

    unsigned getNumRLayers() const;
    unsigned getNumCLayers() const;
    inline unsigned getNumLayers() const { return layers.size(); }
    inline unsigned getNumCells() const { return cells.size(); }
    inline unsigned getNumNets() const { return nets.size(); }
    inline unsigned getNumRegions() const { return regions.size(); }
    inline unsigned getNumIOPins() const { return iopins.size(); }
    inline unsigned getNumCellTypes() const { return celltypes.size(); }

    inline int getCellTypeSpace(const CellType* L, const CellType* R) const {
        return edgetypes.getEdgeSpace(L->edgetypeR, R->edgetypeL);
    }
    inline int getCellTypeSpace(const Cell* L, const Cell* R) const { return getCellTypeSpace(L->ctype(), R->ctype()); }
    int getContainedSites(
        const int lx, const int ly, const int hx, const int hy, int& slx, int& sly, int& shx, int& shy) const;
    int getOverlappedSites(
        const int lx, const int ly, const int hx, const int hy, int& slx, int& sly, int& shx, int& shy) const;

    long long getHPWL();
    long long getCellArea(Region* region = nullptr) const;
    long long getFreeArea(Region* region = nullptr) const;

    bool placed();
    bool globalRouted();
    bool detailedRouted();

    void errorCheck(bool autoFix = true);
    void checkPlaceError();
    void checkDRCError();

    void load();
    void setup();  // call after read
    void reset();
    void save(const std::string& given_prefix);

    /* defined in io/file_lefdef_db.cpp */
public:
    bool readLEF(const std::string& file);
    bool readDEF(const std::string& file);
    bool readDEFPG(const string& file);
    bool writeDEF(const std::string& file);
    bool writeICCAD2017(const string& inputDef, const string& outputDef);
    bool writeICCAD2017(const string& outputDef);
    bool writeDEF2ICCAD2022(const string& file);
    bool writeComponents(ofstream& ofs);
    bool writeComponents(ofstream& ofs, const std::vector<int> node_selected);
    bool writeNets(ofstream& ofs, const std::vector<int> node_selected);
    bool writePins(ofstream& ofs, const std::vector<int> node_selected);
    bool write_Openroad(const string& inputDef, const string& outputDef, const std::vector<int> node_selected);
    bool write_openroad_partition(const string& inputDef, const string& outputDef, const std::vector<int> node_selected);
    bool writeBuffer(ofstream& ofs, const string& line);
    void writeBufferFlush(ofstream& ofs);

    bool readBSAux(const std::string& auxFile, const std::string& plFile);
    bool readBSNodes(const std::string& file);
    bool readBSNets(const std::string& file);
    bool readBSScl(const std::string& file);
    bool readBSRoute(const std::string& file);
    bool readBSShapes(const std::string& file);
    bool readBSWts(const std::string& file);
    bool readBSPl(const std::string& file);
    bool writeBSPl(const std::string& file);

    bool readVerilog(const std::string& file);
    bool readLiberty(const std::string& file);

    bool readConstraints(const std::string& file);
    bool readSize(const std::string& file);

private:
    void SetupLayers();
    void SetupCellLibrary();
    void SetupFloorplan();
    void SetupRegions();
    void SetupSiteMap();
    void SetupRows();
    void SetupRowSegments();
    void SetupTargetDensity();
    void SetupGRGrid();

// ICCAD 2022 only
public:
    int numTechlibs;
    // index 0 -> bottom, index 1 -> top
    vector<double> maxUtilM;
    vector<int> rowHeights;
    vector<int> rowStart;
    vector<vector<CellType*>> celltypesM;
    vector<vector<Row*>> rowsM;
    vector<vector<std::pair<int, int>>> netId2cellPinIds;
    vector<Bonding> bondings;

    int bondingSizeX, bondingSizeY, bondingSpacing, bondingCost;
    int numValidBondings;

    // TODO: Multi Tech: only 2
    vector<Cell*> cells_mT;
    unordered_map<string, Cell*> name_cells_mT;
    vector<Net*> nets_mT;
    Cell* addCell_mT(const string& name, CellType* type = nullptr);
    Net* addNet_mT(const string& name = "", const NDR* ndr = nullptr);
    vector<int> numRows;

    bool readICCAD2022(const string& file);
    bool writeICCAD2022(const string& file);
    bool writeOpenroad_vias(const string& file);
    bool resumeICCAD2022(const string& file);

    void placeBonding(int lx, int ly, int netId);
    void removeBonding(int netId);

private:
};

}  // namespace db
