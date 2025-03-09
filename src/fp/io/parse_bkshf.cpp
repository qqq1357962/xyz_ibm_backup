#include "global.h"
#include "fp/db/Database.h"

using namespace fp;


// class BookshelfData {
// public:
//     int nCells;
//     int nNets;
//     int nMacros;
//     int nTerminals;
    
//     std::string format;
//     unordered_map<string, int> cellMap;

//     vector<string> cellName;
//     vector<int> cellSize;
//     vector<char> cellType;
//     vector<int> cellX;
//     vector<int> cellY;
//     vector<string> netName;
//     vector<int> rectNums;
//     vector<vector<int>> netCells;
//     vector<vector<int>> rectBlocks;

//     void clearData() {
//         cellMap.clear();
//         cellName.clear();
//         netName.clear();
//         netCells.clear();
//     }
// };

//---------------------------------------------------------------------

//bool Database::read(const std::string& blockFile, const std::string& netFile, const std::string& plFile) {
//bool Database::read(dp::DetailedPlaceData& db) {
    
    /*
    std::vector<int> macros;
    for (int i = 0; i < db.num_movable_nodes; ++i) {
        if (db.is_dummy_fixed(i)) {
            if(db.node_weight[i]==0)
            {
                continue;
            }
            // in some extreme case, some macros with 0 area should be ignored
            float area = db.node_size_x[i] * db.node_size_y[i];
            if (area > 0) {
                macros.push_back(i);
            }
        }
    }
    */
    // readBSCell(blockFile);
    // readBSNets(netFile);
    // readBSPl(plFile);

    // this->nCells = bsData.nCells;
    // this->nMacros = bsData.nMacros;
    // this->nNets = bsData.nNets;

    // cells
    // int t_id = 0;
    // int m_id = 0;
    /*
    for (int i = 0; i < bsData.nCells; i++) {
        int cellID = i;
        char cellType = bsData.cellType[cellID];

        if (cellType == '0') {
            Cell* cell = this->addCell(bsData.cellName[cellID]);
            cell->id = t_id++;

            int pos_x = bsData.cellX[cellID];
            int pos_y = bsData.cellY[cellID];
            cell->pos->set_pos(pos_x, pos_y);
        }
        else if (cellType == '1') {
            Macro* macro = this->addMacro(bsData.cellName[cellID]);
            macro->id = m_id++;
            macro->rectNum = bsData.rectNums[cellID];
            macro->rects = bsData.rectBlocks[cellID];
            macro->init();
        }
    }
    */
    // nets
    /*
    for (unsigned i = 0; i != bsData.nNets; ++i) {
        int netID = i;
        Net* net = this->addNet(bsData.netName[netID]);
        net->id = netID;

        for (unsigned j = 0; j != bsData.netCells[netID].size(); ++j) {
            int cellID = bsData.netCells[netID][j];
            char cellType = bsData.cellType[cellID];

            if (cellType == '0') {
                Cell* cell = this->getCell(bsData.cellName[cellID]);
                net->cell_list.push_back(cell);
            }
            else if (cellType == '1') {
                Macro* macro = this->getMacro(bsData.cellName[cellID]);
                net->macro_list.push_back(macro);
                net->macro_list_by_id.push_back(macro->id);
            }
        }
        this->nets_by_id.push_back(net->macro_list_by_id);
        net->init_bx();
        this->net_terminals.push_back(fp::pin(net->min_x_, net->min_y_,
                                            net->max_x_, net->max_y_));
    }
    */

    //return true;
//} //END MODULE

//---------------------------------------------------------------------