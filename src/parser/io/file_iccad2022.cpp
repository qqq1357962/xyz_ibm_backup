#include "global.h"
#include "parser/db/Database.h"

using namespace db;

void loadSettings(int num_instances)
{
    int mode=4;
    if(num_instances<100)
    {
        mode=1;
        st::setting.inner_iter=0;
    }
    else if(num_instances<50000)
    {
        mode=2;
    }
    else if(num_instances<250000)
    {
        mode=3;
    }

    st::setting.mode=mode;


    if(mode==2)
    {
        // --version 23 --num_threads 8 --gpu 0 --partitioner gp3d 
        // --num_bin_x 512 --num_bin_y 512 --num_bin_z 10 --dp true 
        // --lg true --pp false --draw_placement true --num_den_layer 3 
        // --force_coeff_2d 0 --round_recursion 1 --omni_int 1 
        // --eval_params 0 --inner_iter 2700 --use_pre_gp 1 --stack_cells 1 
        // --num_bin_3d 53 --cut_net_thres 12 --net_weight_coef 1 
        // --net_weight_offset -0.01 --stop_overflow_3d 0.11 --quad_penalty 1 
        // --wa_coeff 2 --density_weight 16e-4 --density_weight_coef 1.05 
        // --magic_hpwl 35000 --sideline 400 --fp true --min_gp_step 1700 
        // --gp_padding 0.07 --shrink_size 1.0 --is_fp_permit_change_cross_chip true 
        // --skip_gp3d true
        // st::setting.

    }else if(mode==3)
    {

    }else{

    }
}


bool Database::readICCAD2022(const std::string& file) {
    printlog(LOG_INFO, "start parsing dataset file: %s", file.c_str());
    std::ifstream infile(file);
    if (infile.is_open() == false) {
        printlog(LOG_ERROR, "Cannot open input file!");
        exit(1);
    }

    // start parser
    string line;
    string buf;
    string dummy1, dummy2;
    string techName;
    string botTechName, topTechName;
    int numMCs;
    int tmpUtils;
    bool onlyOneTech = false;
    std::unordered_map<std::string, vector<CellType*>> techname2celltypes;

    // in iccad2022, we assume there is only M1 for routing
    Layer& layer = this->addLayer(string("M").append(to_string(1)), 'r');
    // 1) NumTechnologies
    infile >> buf >> numTechlibs;
    if (numTechlibs == 1) {
        onlyOneTech = true;
    }
    for (int techId = 0; techId < numTechlibs; techId++) {
        infile >> buf >> techName >> numMCs;
        if (techname2celltypes.find(techName) != techname2celltypes.end()) {
            printlog(LOG_WARN, "tech re-defined: %s", techName.c_str());
            continue;
        }
        vector<CellType*> cur_celltypes;
        cur_celltypes.reserve(numMCs);
        for (int mcId = 0; mcId < numMCs; mcId++) {
            string mcName, pinName;
            int mcWidth, mcHeight, numPins, pinRelLx, pinRelLy;
            CellType* celltype;
            string isMacro;
            if(st::setting.withMacro)//2023 version
            {
                
                infile >> buf >> isMacro >> mcName >> mcWidth >> mcHeight >> numPins;
                mcName = techName + "_" + mcName;
                celltype = this->addCellType(mcName + "_" + std::to_string(0), this->celltypes.size());
                celltype->width = mcWidth;
                celltype->height = mcHeight;
                if(isMacro=="N")
                {
                    celltype->stdcell = true;  // NOTE: iccad 2022 does not have fixed cells
                }else{
                    celltype->stdcell = true; 
                    celltype->macro = true;
                }
            }else{
                infile >> buf >> mcName >> mcWidth >> mcHeight >> numPins;
                mcName = techName + "_" + mcName;
                celltype = this->addCellType(mcName, this->celltypes.size());
                celltype->width = mcWidth;
                celltype->height = mcHeight;
                celltype->stdcell = true;
            }
            
            
            celltype->pins.reserve(numPins);
            for (int pinId = 0; pinId < numPins; pinId++) {
                infile >> buf >> pinName >> pinRelLx >> pinRelLy;
                int pinRelHx = pinRelLx;
                int pinRelHy = pinRelLy;
                if(pinId != std::stoi(pinName.substr(1)) - 1)
                {
                    int whattttt=1;
                }
                PinType* pintype = celltype->addPin(pinName, 'x', 's');
                pintype->addShape(layer, pinRelLx, pinRelLy, pinRelHx, pinRelHy);
            }
            cur_celltypes.push_back(celltype);
        }
        techname2celltypes[techName] = std::move(cur_celltypes);
    }
    printlog(LOG_INFO, "finish reading celltype");

    // 2) DieSize
    infile >> buf >> dieLX >> dieLY >> dieHX >> dieHY;

    // 3) Die utilization
    maxUtilM.resize(2);
    infile >> buf >> tmpUtils;  // top
    maxUtilM[1] = static_cast<double>(tmpUtils) / 100.0;
    infile >> buf >> tmpUtils;  // bottom
    maxUtilM[0] = static_cast<double>(tmpUtils) / 100.0;

    // 4) Rows
    rowsM.resize(2);
    rowHeights.resize(2);
    numRows.resize(2);
    rowStart.resize(2);
    int startX, startY, rowLength, rowHeight, repeatCount;
    int64_t coreSize_top, coreSize_bot;
    for (int dieId = 1; dieId >= 0; dieId--) {
        infile >> buf >> startX >> startY >> rowLength >> rowHeight >> repeatCount;
        string type = dieId == 1 ? "TOP" : "BOT";
        if (dieId == 1)
            coreSize_top = (rowLength * rowHeight * repeatCount);
        else
            coreSize_bot = (rowLength * rowHeight * repeatCount);
        for (int i = 0; i < repeatCount; i++) {
            Row* row = this->addRow("core_SITE_" + type + "_ROW_" + to_string(i), "core", startX, startY);
            row->xStep(rowLength);
            row->yStep(rowHeight);
            row->xNum(1);
            row->yNum(1);
            row->flip(false);  // NOTE: in iccad2022, no flip row
            startY += rowHeight;
            rowsM[dieId].push_back(row);
        }
        rowHeights[dieId] = rowHeight;
        numRows[dieId] = repeatCount;
        rowStart[dieId] = startY;
    }
    // if (coreSize_top != coreSize_bot)
    //     printlog(LOG_WARN, "core size %ld not equal %ld", coreSize_top, coreSize_bot);  // FIXME:
    coreLX = dieLX;
    coreLY = dieLY;
    coreHX = rowLength;
    coreHY = rowHeight * repeatCount;
    // TODO: which core to choose

    // 5) Die Tech
    infile >> buf >> topTechName;
    infile >> buf >> botTechName;
    // just copy the vector of ptr, I believe it won't take too much time
    celltypesM.push_back(techname2celltypes[botTechName]);
    celltypesM.push_back(techname2celltypes[topTechName]);

    // 6) hybrid bonding
    infile >> buf >> bondingSizeX >> bondingSizeY;
    infile >> buf >> bondingSpacing;
    if(st::setting.withMacro)//2023 version
    {
        infile >> buf >> bondingCost;
    }
    // 7) cells
    int numCells;
    string cellName, celltypeName;
    infile >> buf >> numCells;
    loadSettings(numCells);
    for (int cellId = 0; cellId < numCells; cellId++) {
        infile >> buf >> cellName >> celltypeName;
        int celltypeId = std::stoi(celltypeName.substr(2)) - 1;
        int cellx = 0;
        int celly = 0;
        Cell* cell = this->addCell(cellName, this->celltypesM[0][celltypeId]);
        if(this->celltypesM[0][celltypeId]->macro){
            cell->setOrient(0);
        } else {
            cell->setOrient(-1);
        }      
        if(cellName=="C16430")
        {
            int debugg=1;
        }
        cell->place(cellx, celly, false, false);
        cell->fixed(false);
        cell->celltypeId = celltypeId;
        cell->setDieId(0);
        // for (int techId = 1; techId < numTechlibs; techId++) {  // FIXME:
        if (!onlyOneTech) {  // FIXME:
            Cell* cell_mT = this->addCell_mT(cellName, this->celltypesM[1][celltypeId]);
            if(this->celltypesM[1][celltypeId]->macro){
                cell_mT->setOrient(0);
            } else {
                cell_mT->setOrient(-1);
            }
            cell_mT->place(cellx, celly, false, false);
            cell_mT->fixed(false);
            cell_mT->celltypeId = celltypeId;
            cell_mT->setDieId(1);
        }
    }
    printlog(LOG_INFO, "finish reading cell");

    // 8) nets
    int numNets, numPins;
    string netName;
    infile >> buf >> numNets;
    netId2cellPinIds.resize(numNets);
    for (int netId = 0; netId < numNets; netId++) {
        infile >> buf >> netName >> numPins;
        string tmp;
        int cellId, pinId;
        netId2cellPinIds[netId].resize(numPins);
        Net* net = this->addNet(netName);
        Net* net_mT = this->addNet_mT(netName);  // FIXME:

        for (int i = 0; i < numPins; i++) {
            infile >> buf >> tmp;
            std::size_t pos = tmp.find("/");
            if (pos != std::string::npos) {
                string cellName = tmp.substr(0, pos);
                string pinName = tmp.substr(pos + 1);
                // int cellId = std::stoi(cellName.substr(1)) - 1;
                if(std::stoi(cellName.substr(1)) - 1!=this->name_cells[cellName]->id)
                {
                    logger.error("ERROR!!!!!!!!!!!!!! INPUT1");
                }
                int cellId = this->name_cells[cellName]->id;
                int pinId = std::stoi(pinName.substr(1)) - 1;
                netId2cellPinIds[netId][i] = {cellId, pinId};
                /* bot */
                //Cell* cell = this->cells[cellId];
                Cell* cell = this->name_cells[cellName];
                Pin* pin = cell->pin(pinId);
                if (pin->is_connected) {
                    string netName(net->name);
                    printlog(LOG_WARN, "Pin is re-connected: %s %s %d", netName.c_str(), cellName.c_str(), pinId);
                }
                cell->is_connected = true;
                pin->net = net;
                pin->is_connected = true;
                net->addPin(pin);
                if (!onlyOneTech) {  // FIXME:
                    /* top */
                    //Cell* cell_mT = this->cells_mT[cellId];
                    Cell* cell_mT = this->name_cells_mT[cellName];
                    Pin* pin_mT = cell_mT->pin(pinId);
                    cell_mT->is_connected = true;
                    pin_mT->net = net_mT;
                    pin_mT->is_connected = true;
                    net_mT->addPin(pin_mT);
                }
            } else {
                printlog(LOG_ERROR, "error format %s", tmp.c_str());
            }
        }
    }

    // setup site, in iccad 2022, we should ignore site size
    siteW = 1;
    siteH = 1;
    // FIXME:
    siteH = rowHeight;
    // init regions
    regions[0]->addRect(dieLX, dieLY, dieHX, dieHY);
    regions[0]->id = 0;
    regions[0]->resetRects();
    for (Cell* cell : cells) {
        if (!cell->region) {
            cell->region = regions[0];
        }
    }
    // for each net, we init a bonding
    for (std::size_t netId = 0; netId < nets.size(); netId++) {
        bondings.emplace_back(
            std::numeric_limits<int>::min(), std::numeric_limits<int>::min(), bondingSizeX, bondingSizeY, netId);
    }

    printlog(LOG_INFO, "Die Area: Lx: %d Ly: %d Hx: %d Hy: %d", dieLX, dieLY, dieHX, dieHY);
    printlog(LOG_INFO, "Core Area: Lx: %d Ly: %d Hx: %d Hy: %d", coreLX, coreLY, coreHX, coreHY);
    printlog(LOG_INFO, ": Bot: %.4f  Top: %.4f", maxUtilM[0], maxUtilM[1]);
    printlog(LOG_INFO, "#rows: Bot: %d Top: %d", rowsM[0].size(), rowsM[1].size());
    // printlog(LOG_INFO, "RowHeight: Bot: %d Top: %d", rowsM[0][0]->yStep(), rowsM[1][0]->yStep());
    printlog(LOG_INFO, "RowHeight: Bot: %d Top: %d", rowHeights[0], rowHeights[1]);
    printlog(LOG_INFO, "#techs: %d #celltyps: %d", numTechlibs, celltypesM[0].size());
    printlog(LOG_INFO, "#cells: %d #nets: %d", cells.size(), nets.size());
    printlog(LOG_INFO, "#bondings: %d", bondings.size());

    return true;
}

bool Database::writeICCAD2022(const string& file) {
    printlog(LOG_INFO, "start writing output...");
    std::ofstream fs(file.c_str());
    if (!fs.good()) {
        printlog(LOG_ERROR, "cannot open file: %s", file.c_str());
        return false;
    }
    // get cell and terminal info
    vector<int> cellInBot;
    vector<int> cellInTop;
    vector<int> validBondingIds;
    for (auto cell : this->cells) {
        if (cell->getDieId() != -1) {
            if (cell->getDieId() == 0) {
                cellInBot.emplace_back(cell->id);
            } else if (cell->getDieId() == 1) {
                cellInTop.emplace_back(cell->id);
            } else {
                printlog(LOG_WARN, "Unknown dieId %d for cell %s", cell->getDieId(), cell->name().c_str());
            }
        } else {
            printlog(LOG_WARN, "Haven't yet assigned dieId for cell %s", cell->name().c_str());
        }
    }
    for (std::size_t id = 0; id < bondings.size(); id++) {
        if (bondings[id].valid) {
            validBondingIds.emplace_back(id);
        }
    }
    // write files
    fs << "TopDiePlacement " << cellInTop.size() << std::endl;
    for (int cellId : cellInTop) {
        Cell* cell = cells[cellId];
        if(st::setting.withMacro) {
            fs << "Inst " << cell->name() << " " << cell->lx() << " " << cell->ly() << " R" << max(cell->orient(), 0) * 90 << std::endl;
        } else {
            fs << "Inst " << cell->name() << " " << cell->lx() << " " << cell->ly() << std::endl;
        }
    }
    fs << "BottomDiePlacement " << cellInBot.size() << std::endl;
    for (int cellId : cellInBot) {
        Cell* cell = cells[cellId];
        if(st::setting.withMacro) {
            fs << "Inst " << cell->name() << " " << cell->lx() << " " << cell->ly() << " R" << max(cell->orient(), 0) * 90 << std::endl;
        } else {
            fs << "Inst " << cell->name() << " " << cell->lx() << " " << cell->ly() << std::endl;
        }
        
    }
    fs << "NumTerminals " << validBondingIds.size() << std::endl;
    for (int bondingId : validBondingIds) {
        auto& bonding = bondings[bondingId];
        int netId = bonding.netId();
        string netName = nets[netId]->name;
        fs << "Terminal " << netName << " " << bonding.lx() << " " << bonding.ly() << std::endl;
    }
    fs.close();
    printlog(LOG_INFO, "finish writing output in %s", file.c_str());
    return true;
}

bool Database::writeOpenroad_vias(const string& file) {
    printlog(LOG_INFO, "start writing output...");
    std::ofstream fs(file.c_str());
    if (!fs.good()) {
        printlog(LOG_ERROR, "cannot open file: %s", file.c_str());
        return false;
    }
    // get cell and terminal info
    vector<int> cellInBot;
    vector<int> cellInTop;
    vector<int> validBondingIds;
    for (auto cell : this->cells) {
        if (cell->getDieId() != -1) {
            if (cell->getDieId() == 0) {
                cellInBot.emplace_back(cell->id);
            } else if (cell->getDieId() == 1) {
                cellInTop.emplace_back(cell->id);
            } else {
                printlog(LOG_WARN, "Unknown dieId %d for cell %s", cell->getDieId(), cell->name().c_str());
            }
        } else {
            printlog(LOG_WARN, "Haven't yet assigned dieId for cell %s", cell->name().c_str());
        }
    }
    for (std::size_t id = 0; id < bondings.size(); id++) {
        if (bondings[id].valid) {
            validBondingIds.emplace_back(id);
        }
    }
    // write files
    fs << "NumTerminals " << validBondingIds.size() << std::endl;
    fs << "Terminal size " << static_cast<float>(bondingSizeX) / 2000 << " " << static_cast<float>(bondingSizeY) / 2000 << std::endl;
    for (int bondingId : validBondingIds) {
        auto& bonding = bondings[bondingId];
        int netId = bonding.netId();
        string netName = nets[netId]->name;
        fs << "Terminal " << netName << " " << bonding.lx() << " " << bonding.ly() << std::endl;
    }
    fs.close();
    printlog(LOG_INFO, "finish writing output in %s", file.c_str());
    return true;
}

bool Database::resumeICCAD2022(const std::string& file) {
    printlog(LOG_INFO, "Resume placement. Starting parsing the file %s", file.c_str());
    std::ifstream infile(file);
    if (infile.is_open() == false) {
        printlog(LOG_ERROR, "Cannot open input file!");
        exit(1);
    }

    // clear placement
    for (auto cell : cells) {
        cell->place(std::numeric_limits<int>::min(), std::numeric_limits<int>::min(), -1);
    }
    for (auto& bonding : bondings) {
        bonding.remove();
    }

    // start parser
    string line;
    string buf;
    string cellName, netName;
    int numTopDiePlacement, numBotDiePlacement, numTerminals;
    int lx, ly;

    infile >> buf >> numTopDiePlacement;
    for (int i = 0; i < numTopDiePlacement; i++) {
        infile >> buf >> cellName >> lx >> ly;
        //int cellId = std::stoi(cellName.substr(1)) - 1;
        if(std::stoi(cellName.substr(1)) - 1!=this->name_cells[cellName]->id)
        {
            logger.error("ERROR!!!!!!!!!!!!!! INPUT");
        }
        Cell* cell = this->name_cells[cellName];
        // cells[cellId]->place(lx, ly, 1);
        this->name_cells[cellName]->place(lx, ly, 1);
    }

    infile >> buf >> numBotDiePlacement;
    for (int i = 0; i < numBotDiePlacement; i++) {
        infile >> buf >> cellName >> lx >> ly;
        // int cellId = std::stoi(cellName.substr(1)) - 1;
        // cells[cellId]->place(lx, ly, 0);
        if(std::stoi(cellName.substr(1)) - 1!=this->name_cells[cellName]->id)
        {
            logger.error("ERROR!!!!!!!!!!!!!! INPUT");
        }
        Cell* cell = this->name_cells[cellName];
        cell->place(lx, ly, 0);
    }

    infile >> buf >> numTerminals;
    for (int i = 0; i < numTerminals; i++) {
        infile >> buf >> netName >> lx >> ly;
        int netId = std::stoi(netName.substr(1)) - 1;
        bondings[netId].place(lx, ly);
    }
    printlog(
        LOG_INFO, "#TopCells: %d #BotCells: %d Bondings: %d", numTopDiePlacement, numBotDiePlacement, numTerminals);
    return true;
}

void Database::placeBonding(int lx, int ly, int netId) {
    if (!bondings[netId].valid) {
        numValidBondings++;
    }
    bondings[netId].place(lx, ly);
}

void Database::removeBonding(int netId) {
    if (bondings[netId].valid) {
        bondings[netId].remove();
        numValidBondings--;
    } else {
        printlog(LOG_WARN, "Cannot remove an invalid bonding.");
    }
}

Cell* Database::addCell_mT(const string& name, CellType* type) {
    Cell* cell = getCell(name);
    if (!cell) {
        printlog(LOG_WARN, "mTech cell not defined: %s", name.c_str());
        exit(1);
    }
    cell = new Cell(name, type);
    cell->id = cells_mT.size();
    cells_mT.push_back(cell);
    name_cells_mT.emplace(name, cell);
    return cell;
}

Net* Database::addNet_mT(const string& name, const NDR* ndr) {
    Net* net = getNet(name);
    if (!net) {
        printlog(LOG_WARN, "mTech net not defined: %s", name.c_str());
        exit(1);
    }
    net = new Net(name, ndr);
    net->id = nets_mT.size();
    nets_mT.push_back(net);
    return net;
}