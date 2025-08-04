#include "GlobalParser.h"

void GlobalParser::set_single_design_params(const st::Setting& setting) {
    db::rawDBArgs.reset();
    db::rawDBArgs.liteMode = true;
    db::rawDBArgs.random_place = false;
    utils::verbose_parser_log = false;
    if (setting.dataset == "ispd2005")
        single_ispd2005(setting);
    else if (setting.dataset == "iccad2022")
        single_iccad2022(setting);
    else if (setting.dataset == "ispd2015_without_fence")
        single_ispd2015_without_fence(setting);
    else if (setting.dataset == "ibm")
        single_IBM(setting);
    else if (setting.dataset == "openroad")
        single_openroad(setting);
}

void GlobalParser::single_IBM(const st::Setting& setting) {
    auto design_name = setting.design_name;
    logger.info("loading information of IBM design %s", design_name.c_str());
    const std::string dataset = setting.dataset;
    const std::string root = "./data_mms";

    std::vector<std::string> all_designs;
    auto dir = opendir(root.c_str());
    if ((dir) != NULL) {
        struct dirent* entry;
        entry = readdir(dir);
        while (entry) {
            all_designs.push_back(entry->d_name);
            entry = readdir(dir);
        }
    }
    if (std::find(all_designs.begin(), all_designs.end(), design_name) == all_designs.end())
        logger.error("Design %s not found", design_name.c_str());

    string aux = root + "/" + design_name + "/" + design_name + ".aux";
    string pl = root + "/" + design_name + "/" + design_name + ".pl";
    // set rawDBArgs
    db::rawDBArgs.Format = "bookshelf";
    db::rawDBArgs.Dataset = dataset;
    db::rawDBArgs.DesignName = design_name;
    db::rawDBArgs.BookshelfVariety = "ibm";
    db::rawDBArgs.BookshelfAux = aux;
    db::rawDBArgs.BookshelfPl = pl;
}

void GlobalParser::single_ispd2005(const st::Setting& setting) {
    auto design_name = setting.design_name;
    logger.info("loading information of ISPD2005 design %s", design_name.c_str());
    const std::string dataset = setting.dataset;
    const std::string root = "./data/cad/ispd2005";

    std::vector<std::string> all_designs;
    auto dir = opendir(root.c_str());
    if ((dir) != NULL) {
        struct dirent* entry;
        entry = readdir(dir);
        while (entry) {
            all_designs.push_back(entry->d_name);
            entry = readdir(dir);
        }
    }
    if (std::find(all_designs.begin(), all_designs.end(), design_name) == all_designs.end())
        logger.error("Design %s not found", design_name.c_str());

    string aux = root + "/" + design_name + "/" + design_name + ".aux";
    string pl = root + "/" + design_name + "/" + design_name + ".pl";
    // set rawDBArgs
    db::rawDBArgs.Format = "bookshelf";
    db::rawDBArgs.Dataset = dataset;
    db::rawDBArgs.DesignName = design_name;
    db::rawDBArgs.BookshelfVariety = "ispd2005";
    db::rawDBArgs.BookshelfAux = aux;
    db::rawDBArgs.BookshelfPl = pl;
}

void GlobalParser::single_iccad2022(const st::Setting& setting) {
    logger.info("loading information of ICCAD2022 design %s", setting.design_name.c_str());
    db::rawDBArgs.Format = "iccad2022";
    db::rawDBArgs.Dataset = setting.dataset;
    db::rawDBArgs.DesignName = setting.design_name;
    db::rawDBArgs.ICCAD2022InputFile = setting.input_path;
    db::rawDBArgs.OutputFile = setting.output_path;
}

void GlobalParser::single_ispd2015_without_fence(const st::Setting& setting) {
    auto design_name = setting.design_name;
    logger.info("loading information of design %s", design_name.c_str());
    const std::string dataset = "ispd2015_without_fence";
    const std::string root = "./data/cad/ispd2015_without_fence";

    std::vector<std::string> all_designs;
    auto dir = opendir(root.c_str());
    if ((dir) != NULL) {
        struct dirent* entry;
        entry = readdir(dir);
        while (entry) {
            all_designs.push_back(entry->d_name);
            entry = readdir(dir);
        }
    }
    if (std::find(all_designs.begin(), all_designs.end(), design_name) == all_designs.end())
        logger.error("Design %s not found", design_name.c_str());

    string tech_lef = root + "/" + design_name + "/" + "tech.lef";
    string cell_lef = root + "/" + design_name + "/" + "cells.lef";
    string def = root + "/" + design_name + "/" + "floorplan.def";

    db::rawDBArgs.Format = "lefdef";
    db::rawDBArgs.Dataset = dataset;
    db::rawDBArgs.DesignName = design_name;
    db::rawDBArgs.LefTech = tech_lef;
    db::rawDBArgs.LefCell = cell_lef;
    db::rawDBArgs.DefFile = def;
}

void GlobalParser::single_iccad2019(const st::Setting& setting) {
    auto design_name = setting.design_name;
    logger.info("loading information of design %s", design_name.c_str());
    const std::string dataset = "iccad2019";
    const std::string root = "./data/cad/iccad2019";

    std::vector<std::string> all_designs;
    auto dir = opendir(root.c_str());
    if ((dir) != NULL) {
        struct dirent* entry;
        entry = readdir(dir);
        while (entry) {
            all_designs.push_back(entry->d_name);
            entry = readdir(dir);
        }
    }
    if (std::find(all_designs.begin(), all_designs.end(), design_name) == all_designs.end())
        logger.error("Design %s not found", design_name.c_str());

    string lef = root + "/" + design_name + "/" + design_name + ".input.lef";
    string def = root + "/" + design_name + "/" + design_name + ".input.def";

    db::rawDBArgs.Format = "lefdef";
    db::rawDBArgs.Dataset = dataset;
    db::rawDBArgs.DesignName = design_name;
    db::rawDBArgs.LefFile = lef;
    db::rawDBArgs.DefFile = def;
}

void GlobalParser::single_openroad(const st::Setting& setting) {
    auto design_name = setting.design_name;
    logger.info("loading information of design %s", design_name.c_str());
    const std::string dataset = "openroad";
    // const std::string lef_json = setting.load_json;
    const std::string root = "./data_openroad";

    string lef = root + "/" + design_name + "/" + "leffiles";

    std::vector<std::string> lef_files;
    std::ifstream file(lef.c_str());

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << lef << std::endl;
    }

    std::string lef_file;
    while (std::getline(file, lef_file)) {
        lef_files.push_back(lef_file);
    }

    file.close();

    std::vector<std::string> all_designs;
    auto dir = opendir(root.c_str());
    if ((dir) != NULL) {
        struct dirent* entry;
        entry = readdir(dir);
        while (entry) {
            all_designs.push_back(entry->d_name);
            entry = readdir(dir);
        }
    }
    if (std::find(all_designs.begin(), all_designs.end(), design_name) == all_designs.end())
        logger.error("Design %s not found", design_name.c_str());


    string def = root + "/" + design_name + "/" + "2_1_floorplan.odb.def";

    db::rawDBArgs.Format = "lefdef";
    db::rawDBArgs.Dataset = dataset;
    db::rawDBArgs.DesignName = design_name;
    db::rawDBArgs.LefFiles = lef_files;
    db::rawDBArgs.DefFile = def;
}

Dict GlobalParser::preprocess_design_info(shared_ptr<gp::GPDatabase> gpdb) {
    const tuple<int, int, int, int>& coreInfo = gpdb->getCoreInfo();
    const tuple<int, int, int, int>& dieInfo = gpdb->getDieInfo();
    int numTechlibs = 1;

    torch::Tensor rowHeights = torch::tensor({gpdb->getSiteHeight(), gpdb->getSiteHeight()}, torch::dtype(torch::kFloat32));
    torch::Tensor numRows = torch::tensor({gpdb->getNumRow(), gpdb->getNumRow()}, torch::dtype(torch::kInt));
    torch::Tensor node_orient_bot = gpdb->getNodeOrientTensor();
    torch::Tensor node_orient_top = gpdb->getNodeOrientTensor();
    torch::Tensor maxUtilM = torch::tensor({0.80, 0.80}, torch::dtype(torch::kFloat32));
    
    int dieLX, dieHX, dieLY, dieHY;
    int coreLX, coreHX, coreLY, coreHY;

    std::tie(dieLX, dieHX, dieLY, dieHY) = dieInfo;
    std::tie(coreLX, coreHX, coreLY, coreHY) = coreInfo;
    torch::Tensor die_info = torch::tensor({dieLX, dieHX, dieLY, dieHY}, torch::dtype(torch::kFloat32));
    torch::Tensor core_info = torch::tensor({coreLX, coreHX, coreLY, coreHY}, torch::dtype(torch::kFloat32));

    int siteWidth = gpdb->getSiteWidth();
    int siteHeight = gpdb->getSiteHeight();
    const tuple<int, int> site_info = make_tuple(double(siteWidth), double(siteHeight));
    int bondingSizeX = st::setting.bondingSizeX == 0 ? siteWidth * 4 : st::setting.bondingSizeX;
    int bondingSizeY = st::setting.bondingSizeY == 0 ? siteHeight : st::setting.bondingSizeY;
    int bondingSpace = st::setting.bondingSpace;
    gpdb->bondingSizeX = bondingSizeX;
    gpdb->bondingSizeY = bondingSizeY;
    torch::Tensor bondingInfo = torch::tensor({bondingSizeX, bondingSizeY, bondingSpace}, torch::dtype(torch::kFloat32));
    int bondingCost = 0;

    torch::Tensor node_pos = gpdb->getNodeCPosTensor();
    //torch::Tensor node_rotate = gpdb->getNodeCRotateTensor();
    torch::Tensor node_size = gpdb->getNodeSizeTensor();
    torch::Tensor node_size_bot = gpdb->getNodeSizeTensor();
    torch::Tensor node_size_top = gpdb->getNodeSizeTensor();
    torch::Tensor pin_rel_cpos = gpdb->getPinRelCPosTensor();
    torch::Tensor pin_rel_cpos_bot = gpdb->getPinRelCPosTensor();
    torch::Tensor pin_rel_cpos_top = gpdb->getPinRelCPosTensor();
    torch::Tensor pin_size = gpdb->getPinSizeTensor();
    torch::Tensor pin_size_bot = gpdb->getPinSizeTensor();
    torch::Tensor pin_size_top = gpdb->getPinSizeTensor();
    torch::Tensor node_type = gpdb->getNodeTypeTensor();

    torch::Tensor pin_id2node_id = gpdb->getPinId2NodeIdTensor();
    vector<torch::Tensor> hyperedge_info = gpdb->getHyperedgeInfoTensor();
    torch::Tensor hyperedge_index = hyperedge_info[0];
    torch::Tensor hyperedge_list = hyperedge_info[1];
    torch::Tensor hyperedge_list_end = hyperedge_info[2];

    vector<torch::Tensor> node2pin_info = gpdb->getNode2PinInfoTensor();
    torch::Tensor node2pin_index = node2pin_info[0];
    torch::Tensor node2pin_list = node2pin_info[1];
    torch::Tensor node2pin_list_end = node2pin_info[2];

    vector<torch::Tensor> region_info = gpdb->getRegionInfoTensor();
    torch::Tensor node_id2region_id = region_info[0];
    torch::Tensor region_boxes = region_info[1];
    torch::Tensor region_boxes_end = region_info[2];

    vector<tuple<gp::index_type, gp::index_type, string>> node_type_indices = gpdb->getNodeTypeIndices();
    vector<string> node_id2node_name = gpdb->getNodeId2NodeName();
    torch::Tensor macro_mask = gpdb->getMacroMaskTensor();

    vector<string> all_node_types;
    int mov_end_idx = -1;
    int fix_end_idx = -1;
    int connected_end_idx = -1;
    for (auto node_info : node_type_indices) {
        gp::index_type start_idx;
        gp::index_type end_idx;
        string type_name;
        std::tie(start_idx, end_idx, type_name) = node_info;
        all_node_types.push_back(type_name);
        if (type_name == "FloatMov") mov_end_idx = end_idx;
        if (type_name == "FloatFix") fix_end_idx = end_idx;
        if (type_name == "IOPin") {
            connected_end_idx = end_idx;
            mov_end_idx = end_idx;
        }
    }
    cout << "mov_end_idx: " << mov_end_idx << ", fix_end_idx: " << fix_end_idx
         << ", connected_end_idx: " << connected_end_idx << endl;

    // Mov + FloatMov
    const tuple<int, int> movable_index = make_tuple(0, mov_end_idx);
    // Mov + FloatMov + Fix + IOPin
    const tuple<int, int> connected_index = make_tuple(0, connected_end_idx);
    // Fix + IOPin + Blkg + FloatIOPin + FloatFix
    const tuple<int, int> fixed_index = make_tuple(mov_end_idx, fix_end_idx);

    /* bonding info */
    torch::Tensor bonding_size = gpdb->getBondingSizeTensor();
    torch::Tensor bonding_pos = gpdb->getBondingCPosTensor();
    // cout << bonding_size << endl;

    Dict design_info = {
        {"dataset", db::rawDBArgs.Dataset},
        {"design_name", db::rawDBArgs.DesignName},
        {"node_type_indices", node_type_indices},
        {"node_id2node_name", node_id2node_name},
        {"movable_index", movable_index},
        {"connected_index", connected_index},
        {"fixed_index", fixed_index},
        {"site_info", site_info},
        {"die_info", die_info},
        {"core_info", core_info},

        {"numTechlibs", numTechlibs},
        {"maxUtilM", maxUtilM},
        {"rowHeights", rowHeights},
        {"numRows", numRows},

        {"node_pos", node_pos.contiguous()},
        {"macro_mask", macro_mask.contiguous()},
        {"node_size_bot", node_size_bot.contiguous()},
        {"pin_rel_cpos_bot", pin_rel_cpos_bot.contiguous()},
        {"pin_size_bot", pin_size_bot.contiguous()},
        {"node_size_top", node_size_top.contiguous()},
        {"pin_rel_cpos_top", pin_rel_cpos_top.contiguous()},
        {"pin_size_top", pin_size_top.contiguous()},
        {"node_size", node_size_bot.contiguous()},
        {"pin_rel_cpos", pin_rel_cpos_bot.contiguous()},
        {"pin_size", pin_size_bot.contiguous()},
        {"node_orient_bot", node_orient_bot.contiguous()},
        {"node_orient_top", node_orient_top.contiguous()},
        {"node_type", node_type.contiguous()},

        {"bondingInfo", bondingInfo.contiguous()},
        {"bondingCost", bondingCost},
        {"bonding_size", bonding_size.contiguous()},
        {"bonding_pos", bonding_pos.contiguous()},

        {"pin_id2node_id", pin_id2node_id.to(torch::kLong).contiguous()},
        {"hyperedge_index", hyperedge_index.to(torch::kLong).contiguous()},
        {"hyperedge_list", hyperedge_list.to(torch::kLong).contiguous()},
        {"hyperedge_list_end", hyperedge_list_end.to(torch::kLong).contiguous()},
        {"node2pin_index", node2pin_index.to(torch::kLong).contiguous()},
        {"node2pin_list", node2pin_list.to(torch::kLong).contiguous()},
        {"node2pin_list_end", node2pin_list_end.to(torch::kLong).contiguous()},
        {"node_id2region_id", node_id2region_id.to(torch::kLong).contiguous()},
        {"region_boxes", region_boxes.contiguous()},
        {"region_boxes_end", region_boxes_end.to(torch::kLong).contiguous()},
    };
    return design_info;
}

Dict GlobalParser::preprocess_design_info_iccad2022(shared_ptr<gp::GPDatabase> gpdb) {
    /* tech info */
    int numTechlibs = gpdb->numTechlibs;
    torch::Tensor maxUtilM = torch::tensor({gpdb->maxUtilM[0], gpdb->maxUtilM[1]}, torch::dtype(torch::kFloat32));
    torch::Tensor rowHeights = torch::tensor({gpdb->rowHeights[0], gpdb->rowHeights[1]}, torch::dtype(torch::kFloat32));
    torch::Tensor bondingInfo =
        torch::tensor({gpdb->bondingSizeX, gpdb->bondingSizeY, gpdb->bondingSpacing}, torch::dtype(torch::kFloat32));
    int bondingCost = gpdb->bondingCost;
    torch::Tensor numRows = torch::tensor({gpdb->numRows[0], gpdb->numRows[1]}, torch::dtype(torch::kInt));

    const tuple<int, int, int, int>& coreInfo = gpdb->getCoreInfo();
    const tuple<int, int, int, int>& dieInfo = gpdb->getDieInfo();

    int dieLX, dieHX, dieLY, dieHY;
    int coreLX, coreHX, coreLY, coreHY;

    std::tie(dieLX, dieHX, dieLY, dieHY) = dieInfo;
    std::tie(coreLX, coreHX, coreLY, coreHY) = coreInfo;
    torch::Tensor die_info = torch::tensor({dieLX, dieHX, dieLY, dieHY}, torch::dtype(torch::kFloat32));
    torch::Tensor core_info = torch::tensor({coreLX, coreHX, coreLY, coreHY}, torch::dtype(torch::kFloat32));

    int siteWidth = gpdb->getSiteWidth();
    int siteHeight = gpdb->getSiteHeight();
    const tuple<int, int> site_info = make_tuple(double(siteWidth), double(siteHeight));

    /* node/pin info tech A/B*/
    torch::Tensor node_pos = gpdb->getNodeCPosTensor();
    torch::Tensor macro_mask = gpdb->getMacroMaskTensor();
    //torch::Tensor node_rotate = gpdb->getNodeCRotateTensor();
    torch::Tensor node_size_bot = gpdb->getNodeSizeTensor();
    torch::Tensor node_size_top = gpdb->getNodeSizeTensor_mT();
    torch::Tensor node_orient_bot = gpdb->getNodeOrientTensor();
    torch::Tensor node_orient_top = gpdb->getNodeOrientTensor_mT();
    torch::Tensor node_type = gpdb->getNodeTypeTensor();
    torch::Tensor pin_rel_cpos_bot = gpdb->getPinRelCPosTensor();
    torch::Tensor pin_rel_cpos_top = gpdb->getPinRelCPosTensor_mT();
    torch::Tensor pin_size_bot = gpdb->getPinSizeTensor();
    torch::Tensor pin_size_top = gpdb->getPinSizeTensor_mT();

    torch::Tensor pin_id2node_id = gpdb->getPinId2NodeIdTensor();
    vector<torch::Tensor> hyperedge_info = gpdb->getHyperedgeInfoTensor();
    torch::Tensor hyperedge_index = hyperedge_info[0];
    torch::Tensor hyperedge_list = hyperedge_info[1];
    torch::Tensor hyperedge_list_end = hyperedge_info[2];
    vector<torch::Tensor> node2pin_info = gpdb->getNode2PinInfoTensor();
    torch::Tensor node2pin_index = node2pin_info[0];
    torch::Tensor node2pin_list = node2pin_info[1];
    torch::Tensor node2pin_list_end = node2pin_info[2];

    vector<torch::Tensor> region_info = gpdb->getRegionInfoTensor();
    torch::Tensor node_id2region_id = region_info[0];
    torch::Tensor region_boxes = region_info[1];
    torch::Tensor region_boxes_end = region_info[2];

    vector<tuple<gp::index_type, gp::index_type, string>> node_type_indices = gpdb->getNodeTypeIndices();
    vector<string> node_id2node_name = gpdb->getNodeId2NodeName();

    vector<string> all_node_types;
    int mov_end_idx = -1;
    int fix_end_idx = -1;
    int connected_end_idx = -1;
    for (auto node_info : node_type_indices) {
        gp::index_type start_idx;
        gp::index_type end_idx;
        string type_name;
        std::tie(start_idx, end_idx, type_name) = node_info;
        all_node_types.push_back(type_name);
        if (type_name == "FloatMov") mov_end_idx = end_idx;
        if (type_name == "FloatFix") fix_end_idx = end_idx;
        if (type_name == "IOPin") connected_end_idx = end_idx;
    }

    // Mov + FloatMov
    const tuple<int, int> movable_index = make_tuple(0, mov_end_idx);
    // Mov + FloatMov + Fix + IOPin
    const tuple<int, int> connected_index = make_tuple(0, connected_end_idx);
    // Fix + IOPin + Blkg + FloatIOPin + FloatFix
    const tuple<int, int> fixed_index = make_tuple(mov_end_idx, fix_end_idx);

    /* bonding info */
    torch::Tensor bonding_size = gpdb->getBondingSizeTensor();
    torch::Tensor bonding_pos = gpdb->getBondingCPosTensor();

    Dict design_info = {
        {"dataset", db::rawDBArgs.Dataset},
        {"design_name", db::rawDBArgs.DesignName},
        {"node_type_indices", node_type_indices},
        {"node_id2node_name", node_id2node_name},
        {"movable_index", movable_index},
        {"connected_index", connected_index},
        {"fixed_index", fixed_index},
        {"site_info", site_info},
        {"die_info", die_info},
        {"core_info", core_info},

        {"numTechlibs", numTechlibs},
        {"maxUtilM", maxUtilM},
        {"rowHeights", rowHeights},
        {"numRows", numRows},

        {"node_pos", node_pos.contiguous()},
        {"macro_mask", macro_mask.contiguous()},
        {"node_size_bot", node_size_bot.contiguous()},
        {"pin_rel_cpos_bot", pin_rel_cpos_bot.contiguous()},
        {"pin_size_bot", pin_size_bot.contiguous()},
        {"node_size_top", node_size_top.contiguous()},
        {"pin_rel_cpos_top", pin_rel_cpos_top.contiguous()},
        {"pin_size_top", pin_size_top.contiguous()},
        {"node_size", node_size_bot.contiguous()},
        {"pin_rel_cpos", pin_rel_cpos_bot.contiguous()},
        {"pin_size", pin_size_bot.contiguous()},
        {"node_orient_bot", node_orient_bot.contiguous()},
        {"node_orient_top", node_orient_top.contiguous()},
        {"node_type", node_type.contiguous()},

        {"bondingInfo", bondingInfo.contiguous()},
        {"bondingCost", bondingCost},
        {"bonding_size", bonding_size.contiguous()},
        {"bonding_pos", bonding_pos.contiguous()},

        {"pin_id2node_id", pin_id2node_id.to(torch::kLong).contiguous()},
        {"hyperedge_index", hyperedge_index.to(torch::kLong).contiguous()},
        {"hyperedge_list", hyperedge_list.to(torch::kLong).contiguous()},
        {"hyperedge_list_end", hyperedge_list_end.to(torch::kLong).contiguous()},
        {"node2pin_index", node2pin_index.to(torch::kLong).contiguous()},
        {"node2pin_list", node2pin_list.to(torch::kLong).contiguous()},
        {"node2pin_list_end", node2pin_list_end.to(torch::kLong).contiguous()},
        {"node_id2region_id", node_id2region_id.to(torch::kLong).contiguous()},
        {"region_boxes", region_boxes.contiguous()},
        {"region_boxes_end", region_boxes_end.to(torch::kLong).contiguous()},
    };
    return design_info;
}