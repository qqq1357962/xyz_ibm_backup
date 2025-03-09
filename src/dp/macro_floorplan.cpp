#include "macro_floorplan.h"
#include "die_partitioner.h"

double gaussianKernel(double mean, double stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(mean, stddev);
    return dis(gen);
}

namespace dp {
/// @brief The macro legalization follows the way of floorplanning,
/// because macros have quite different sizes.


//vector<int> devide_to_pieces(NodeData &data, int die_id, int numPart) {

bool macroFloorplan(NodeData& data,
                    //DetailedPlaceDataTensor& lg_db_at,
                    torch::Tensor node_pos_lg,
                    torch::Tensor mov_node_weights,
                    torch::Tensor num_sites_y,
                    torch::Tensor row_height,
                    float row_start,
                    int cell_mov_lhs, 
                    int cell_mov_rhs) {
    if(st::setting.use_greedy_place_in_fp)
    {
        logger.info("skip fp, only use greedy place for macros");
        return false;
    }
    //需要的材料：size,node_weight,row_height,没了
    fp::Database fpDatabase;
    fpDatabase.numRotatable = 0;
    fpDatabase.nMacros = 0;
    fpDatabase.nMacros1 = 0;
    fpDatabase.nMacros2 = 0;
    std::vector<int> macros;
    vector<int> partvec2clusterID;
    double ratio_std=1.05;
    auto node_pos_a = data.node_pos.accessor<float,2>();
    auto node_die_a = data.node_die.accessor<int,1>();
    auto node_pos_lg_a = node_pos_lg.accessor<float,2>();
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //DetailedPlaceData dp_db(data, lg_db_at, num_sites_y, row_height);//dp_db need row_height as a input
    //dp_db.yl = max(dp_db.yl , row_start);
    //dp_db.yh = min(dp_db.yh , row_start + row_height * num_sites_y);

    int numPart = st::setting.numPart;
    DiePartitioner partitioner1;
    torch::Tensor node_weight = mov_node_weights[0];
    //下面两块很像，能否合并？
    vector<int> partition_result1 = partitioner1.devide_to_pieces(data, node_weight, numPart);
    
    

    for (int i = 0; i < data.cell_mov_rhs; ++i) {
        //if (dp_db.is_dummy_fixed(i)) {
            //if (dp_db.node_weight[i] == 0) {
        if (data.macro_mask[i].item<int>()==1) {
            if(node_weight[i].item<int>()==0){
                continue;
            }
            // in some extreme case, some macros with 0 area should be ignored
            // float area = dp_db.node_size_x[i] * dp_db.node_size_y[i];
            float area = data.node_size[i][0].item<float>() * data.node_size[i][1].item<float>();//@@check
            if (area > 0) {
                macros.push_back(i);
            }
        }
    }
    
    // if(macros.size()<=1)
    // {
    //     return true;
    // }

    //add standard cells blocks
    vector<float> areas;
    areas.resize(numPart);
    for(int i=0;i<numPart;i++) areas[i]=0;
    for(int i=0;i<data.cell_mov_rhs;i++)
    {
        //if(dp_db.node_weight[i]==0)
        if(node_weight[i].item<int>()==0)
        {
            continue;
        }
        //if (dp_db.is_dummy_fixed(i)) {
        if (data.macro_mask[i].item<int>()==1) {
            continue;
        }
        int block_id = partition_result1[i];
        //float area = dp_db.node_size_x[i] * dp_db.node_size_y[i];
        float area = data.node_size[i][0].item<float>() * data.node_size[i][1].item<float>();//@@check
        if(block_id<0)
        {
            logger.info("ERROR! block_id: %d area: %f node_id: %d", block_id, area, i);
        }
        areas[block_id]+=area;
    }
    
    int partNum_valid=0;
    for(int i=0;i<numPart;i++)
    {
        if(areas[i]<=0)
        {
            partvec2clusterID.push_back(-1);
            continue;
        }
        string name = "STD"+to_string(partNum_valid);
        cout<<"adding "<<name<<endl;
        //logger.info("adding %s",name);
        fp::Macro* macro = fpDatabase.addMacro(name);
        macro->id = -partNum_valid;
        int width = int(ratio_std*sqrt(areas[i]))+1;
        int height = int(ratio_std*sqrt(areas[i]))+1;
        macro->setWidth(width,width);
        macro->setHeight(height,height);
        partvec2clusterID.push_back(partNum_valid);
        partNum_valid++;
    }
    
    
    //add macros
    for (int i = 0; i < macros.size(); i++) {
        int cellID = macros[i];// id in dp dataset
        fp::Macro* macro = fpDatabase.addMacro(data.node_id2node_name[cellID]);
        cout<<"adding "<<data.node_id2node_name[cellID]<<endl;
        macro->id = cellID;
        //float area = data.node_size_bot[i][0].item<float>() * data.node_size[i][1].item<float>();//@@check
        int width0 = int(1.0*data.node_size_bot[cellID][0].item<int>());
        int width1 = int(1.0*data.node_size_top[cellID][0].item<int>());
        int height0 = int(1.0*data.node_size_bot[cellID][1].item<int>());
        int height1 = int(1.0*data.node_size_top[cellID][1].item<int>());
        macro->setWidth(width0,width1);
        // macro->setHeight(dp_db.node_size_y[cellID]);
        macro->setHeight(height0,height1);
        partition_result1[macros[i]] = numPart+i;//set id
        partvec2clusterID.push_back(partNum_valid+i);//numPart+i-->partNum_valid+i
        fpDatabase.macro_sequence_rotatable.push_back(partNum_valid+i);
    }
    fpDatabase.numRotatable+=macros.size();
    fpDatabase.nMacros1 = macros.size() + partNum_valid;
    int partNum_valid1 = partNum_valid;
    int top_start_partvec_idx = numPart+macros.size();

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //int numPart = 50;
    macros.clear();
    DiePartitioner partitioner2;
    node_weight = mov_node_weights[1];
    vector<int> partition_result2 = partitioner2.devide_to_pieces(data, node_weight, numPart);

    for (int i = 0; i < data.cell_mov_rhs; ++i) {
        //if (dp_db.is_dummy_fixed(i)) {
            //if (dp_db.node_weight[i] == 0) {
        if (data.macro_mask[i].item<int>()==1) {
            if(node_weight[i].item<int>()==0){
                continue;
            }
            // in some extreme case, some macros with 0 area should be ignored
            // float area = dp_db.node_size_x[i] * dp_db.node_size_y[i];
            float area = data.node_size[i][0].item<float>() * data.node_size[i][1].item<float>();//@@check
            if (area > 0) {
                macros.push_back(i);
            }
        }
    }
    
    // if(macros.size()<=1)
    // {
    //     return true;
    // }

    //add standard cells blocks
    //vector<float> areas;
    areas.resize(numPart);
    for(int i=0;i<numPart;i++) areas[i]=0;
    for(int i=0;i<data.cell_mov_rhs;i++)
    {
        //if(dp_db.node_weight[i]==0)
        if(node_weight[i].item<int>()==0)
        {
            continue;
        }
        //if (dp_db.is_dummy_fixed(i)) {
        if (data.macro_mask[i].item<int>()==1) {
            continue;
        }
        int block_id = partition_result2[i];
        //float area = dp_db.node_size_x[i] * dp_db.node_size_y[i];
        float area = data.node_size[i][0].item<float>() * data.node_size[i][1].item<float>();//@@check
        if(block_id<0)
        {
            logger.info("ERROR! block_id: %d area: %f node_id: %d", block_id, area, i);
        }
        areas[block_id]+=area;
    }
    
    partNum_valid=fpDatabase.nMacros1;
    //add std cells
    for(int i=0;i<numPart;i++)
    {
        if(areas[i]<=0)
        {
            partvec2clusterID.push_back(-1);
            continue;
        }
        //int cell_id =  + partNum_valid;
        string name = "STD"+to_string(partNum_valid);
        //logger.info("adding %s",name);
        cout<<"adding "<<name<<endl;
        fp::Macro* macro = fpDatabase.addMacro(name);
        macro->id = -partNum_valid;
        macro->isFromSTD = 1;
        int width = int(ratio_std*sqrt(areas[i]))+1;
        int height = int(ratio_std*sqrt(areas[i]))+1;
        macro->setWidth(width,width);
        macro->setHeight(height,height);
        partvec2clusterID.push_back(partNum_valid);
        partNum_valid++;
    }

    //add macros
    for (int i = 0; i < macros.size(); i++) {
        int cellID = macros[i];// id in dp dataset
        fp::Macro* macro = fpDatabase.addMacro(data.node_id2node_name[cellID]);
        macro->id = cellID;
        float area = data.node_size[i][0].item<float>() * data.node_size[i][1].item<float>();//@@check
        int width0 = int(1.0*data.node_size_bot[cellID][0].item<int>());
        int width1 = int(1.0*data.node_size_top[cellID][0].item<int>());
        int height0 = int(1.0*data.node_size_bot[cellID][1].item<int>());
        int height1 = int(1.0*data.node_size_top[cellID][1].item<int>());
        macro->setWidth(width0,width1);
        // macro->setHeight(dp_db.node_size_y[cellID]);
        macro->setHeight(height0,height1);
        //partition_result2[macros[i]] = partNum_valid+i;//set id

        partition_result1[macros[i]] = top_start_partvec_idx+numPart+i;//set id
        partvec2clusterID.push_back(partNum_valid+i);//numPart+i-->partNum_valid+i

        partvec2clusterID.push_back(partNum_valid+i);
        fpDatabase.macro_sequence_rotatable.push_back(partNum_valid+i);
    }
    fpDatabase.numRotatable+=macros.size();
    fpDatabase.nMacros2 = macros.size() + partNum_valid-fpDatabase.nMacros1;
    fpDatabase.nMacros = fpDatabase.nMacros1 + fpDatabase.nMacros2;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //DiePartitioner partitioner;
    vector<int> partition_result = partitioner2.merge(partition_result1,top_start_partvec_idx,partition_result2);
    //add nets
    int net_count=0;
    for (int i = 0; i < partitioner2.num_nets; ++i) {
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = partitioner2.net_list_real_end[i - 1].item<int64_t>();
        }
        int64_t end_idx = partitioner2.net_list_real_end[i].item<int64_t>();
        /* add cell pin */
        set<int> pins_tmp;
        for (int64_t idx = start_idx; idx < end_idx; idx++) {
            int pin = partitioner2.net_list_real[idx].item<int>();
            int cell_id = data.pin_id2node_id[pin].item<int>();
            // if(node_weight[cell_id].item == 0)
            // {
            //     continue;
            // }
            int partvecID = partition_result[cell_id];
            int block_id = partvec2clusterID[partvecID];
            if(block_id<0)
            {
                int debugggg=0;
            }
            pins_tmp.insert(block_id);
        }
        if(pins_tmp.size()>1)
        {
            int netID = net_count++;
            fp::Net* net = fpDatabase.addNet("net_"+to_string(i));
            for(int block_id:pins_tmp)
            {
                auto macro = fpDatabase.getMacroByID(block_id);
                net->macro_list.push_back(macro);
                net->macro_list_by_id.push_back(block_id);
            }
            fpDatabase.nets_by_id.push_back(net->macro_list_by_id);
            net->init_bx();
            fpDatabase.net_terminals.push_back(fp::pin(net->min_x_, net->min_y_,
            net->max_x_, net->max_y_));
        }else{
            int kkk=1;
        }
    }

    fpDatabase.nNets = net_count;
    logger.info("FLOORPLAN optimizing %d nets and %d cells", net_count, fpDatabase.nMacros1+fpDatabase.nMacros2);

    float ratio = 1;
    //auto [lx,ly,ux,uy] = data.die_info;
    auto die_info_a = data.die_info.accessor<float, 1>();
    torch::Tensor node_pos_lb = data.die_ll + 1 + (float)st::setting.sideline;
    torch::Tensor node_pos_ub = data.die_ll + data.die_ur - 1 - (float)st::setting.sideline;
    
    // int ux = int(die_info_a[1]);
    // int uy = int(die_info_a[3]);
    int ux = (node_pos_ub[0]-node_pos_lb[0]).item<int>();
    int uy = (node_pos_ub[1]-node_pos_lb[1]).item<int>();
    int lx = node_pos_lb[0].item<int>()+1;
    int ly = node_pos_lb[1].item<int>()+1;
    fpDatabase.outline_width=ux;
    fpDatabase.outline_height=uy;
    fpDatabase.init(ratio);
    // io time
    // fp::Floorplan fp(&fpDatabase, 0.9, "fast", false, st::setting.output_path + "/top.floorplan");
    fp::Floorplan fp(&fpDatabase, 0.8, "fast", false, st::setting.output_path + "/top.floorplan");
    fp.Run();
    //fp::Packer best_floorplan = fp.best_floorplan();
    fp::Packer best_floorplan = fp.best_floorplan_;
    if(best_floorplan.wirelength()<=0)
    {
        logger.info("fp can't find legal solution, use greedy place!");
        return false;
    }
    
    //the following codes to write floorplan output back to "data", but it seems too cumbersome
    auto node_orient_top_a = data.node_orient_top.accessor<int64_t, 1>();
    auto node_orient_bot_a = data.node_orient_bot.accessor<int64_t, 1>();
    auto node_size_a = data.node_size.accessor<float, 2>();
    auto node_size_top_a = data.node_size_top.accessor<float, 2>();
    auto node_size_bot_a = data.node_size_bot.accessor<float, 2>();
    //vector<int >
    vector<int> macros_sequence;

    for (int i = partNum_valid1; i <  fpDatabase.nMacros1; i++) macros_sequence.push_back(i);
    for (int i = partNum_valid; i < partNum_valid + macros.size(); i++) macros_sequence.push_back(i);

    // for(int i=0;i<fpDatabase.nMacros;i++)
    // {
    //     logger.info("%d, pos: %d, %d", i, 
    //     best_floorplan.macro_bounding_box_by_id_[i].first.x(),
    //     best_floorplan.macro_bounding_box_by_id_[i].first.y());
    // }
    for(int i=0;i<fpDatabase.nMacros;i++)
    {
        logger.info("%d, pos: %d, %d, name: %s, die: %d", 
        i, 
        best_floorplan.macro_bounding_box_by_id_[i].first.x(),
        best_floorplan.macro_bounding_box_by_id_[i].first.y(),
        fpDatabase.macros[i]->name().c_str(),
        best_floorplan.macro_die_[i]
        );
    }
    std::unordered_map<int, int> macros_rotated_backup;

    for (int ii = 0; ii < macros_sequence.size(); ii++) {
        const int macro_sequence = macros_sequence[ii];
        const string macro_name = fpDatabase.macros[macro_sequence]->name();
        auto bounding_box = best_floorplan.macro_bounding_box(macro_sequence);
        int die_id = best_floorplan.macro_die_[macro_sequence];
        const fp::Point& lower_left = bounding_box.first;
        const fp::Point& upper_right = bounding_box.second;
        int pos_x = lower_left.x()+lx;
        int pos_y = lower_left.y()+ly;
        const int macro_id_original = fpDatabase.macros[macro_sequence]->id;
        std::cout << "sequence: "<<macro_sequence<<" name: "<<macro_name << ' ' << "macro_id_original = "<< macro_id_original
            << " pos: "<<pos_x << ' ' << pos_y << ' '
            << best_floorplan.is_macro_rotated_by_id_.at(macro_sequence) << std::endl;
        cout<<fpDatabase.macros[macro_sequence]->width(best_floorplan.macro_die_[macro_sequence])
            <<" "<<fpDatabase.macros[macro_sequence]->height(best_floorplan.macro_die_[macro_sequence])<<" "
        <<node_size_a[macro_id_original][0]<<" "<<node_size_a[macro_id_original][1]<<endl;
        //dp_db.x[macro_id_original] = pos_x;
        //dp_db.y[macro_id_original] = pos_y;
        // lg_db_at.x[macro_id_original] = pos_x;
        // lg_db_at.y[macro_id_original] = pos_y;
        macros_rotated_backup[macro_id_original] = node_orient_top_a[macro_id_original];

        node_orient_top_a[macro_id_original] = best_floorplan.is_macro_rotated_by_id_.at(macro_sequence);
        node_orient_bot_a[macro_id_original] = best_floorplan.is_macro_rotated_by_id_.at(macro_sequence);
        node_die_a[macro_id_original] = die_id;
        logger.info("put macro %d on die %d",macro_id_original, die_id);
        int orient = node_orient_top_a[macro_sequence];
        if(macros_rotated_backup.find(macro_sequence) != macros_rotated_backup.end()) {
            orient = (orient - macros_rotated_backup.at(macro_sequence) + 40) % 4;
        }
        if(orient==1||orient==3)
        {
            float tmp_float = node_size_top_a[macro_id_original][1];
            node_size_top_a[macro_id_original][1] = node_size_top_a[macro_id_original][0];
            node_size_top_a[macro_id_original][0] = tmp_float;

            tmp_float = node_size_bot_a[macro_id_original][1];
            node_size_bot_a[macro_id_original][1] = node_size_bot_a[macro_id_original][0];
            node_size_bot_a[macro_id_original][0] = tmp_float;

            
            tmp_float = node_size_a[macro_id_original][1];
            node_size_a[macro_id_original][1] = node_size_a[macro_id_original][0];
            node_size_a[macro_id_original][0] = tmp_float;

            // tmp_float = dp_db.node_size_x[macro_id_original];
            // dp_db.node_size_x[macro_id_original] = dp_db.node_size_y[macro_id_original];
            // dp_db.node_size_y[macro_id_original] = tmp_float;
            //int tmp_int = lg_db_at.node_size_x[macro_id_original].item<int>();
            //lg_db_at.node_size_x[macro_id_original] = lg_db_at.node_size_y[macro_id_original];
            //lg_db_at.node_size_y[macro_id_original] = tmp_int;

        }
        if(die_id==0)
        {
            node_pos_a[macro_id_original][0] = pos_x + node_size_bot_a[macro_id_original][0]/2;
            node_pos_a[macro_id_original][1] = pos_y + node_size_bot_a[macro_id_original][1]/2;
            node_pos_lg_a[macro_id_original][0] = pos_x + node_size_bot_a[macro_id_original][0]/2;
            node_pos_lg_a[macro_id_original][1] = pos_y + node_size_bot_a[macro_id_original][1]/2;
            node_size_a[macro_id_original][0] = node_size_bot_a[macro_id_original][0];
            node_size_a[macro_id_original][1] = node_size_bot_a[macro_id_original][1];
        }else
        {
            node_pos_a[macro_id_original][0] = pos_x + node_size_top_a[macro_id_original][0]/2;
            node_pos_a[macro_id_original][1] = pos_y + node_size_top_a[macro_id_original][1]/2;
            node_pos_lg_a[macro_id_original][0] = pos_x + node_size_top_a[macro_id_original][0]/2;
            node_pos_lg_a[macro_id_original][1] = pos_y + node_size_top_a[macro_id_original][1]/2;
            node_size_a[macro_id_original][0] = node_size_top_a[macro_id_original][0];
            node_size_a[macro_id_original][1] = node_size_top_a[macro_id_original][1];
        }
        
    }

    auto pin_rel_cpos_a = data.pin_rel_cpos.accessor<float, 2>();
    auto pin_rel_cpos_top_a = data.pin_rel_cpos_top.accessor<float, 2>();
    auto pin_rel_cpos_bot_a = data.pin_rel_cpos_bot.accessor<float, 2>();
    auto pin_id2node_id_a = data.pin_id2node_id.accessor<int64_t, 1>();
    for (int pin_id=0;pin_id<data.num_pins;pin_id++) {
        int node_id = pin_id2node_id_a[pin_id];
        //auto& node = nodes[pin.getParNodeId()];
        int orient = node_orient_top_a[node_id];
        if(macros_rotated_backup.find(node_id) != macros_rotated_backup.end()) {
            orient = (orient - macros_rotated_backup.at(node_id) + 40) % 4;
        }
        
        if(orient<=0) continue;
        //@FIX ME: here not confirm orient 1 and 3 who is lockwise and who iscounterclockwise
        if(orient==1)
        {
            float tmp_float = pin_rel_cpos_a[pin_id][0];
            pin_rel_cpos_a[pin_id][0] = -pin_rel_cpos_a[pin_id][1];
            pin_rel_cpos_a[pin_id][1] = tmp_float;
            
            tmp_float = pin_rel_cpos_top_a[pin_id][0];
            pin_rel_cpos_top_a[pin_id][0] = -pin_rel_cpos_top_a[pin_id][1];
            pin_rel_cpos_top_a[pin_id][1] = tmp_float;
            
            tmp_float = pin_rel_cpos_bot_a[pin_id][0];
            pin_rel_cpos_bot_a[pin_id][0] = -pin_rel_cpos_bot_a[pin_id][1];
            pin_rel_cpos_bot_a[pin_id][1] = tmp_float;
        }else if(orient==2)
        {
            pin_rel_cpos_a[pin_id][0] = -pin_rel_cpos_a[pin_id][0];
            pin_rel_cpos_a[pin_id][1] = -pin_rel_cpos_a[pin_id][1];

            pin_rel_cpos_top_a[pin_id][0] = -pin_rel_cpos_top_a[pin_id][0];
            pin_rel_cpos_top_a[pin_id][1] = -pin_rel_cpos_top_a[pin_id][1];

            pin_rel_cpos_bot_a[pin_id][0] = -pin_rel_cpos_bot_a[pin_id][0];
            pin_rel_cpos_bot_a[pin_id][1] = -pin_rel_cpos_bot_a[pin_id][1];
        }else if(orient==3){
            float tmp_float = pin_rel_cpos_a[pin_id][0];
            pin_rel_cpos_a[pin_id][0] = pin_rel_cpos_a[pin_id][1];
            pin_rel_cpos_a[pin_id][1] = -tmp_float;
            
            tmp_float = pin_rel_cpos_top_a[pin_id][0];
            pin_rel_cpos_top_a[pin_id][0] = pin_rel_cpos_top_a[pin_id][1];
            pin_rel_cpos_top_a[pin_id][1] = -tmp_float;
            
            tmp_float = pin_rel_cpos_bot_a[pin_id][0];
            pin_rel_cpos_bot_a[pin_id][0] = pin_rel_cpos_bot_a[pin_id][1];
            pin_rel_cpos_bot_a[pin_id][1] = -tmp_float;
        }
    }
    
    //update std cells

    for (int i = 0; i < data.cell_mov_rhs; i++) {
        if(data.macro_mask[i].item<int>()==1)
        {
            continue;
        }
        int partvecID = partition_result[i];
        int block_id = partvec2clusterID[partvecID];
        auto bounding_box = best_floorplan.macro_bounding_box(block_id);
        const fp::Point& lower_left = bounding_box.first;
        const fp::Point& upper_right = bounding_box.second;
        int width = upper_right.x()-lower_left.x();
        int height = upper_right.y()-lower_left.y();
        int offsetX = int(gaussianKernel(0.0, 1.0)*0.2*width);
        int offsetY = int(gaussianKernel(0.0, 1.0)*0.2*height);
        // int offsetX = 0;
        // int offsetY = 0;
        int centerX = lower_left.x()+width/2+lx;
        int centerY = lower_left.y()+width/2+ly;
        node_pos_a[i][0]= centerX + offsetX;// + node_size_a[i][0];
        node_pos_a[i][1]= centerY + offsetY;// + node_size_a[i][1];
        node_pos_lg_a[i][0]= centerX + offsetX;// + node_size_a[i][0];
        node_pos_lg_a[i][1]= centerY + offsetY;// + node_size_a[i][1];
        //lg_db_at.x[i] = centerX + offsetX;
        //lg_db_at.y[i] = centerY + offsetY;
    }

    //lg_db_at.update_node_pos(node_pos_lg);
    return true;
}

}  // namespace dp