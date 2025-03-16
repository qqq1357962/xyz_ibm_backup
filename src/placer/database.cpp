#include "database.h"

tuple<Dict, shared_ptr<db::Database>, shared_ptr<gp::GPDatabase>> load_dataset() {
    Dict design_info;//@@
    shared_ptr<db::Database> rawdb;
    shared_ptr<gp::GPDatabase> gpdb;
    if (st::setting.load_from_raw) {
        logger.info("loading from original dataset...");
        GlobalParser parser;
        parser.set_single_design_params(st::setting);
        rawdb = parser.create_database();
        utils::verbose_parser_log = true;
        rawdb->load();
        rawdb->setup();
        utils::verbose_parser_log = true;
        gpdb = parser.create_gpdatabase(rawdb);
        gpdb->setup();
        if (st::setting.dataset == "iccad2022") {
            design_info = parser.preprocess_design_info_iccad2022(gpdb);
        } else
            design_info = parser.preprocess_design_info(gpdb);
    } else {
        // TODO:
        logger.info("loading from pt dataset...");
    }
    return {std::move(design_info), rawdb, gpdb};
}  // END MODULE

//---------------------------------------------------------------------

NodeData::NodeData(Dict& design_info, torch::Device device_) {
    device = device_;
    // mov_node_weights.resize(3);  // FIXME:
    // mov_node_weights = torch::empty({3}, dtype(torch::kFloat));
    hyperedge_list_cc.resize(2);
    hyperedge_list_end_cc.resize(2);

    dataset_format = db::rawDBArgs.Format;
    dataset = get<string>(design_info["dataset"]);
    // design_name = get<string>(design_info["design_name"]);
    node_type_indices = get<vector<tuple<gp::index_type, gp::index_type, string>>>(design_info["node_type_indices"]);
    node_id2node_name = get<vector<string>>(design_info["node_id2node_name"]);
    movable_index = get<tuple<int, int>>(design_info["movable_index"]);
    connected_index = get<tuple<int, int>>(design_info["connected_index"]);
    fixed_index = get<tuple<int, int>>(design_info["fixed_index"]);

    /* die info */
    die_info = get<torch::Tensor>(design_info["core_info"]);
    cout << die_info << endl;
    core_info = get<torch::Tensor>(design_info["core_info"]);  // FIXME: only used in iccad2022 contest
    rowHeights = get<torch::Tensor>(design_info["rowHeights"]);
    numRows = get<torch::Tensor>(design_info["numRows"]);
    macro_mask = get<torch::Tensor>(design_info["macro_mask"]);

    /* node/pin info */  // FIXME: see below
    node_pos = get<torch::Tensor>(design_info["node_pos"]);
    //node_rotate = get<torch::Tensor>(design_info["node_rotate"]);

    /* hyperlist info */
    pin_id2node_id = get<torch::Tensor>(design_info["pin_id2node_id"]);
    hyperedge_index = get<torch::Tensor>(design_info["hyperedge_index"]);
    hyperedge_list = get<torch::Tensor>(design_info["hyperedge_list"]);
    hyperedge_list_end = get<torch::Tensor>(design_info["hyperedge_list_end"]);
    pin_id2net_id = hyperedge_index[1];

    /* node2pin info */
    node2pin_index = get<torch::Tensor>(design_info["node2pin_index"]);
    node2pin_list = get<torch::Tensor>(design_info["node2pin_list"]);
    node2pin_list_end = get<torch::Tensor>(design_info["node2pin_list_end"]);

    /* region info */
    node_id2region_id = get<torch::Tensor>(design_info["node_id2region_id"]);
    region_boxes = get<torch::Tensor>(design_info["region_boxes"]);
    region_boxes_end = get<torch::Tensor>(design_info["region_boxes_end"]);

    /* node type info */
    movable_connected_index = make_tuple(get<0>(movable_index), get<1>(node_type_indices[0]));
    fixed_connected_index = make_tuple(get<0>(fixed_index), get<1>(connected_index));
    fixed_unconnected_index = make_tuple(get<1>(connected_index), get<1>(fixed_index));


    // fixed_connected_index = make_tuple(get<1>(connected_index), get<1>(connected_index));
    // movable_index = make_tuple(0, get<1>(connected_index));
    // auto [mov_lhs, mov_rhs] = make_tuple(get<0>(movable_index), get<1>(node_type_indices[0]));;
    // cell_mov_lhs = mov_lhs;
    // cell_mov_rhs = mov_rhs;
    // cell_movable_index = make_tuple(cell_mov_lhs, cell_mov_rhs);

    auto [mov_lhs, mov_rhs] = movable_index;
    cell_mov_lhs = mov_lhs;
    cell_mov_rhs = mov_rhs;
    iopin_mov_lhs = std::get<0>(node_type_indices[3]);
    iopin_mov_rhs = std::get<1>(node_type_indices[3]);
    for(int i=0;i<mov_rhs;i++)
    {
        if(macro_mask[i].item<int>()==1)
        {
            macro_list.push_back(i);
        }
    }
    cell_movable_index = make_tuple(cell_mov_lhs, cell_mov_rhs);

    /* site info */
    std::tie(site_width, site_height) = get<tuple<int, int>>(design_info["site_info"]);
    __ori_die_lx__ = die_info[0].item<int>();
    __ori_die_hx__ = die_info[1].item<int>();
    __ori_die_ly__ = die_info[2].item<int>();
    __ori_die_hy__ = die_info[3].item<int>();

    /* node info */
    num_nodes = node_pos.size(0);
    num_pins = pin_id2node_id.size(0);
    num_nets = hyperedge_list_end.size(0);
    
    node_to_num_pins = torch::zeros(num_nodes);
    torch::Tensor v = torch::ones(pin_id2node_id.sizes()[0]);
    node_to_num_pins.scatter_add_(0, pin_id2node_id, v);
    node_to_num_pins.unsqueeze_(1);

    /* fence info */
    num_regions = 1;
    enable_fence = false;

    /* node info */
    numTechlibs = get<int>(design_info["numTechlibs"]);
    maxUtilM = get<torch::Tensor>(design_info["maxUtilM"]);
    rowHeights = get<torch::Tensor>(design_info["rowHeights"]);
    numRows = get<torch::Tensor>(design_info["numRows"]);

    node_size_bot = get<torch::Tensor>(design_info["node_size_bot"]);
    pin_rel_cpos_bot = get<torch::Tensor>(design_info["pin_rel_cpos_bot"]);
    pin_size_bot = get<torch::Tensor>(design_info["pin_size_bot"]);
    node_orient_bot = get<torch::Tensor>(design_info["node_orient_bot"]);
    node_type = get<torch::Tensor>(design_info["node_type"]);

    /* bonding info */
    bondingCost = get<int>(design_info["bondingCost"]);
    bondingInfo = get<torch::Tensor>(design_info["bondingInfo"]);
    bonding_size = get<torch::Tensor>(design_info["bonding_size"]);
    bonding_pos = get<torch::Tensor>(design_info["bonding_pos"]);
    num_bondings = bonding_pos.size(0);
    if (numTechlibs > 1) {
        node_size_top = get<torch::Tensor>(design_info["node_size_top"]);
        pin_rel_cpos_top = get<torch::Tensor>(design_info["pin_rel_cpos_top"]);
        pin_size_top = get<torch::Tensor>(design_info["pin_size_top"]);
        node_orient_top = get<torch::Tensor>(design_info["node_orient_top"]);
    } else {
        node_size_top = get<torch::Tensor>(design_info["node_size_bot"]).clone();
        pin_rel_cpos_top = get<torch::Tensor>(design_info["pin_rel_cpos_bot"]).clone();
        pin_size_top = get<torch::Tensor>(design_info["pin_size_bot"]).clone();
        node_orient_top = get<torch::Tensor>(design_info["node_orient_bot"]).clone();

        numTechlibs = 1;
        maxUtilM[1] = maxUtilM[0];
        rowHeights[1] = rowHeights[0];
        numRows[1] = numRows[0];
    }

    at::Tensor mov_node_size_bot = node_size_bot.index({Slice(mov_lhs, mov_rhs)});
    at::Tensor mov_node_size_top = node_size_top.index({Slice(mov_lhs, mov_rhs)});
    // auto [sorted_tensor, indices] = (torch::prod(mov_node_size_bot, 1)/torch::prod(mov_node_size_top, 1)).sort();
    // float ratio_pre=-1;
    // for(int i=0;i<mov_node_size_bot.size(0);i++)
    // {
    //     float tmp_float = sorted_tensor[i].item<float>();
    //     if(tmp_float==ratio_pre)
    //     {
    //         continue;
    //     }
    //     ratio_pre = tmp_float;
    //     cout<<"ratio "<<i<<": "<<ratio_pre<<endl;
    // }
    at::Tensor mov_cell_area_bot = torch::sum(torch::prod(mov_node_size_bot, 1));
    at::Tensor mov_cell_area_top = torch::sum(torch::prod(mov_node_size_top, 1));

    at::Tensor mov_node_sizes = torch::cat({mov_node_size_bot.unsqueeze(0), mov_node_size_top.unsqueeze(0)}, 0);
    at::Tensor mov_node_areas = torch::cat({mov_cell_area_bot.unsqueeze(0), mov_cell_area_top.unsqueeze(0)}, 0);

    die_ur = die_info.reshape({2, 2}).t()[1].clone();
    die_ll = die_info.reshape({2, 2}).t()[0].clone();
    at::Tensor die_area = at::prod(die_ur - die_ll);

    logger.info("Move cell total areas [%.4E, %.4E]", mov_cell_area_bot.item<float>(), mov_cell_area_top.item<float>());

    /* preprocess for multi lib */
    node_ratio = mov_cell_area_bot / mov_cell_area_top;
    tech_ratio = mov_cell_area_top / (mov_cell_area_top + mov_cell_area_bot);  // a * bot + ( 1 - a ) * top
    double estimated_target_density_bot = (tech_ratio * mov_cell_area_bot / die_area).item<float>();
    double estimated_target_density_top = (((1 - tech_ratio) * mov_cell_area_top) / die_area).item<float>();
    actualUtilM = maxUtilM.clone();
    actualUtilM[0] = estimated_target_density_bot;
    actualUtilM[1] = estimated_target_density_top;
    maxUtilM = actualUtilM + 0.05;

    upper_lower_bound_ratio = maxUtilM.clone();
    auto area_upper = (node_ratio * maxUtilM[0]) / (mov_node_areas[0] / die_area - maxUtilM[0]);
    auto area_lower = (mov_node_areas[1] / die_area - maxUtilM[1]) / (1 / node_ratio * maxUtilM[1]);
    upper_lower_bound_ratio[0] = area_lower; // A_bot / A_top
    upper_lower_bound_ratio[1] = area_upper;

    logger.info("Valid area ratio of two chips [%.3f ~ %.3f]", upper_lower_bound_ratio[0].item<float>(), upper_lower_bound_ratio[1].item<float>());

    if (!(maxUtilM[0] == maxUtilM[1]).item<bool>() && (st::setting.force_coeff_2d == 0)) {
    // if (st::setting.omni_int) {
        logger.info("Utilization clamped to the chip with smaller footprint");
        /* preprocess max util */
        float estimated_target_density = (tech_ratio * mov_cell_area_bot / die_area).item<float>();
        // int minUtil_idx = torch::argmin(maxUtilM).item<int>();
        // float minUtil = std::min(maxUtilM.min().item<float>(), estimated_target_density);
        
        int minUtil_idx = torch::argmin(maxUtilM).item<int>();
        float minUtil = maxUtilM.min().item<float>();

        // int minUtil_idx = st::setting.stack_cells;
        // float minUtil = maxUtilM[minUtil_idx].item<float>();

        float maxUtil = (mov_node_areas[1 - minUtil_idx] / die_area - minUtil * mov_node_areas[1 - minUtil_idx] / mov_node_areas[minUtil_idx]).item<float>();
        actualUtilM[minUtil_idx] = minUtil;
        actualUtilM[1 - minUtil_idx] = maxUtil;
        torch::Tensor actualArea = (actualUtilM[0] + actualUtilM[1]) * die_area;

        tech_ratio = (actualArea - mov_node_areas[1]) / (mov_node_areas[0] - mov_node_areas[1]);
        node_ratio = (1 - tech_ratio) / tech_ratio;

        // logger.info("Utilization clamped to the chip with smaller footprint");
        // /* preprocess max util */
        // int minHeight_idx = torch::argmin(rowHeights).item<int>();

        // actualUtilM[minHeight_idx] = maxUtilM[minHeight_idx];
        // actualUtilM[1 - minHeight_idx] = (mov_node_areas[1 - minHeight_idx] / die_area) 
        //                                 - (maxUtilM[1 - minHeight_idx] * mov_node_areas[1 - minHeight_idx] / mov_node_areas[minHeight_idx]);
        // torch::Tensor actualArea = (actualUtilM[0] + actualUtilM[1]) * die_area;
        // tech_ratio = (actualArea - mov_node_areas[1]) / (mov_node_areas[0] - mov_node_areas[1]);
        // node_ratio = (1 - tech_ratio) / tech_ratio;
    } else {
        logger.info("Utilization balanced");
        auto area_balance = (area_lower + area_upper) / 2;
        tech_ratio = (area_balance * mov_node_areas[1]) / (mov_node_areas[0] + area_balance * mov_node_areas[1]);
        node_ratio = (1 - tech_ratio) / tech_ratio;

        // actualUtilM[0] = mov_node_areas[1] / (1 / area_balance + mov_node_areas[1] / mov_node_areas[0]) / die_area;
        // actualUtilM[1] = mov_node_areas[0] / (area_balance + mov_node_areas[0] / mov_node_areas[1]) / die_area;
    }
    logger.info("Estimated utilization [%.4f, %.4f]", actualUtilM[0].item<float>(), actualUtilM[1].item<float>());
    // node_die = torch::ones({num_nodes}, dtype(torch::kFloat)) * (1 - st::setting.stack_cells) * 0.8
    //         + 0.2 * torch::rand({num_nodes}, dtype(torch::kFloat));
    // node_die = torch::ones({num_nodes}, dtype(torch::kFloat)) * (1 - st::setting.stack_cells);

    // if (estimated_target_density_bot > maxUtilM[0].item<float>()) {
    //     tech_ratio = (maxUtilM[0] * die_area / mov_cell_area_bot) - 0.01;
    // } else if (estimated_target_density_top > maxUtilM[1].item<float>()) {
    //     tech_ratio = 1 - (maxUtilM[1] * die_area / mov_cell_area_top) + 0.01;
    // }
    // estimated_target_density_bot = (tech_ratio * mov_cell_area_bot / die_area).item<float>();
    // estimated_target_density_top = (((1 - tech_ratio) * mov_cell_area_top) / die_area).item<float>();

    // area_ratio = estimated_target_density_top / estimated_target_density_bot; // FIXME:

    mov_cell_area_bot_bound = ((maxUtilM[0] - 0.001) * die_area).item<int64_t>();
    mov_cell_area_top_bound = ((maxUtilM[1] - 0.001) * die_area).item<int64_t>();
    logger.info("max util_bot: %f, max util_top: %f, bot bound: %lld, top bound: %lld", 
                    maxUtilM[0].item<float>(), maxUtilM[1].item<float>(), (long long)mov_cell_area_bot_bound, (long long)mov_cell_area_top_bound);

    max_mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    max_mov_cell_areas[0] = mov_cell_area_bot_bound;
    max_mov_cell_areas[1] = mov_cell_area_top_bound;

    // double target_density = ((tech_ratio * mov_cell_area_bot + (1 - tech_ratio) * mov_cell_area_top) / die_area)
    //                             .item<float>();  // FIXME:

    // assert_msg(estimated_target_density_bot < maxUtilM[0].item<float>(), "Util bot exceeds");
    // assert_msg(estimated_target_density_top < maxUtilM[1].item<float>(), "Util top exceeds");

    // st::setting.target_density = target_density;
    // FIXME: just set target_density = 2
    // TODO: better ratio
    // tech_ratio = at::full({1}, 0); //@@
    node_size = (tech_ratio * node_size_bot + (1 - tech_ratio) * node_size_top);
    // for(auto macro_id:macro_list)
    // {
    //     node_size[macro_id] = node_size_top[macro_id];
    // }
    pin_rel_cpos = tech_ratio * pin_rel_cpos_bot + (1 - tech_ratio) * pin_rel_cpos_top;
    
    // for(auto macro_id:macro_list)
    // {
    //     int start_idx =0;
    //     if(macro_id>0)
    //     {
    //         start_idx = node2pin_list_end[macro_id-1].item<int>();
    //     }
    //     int end_idx = node2pin_list_end[macro_id].item<int>();
    //     for(int ii=start_idx;ii<end_idx;ii++)
    //     {
    //         int pin_id = node2pin_list[ii].item<int>();
    //         pin_rel_cpos[pin_id] = pin_rel_cpos_top[pin_id];
    //     }
    // }
    pin_size = tech_ratio * pin_size_bot + (1 - tech_ratio) * pin_size_top;
    node_size_selector = torch::cat({node_size_bot, node_size_top}, 0);

    // node_size = node_size_top.clone();
    // pin_rel_cpos = pin_rel_cpos_top;
    // pin_size = pin_size_top.clone();

    mov_cell_util = torch::sum(torch::prod(node_size, 1)) / die_area / st::setting.target_density;

    logger.info("Tech/Node Ratio: %.2f, %.2f", tech_ratio.item<float>(), node_ratio.item<float>());
    logger.info("Row Height = %d/%d", rowHeights[0].item<int>(), rowHeights[1].item<int>());

    node_area_bot = torch::prod(node_size_bot, 1);
    node_area_top = torch::prod(node_size_top, 1);

    if (st::setting.eval_params) {
        // st::setting.target_density = 1;
        // st::setting.lg = true;
        // st::setting.dp = false;
        // st::setting.dataset = "iccad2022_one_die";
        // int mat_c = 2;
        // die_info *= mat_c;
        // core_info *= mat_c;
        // numRows = numRows[1] * mat_c;
        // rowHeights = rowHeights[1];
        // node_size = node_size_top.clone();
        // pin_rel_cpos = pin_rel_cpos_top.clone();
        // pin_size = pin_size_top.clone();
    }
    if (st::setting.site_width > 0) site_width = st::setting.site_width;


    aspect_ratio = (node_size.index({Slice(mov_lhs, mov_rhs), 0}) / node_size.index({Slice(mov_lhs, mov_rhs), 1}));
    auto aspect_ratio_bot = (node_size_bot.index({Slice(mov_lhs, mov_rhs), 0}) / node_size_bot.index({Slice(mov_lhs, mov_rhs), 1}));
    auto aspect_ratio_top = (node_size_top.index({Slice(mov_lhs, mov_rhs), 0}) / node_size_top.index({Slice(mov_lhs, mov_rhs), 1}));
    int count = 0;
    cout << "==================================== #pins ====================================\n";
    auto long_cells = torch::_cast_Float(aspect_ratio > 6);
    auto pin_cells = torch::_cast_Float(node_to_num_pins > 5);
    // stack_cells = torch::logical_and(node_to_num_pins.squeeze(1) >= 5, aspect_ratio > 6);
    // stack_cells = torch::logical_or(node_to_num_pins.squeeze(1) >= 5, aspect_ratio > 6);
    stack_cells = node_to_num_pins.squeeze(1) >= 5;

    logger.info("%d cells are stacked to chip %d", torch::_cast_Int(stack_cells).sum().item<int>(), st::setting.stack_cells);

    cout << "pins/area: " << node_to_num_pins.sum().item<int>() / mov_cell_area_bot.item<float>() << endl;
    cout << "long cells: " << long_cells.sum().item<int>() << " cells are long\n";
    cout << "5 pin cells: " << (torch::_cast_Float(node_to_num_pins >= 5)).sum().item<int>() << endl;
    cout << "4 pin cells: " << (torch::_cast_Float(node_to_num_pins >= 4)).sum().item<int>() << endl;
    cout << "3 pin cells: " << (torch::_cast_Float(node_to_num_pins >= 3)).sum().item<int>() << endl;
    
    cout << "avg #pins: " << node_to_num_pins.mean().item<float>() << endl;
    cout << "max #pins: " << node_to_num_pins.max().item<float>() << endl;
    cout << "avg long #pins: " << ((node_to_num_pins.squeeze() * long_cells).sum() / long_cells.sum()).item<float>() << endl;
    cout << "max long #pins: " << (node_to_num_pins.squeeze() * long_cells).max().item<float>() << endl;

    cout << "==================================== size ====================================\n";
    auto enlarge = torch::prod(node_size_bot, 1) / torch::prod(node_size_top, 1);
    cout << "size enlarged: " << enlarge.max().item<float>() << endl;
    cout << "size enlarged: " << enlarge.mean().item<float>() << endl;
    cout << "size enlarged: " << enlarge.min().item<float>() << endl;
    // node_size.index_put_({Slice(get<0>(fixed_index), get<1>(connected_index)), 0}, rowHeights);
    // node_size.index_put_({Slice(get<0>(fixed_index), get<1>(connected_index)), 1}, rowHeights);

    /* bin info */  // TODO:
    num_bin_x = st::setting.num_bin_x;
    num_bin_y = st::setting.num_bin_y;
    clamp_node = st::setting.clamp_node;
    target_density = st::setting.target_density;
    
    macro_neighbors.resize(macro_list.size());
    int search_depth=2;
    vector<int> visited;
    visited.resize(cell_mov_rhs);
    // cout << macro_list << endl;
    // for(int i=0;i<macro_list.size();i++)
    // {
    //     std::fill(visited.begin(), visited.end(), 0);
    //     int macro_id = macro_list[i];
    //     queue<pair<int,int> > q;
    //     q.push(make_pair(macro_id,0));
    //     while(!q.empty())
    //     {
    //         int node_id = q.front().first;
    //         int layer = q.front().second;
    //         // cout<<"visiting "<<node_id <<endl;
    //         q.pop();
    //         if(visited[node_id])
    //         {
    //             continue;
    //         }
    //         visited[node_id]=1;
    //         if(node_id!=macro_id)
    //         {
    //             macro_neighbors[i].push_back(node_id);
    //         }
    //         if(layer>=search_depth)
    //         {
    //             continue;
    //         }
    //         int start_idx=0;
    //         if(node_id>0)
    //         {
    //             start_idx = node2pin_list_end[node_id - 1].item<int>();
    //         }
    //         int end_idx = node2pin_list_end[node_id].item<int>();
    //         for(int i=start_idx;i<end_idx;i++)
    //         {
    //             int pin_id = node2pin_list[i].item<int>();
    //             int net_id = pin_id2net_id[pin_id].item<int>();
    //             int start_idx_net = 0;
    //             if(net_id>0)
    //             {
    //                 start_idx_net = hyperedge_list_end[net_id-1].item<int>();
    //             }
    //             int end_idx_net = hyperedge_list_end[net_id].item<int>();
    //             for(int j=start_idx_net;j<end_idx_net;j++)
    //             {
    //                 int pin_id_next = hyperedge_list[j].item<int>();
    //                 if(pin_id==pin_id_next)
    //                 {
    //                     continue;
    //                 }
    //                 int next_id = pin_id2node_id[pin_id_next].item<int>();
    //                 q.push(make_pair(next_id,layer+1));
    //             }            
    //         }
    //     }
    // }
    
    pin_direction.resize(macro_list.size());
    pin_rel_pos_mean.resize(macro_list.size());
    
    for(int i=0;i<macro_list.size();i++)
    {
        int macro_id = macro_list[i];
        int start_idx=0;
        if(macro_id>0)
        {
            start_idx = node2pin_list_end[macro_id - 1].item<int>();
        }
        int end_idx = node2pin_list_end[macro_id].item<int>();
        float x_total=0;
        float y_total=0;
        int num_pin = end_idx-start_idx+1;
        for(int i=start_idx;i<end_idx;i++)
        {
            int pin_id = node2pin_list[i].item<int>();
            float pin_x = pin_rel_cpos_bot[pin_id][0].item<float>();
            float pin_y = pin_rel_cpos_bot[pin_id][1].item<float>();
            x_total+=pin_x;
            y_total+=pin_y;
        }
        pin_rel_pos_mean[i] = make_pair(x_total/num_pin, y_total/num_pin);
        x_total/=node_size_bot[macro_id][0].item<float>();
        y_total/=node_size_bot[macro_id][1].item<float>();
        if(x_total>=y_total) {
            if(x_total> -y_total){
                pin_direction[i]=0;
            }else{
                pin_direction[i]=1;
            }
        }else {
            if(x_total> -y_total){
                pin_direction[i]=2;
            }else{
                pin_direction[i]=3;
            }
        }
    }
    
}  // END MODULE

//---------------------------------------------------------------------
void NodeData::setMacroOrient() {
    int core_dim = core_info[1].item().toInt() >= core_info[3].item().toInt() ? 0 : 1;
    // auto node_orient_top_a = node_orient_top.accessor<int64_t, 1>();
    // auto node_orient_bot_a = node_orient_bot.accessor<int64_t, 1>();
    // auto node_size_a = node_size.accessor<float, 2>();
    // auto node_size_top_a = node_size_top.accessor<float, 2>();
    // auto node_size_bot_a = node_size_bot.accessor<float, 2>();
    auto new_orient = node_orient_top.clone();

    std::unordered_map<int, int> type_occurance;

    for(int i = 0 ;i < num_nodes; i++) {
        if (macro_mask[i].item<int>() == 1) {
            int macro_dim = node_size[i][0].item().toInt() >= node_size[i][1].item().toInt() ? 0 : 1;
            if(core_dim != macro_dim) {
                new_orient[i]=1;
                // node_orient_top[i] = 1;
                // node_orient_bot[i] = 1;
                // float tmp_float = node_size_top_a[i][1];
                // node_size_top_a[i][1] = node_size_top_a[i][0];
                // node_size_top_a[i][0] = tmp_float;

                // tmp_float = node_size_bot_a[i][1];
                // node_size_bot_a[i][1] = node_size_bot_a[i][0];
                // node_size_bot_a[i][0] = tmp_float;

            
                // tmp_float = node_size_a[i][1];
                // node_size_a[i][1] = node_size_a[i][0];
                // node_size_a[i][0] = tmp_float;
            }
            
            int node_celltypeID = node_type[i].item<int>();
            if(type_occurance.find(node_celltypeID) != type_occurance.end()) {
                if(type_occurance[node_celltypeID] %2 == 1) {
                    // std::cout << i << " " << node_orient_top[i].item<int>() << "\n";
                    // std::cout << i << " " << new_orient[i].item<int>() << "\n";
                    // node_orient_top_a[i] += 2;
                    // node_orient_bot_a[i] += 2;
                    new_orient[i] += 2;
                    // std::cout << i << " " << node_orient_top[i].item<int>() << "\n";
                    // std::cout << i << " " << new_orient[i].item<int>() << "\n";
                }
                type_occurance[node_celltypeID] = type_occurance[node_celltypeID] + 1;
            } else {
                type_occurance[node_celltypeID] = 1;
            }
            
        }
    }
    update_macro_orientation(new_orient);
    // auto pin_rel_cpos_a = pin_rel_cpos.accessor<float, 2>();
    // auto pin_rel_cpos_top_a = pin_rel_cpos_top.accessor<float, 2>();
    // auto pin_rel_cpos_bot_a = pin_rel_cpos_bot.accessor<float, 2>();
    // auto pin_id2node_id_a = pin_id2node_id.accessor<int64_t, 1>();

    // for (int pin_id=0;pin_id<num_pins;pin_id++) {
    //     int node_id = pin_id2node_id_a[pin_id];
    //     //auto& node = nodes[pin.getParNodeId()];
    //     int orient = node_orient_top[node_id].item<int>();
    //     if(orient<=0) continue;
    //     //@FIX ME: here not confirm orient 1 and 3 who is lockwise and who iscounterclockwise
    //     if(orient==1)
    //     {
    //         float tmp_float = pin_rel_cpos_a[pin_id][0];
    //         pin_rel_cpos_a[pin_id][0] = -pin_rel_cpos_a[pin_id][1];
    //         pin_rel_cpos_a[pin_id][1] = tmp_float;
            
    //         tmp_float = pin_rel_cpos_top_a[pin_id][0];
    //         pin_rel_cpos_top_a[pin_id][0] = -pin_rel_cpos_top_a[pin_id][1];
    //         pin_rel_cpos_top_a[pin_id][1] = tmp_float;
            
    //         tmp_float = pin_rel_cpos_bot_a[pin_id][0];
    //         pin_rel_cpos_bot_a[pin_id][0] = -pin_rel_cpos_bot_a[pin_id][1];
    //         pin_rel_cpos_bot_a[pin_id][1] = tmp_float;
    //     }else if(orient==2)
    //     {
    //         pin_rel_cpos_a[pin_id][0] = -pin_rel_cpos_a[pin_id][0];
    //         pin_rel_cpos_a[pin_id][1] = -pin_rel_cpos_a[pin_id][1];

    //         pin_rel_cpos_top_a[pin_id][0] = -pin_rel_cpos_top_a[pin_id][0];
    //         pin_rel_cpos_top_a[pin_id][1] = -pin_rel_cpos_top_a[pin_id][1];

    //         pin_rel_cpos_bot_a[pin_id][0] = -pin_rel_cpos_bot_a[pin_id][0];
    //         pin_rel_cpos_bot_a[pin_id][1] = -pin_rel_cpos_bot_a[pin_id][1];
    //     }else if(orient==3){
    //         float tmp_float = pin_rel_cpos_a[pin_id][0];
    //         pin_rel_cpos_a[pin_id][0] = pin_rel_cpos_a[pin_id][1];
    //         pin_rel_cpos_a[pin_id][1] = -tmp_float;
            
    //         tmp_float = pin_rel_cpos_top_a[pin_id][0];
    //         pin_rel_cpos_top_a[pin_id][0] = pin_rel_cpos_top_a[pin_id][1];
    //         pin_rel_cpos_top_a[pin_id][1] = -tmp_float;
            
    //         tmp_float = pin_rel_cpos_bot_a[pin_id][0];
    //         pin_rel_cpos_bot_a[pin_id][0] = pin_rel_cpos_bot_a[pin_id][1];
    //         pin_rel_cpos_bot_a[pin_id][1] = -tmp_float;
    //     }
    // }
}

//---------------------------------------------------------------------

void NodeData::backup_ori_var() {
    __ori_die_info__ = die_info.clone().cpu();
    __ori_core_info__ = core_info.clone().cpu();
    __ori_node_pos__ = node_pos.clone().cpu();
    //__ori_node_rotate__ = node_rotate.clone().cpu();
    __ori_node_size__ = node_size.clone().cpu();
    __ori_pin_size__ = pin_size.clone().cpu();
    __ori_region_boxes__ = region_boxes.clone().cpu();
    // device = die_info.device();
    __die_shift__ = torch::tensor({0.0, 0.0}, torch::dtype(die_info.dtype()));
    __die_scale__ = torch::tensor({1.0, 1.0}, torch::dtype(die_info.dtype()));

    /* net info */
    __ori_pin_id2node_id__ = pin_id2node_id.clone();
    __ori_pin_id2net_id__ = pin_id2net_id.clone();
    __ori_hyperedge_list__ = hyperedge_list.clone();
    __ori_hyperedge_list_end__ = hyperedge_list_end.clone();
    __ori_pin_rel_cpos__ = pin_rel_cpos.clone();

    __ori_node_size_bot__ = node_size_bot.numel() ? node_size_bot.clone() : node_size.clone();
    __ori_node_size_top__ = node_size_top.numel() ? node_size_top.clone() : node_size.clone();

    __ori_node_orient_bot__ = node_orient_bot.clone();
    __ori_node_orient_top__ = node_orient_top.clone();
}  // END MODULE

//---------------------------------------------------------------------

void NodeData::preshift() {
    // shift die info to (0.0, hx, 0.0, hy)
    torch::Tensor die_shift = torch::tensor({__ori_die_lx__, __ori_die_ly__}, torch::dtype(die_info.dtype()));
    die_info = (die_info.reshape({2, 2}).t() - die_shift).t().reshape(-1);
    node_pos -= die_shift;
    __die_shift__ += die_shift;
}  // END MODULE

//---------------------------------------------------------------------

void NodeData::prescale_by_site_width() {
    logger.info("design scaled by /%d", site_width);
    // inplace scaling
    die_info /= site_width;
    core_info /= site_width;
    rowHeights /= site_width;
    region_boxes /= site_width;
    node_pos /= site_width;
    node_size /= site_width;
    pin_rel_cpos /= site_width;
    pin_size /= site_width;

    if (bondingInfo.numel()) bondingInfo /= site_width;
    if (node_size_bot.numel()) node_size_bot /= site_width;
    if (node_size_top.numel()) node_size_top /= site_width;
    if (pin_rel_cpos_bot.numel()) pin_rel_cpos_bot /= site_width;
    if (pin_rel_cpos_top.numel()) pin_rel_cpos_top /= site_width;
    if (pin_size_bot.numel()) pin_size_bot /= site_width;
    if (pin_size_top.numel()) pin_size_top /= site_width;

    __die_scale__ *= site_width;
}  // END MODULE

//---------------------------------------------------------------------

void NodeData::postscale_by_site_width() {
    // inplace scaling
    logger.info("design scaled by %d", site_width);
    die_info *= site_width;
    core_info *= site_width;
    rowHeights *= site_width;
    region_boxes *= site_width;
    node_pos *= site_width;
    node_size *= site_width;
    pin_rel_cpos *= site_width;
    pin_size *= site_width;

    if (bondingInfo.numel()) bondingInfo *= site_width;
    if (node_size_bot.numel()) node_size_bot *= site_width;
    if (node_size_top.numel()) node_size_top *= site_width;
    if (pin_rel_cpos_bot.numel()) pin_rel_cpos_bot *= site_width;
    if (pin_rel_cpos_top.numel()) pin_rel_cpos_top *= site_width;
    if (pin_size_bot.numel()) pin_size_bot *= site_width;
    if (pin_size_top.numel()) pin_size_top *= site_width;

    __die_scale__ /= site_width;
    hpwl_scale = __die_scale__;
    // site_width = 1;
}  // END MODULE

//---------------------------------------------------------------------

void NodeData::prescale() {
    //  scale die info to (0.0, 1.0, 0.0, 1.0)
    int die_lx = die_info[0].item().toInt();
    int die_hx = die_info[1].item().toInt();
    int die_ly = die_info[2].item().toInt();
    int die_hy = die_info[3].item().toInt();
    torch::Tensor die_scale = torch::tensor({die_hx - die_lx, die_hy - die_ly}, torch::dtype(die_info.dtype()));
    node_pos /= die_scale;
    node_size /= die_scale;
    pin_rel_cpos /= die_scale;
    pin_size /= die_scale;
    die_info = (die_info.reshape({2, 2}).t() / die_scale).t().reshape(-1);
    __die_scale__ *= die_scale;
}  // END MODULE

//---------------------------------------------------------------------

bool NodeData::check_design() { return true; }  // END MODULE

//---------------------------------------------------------------------

void NodeData::pre_compute_var() {
    // die related
    float lx = die_info[0].item<float>();
    float hx = die_info[1].item<float>();
    float ly = die_info[2].item<float>();
    float hy = die_info[3].item<float>();
    unit_len =
        torch::tensor({(hx - lx) / double(num_bin_x), (hy - ly) / double(num_bin_y)}, torch::dtype(torch::kFloat32));
    die_ur = die_info.reshape({2, 2}).t()[1].clone();
    die_ll = die_info.reshape({2, 2}).t()[0].clone();
    core_ur = core_info.reshape({2, 2}).t()[1].clone();
    core_ll = core_info.reshape({2, 2}).t()[0].clone();
    hpwl_scale = __die_scale__ / site_width;
    // node related
    node_area = torch::prod(node_size, 1).unsqueeze(1);

    // node_to_num_pins = torch::zeros(num_nodes);
    // torch::Tensor v = torch::ones(pin_id2node_id.sizes()[0]);
    // node_to_num_pins.scatter_add_(0, pin_id2node_id, v);
    // node_to_num_pins.unsqueeze_(1);
    // net related
    torch::Tensor start_idx = hyperedge_list_end.roll(1);
    start_idx[0] = 0;
    net_to_num_pins = hyperedge_list_end - start_idx;
    net_mask = torch::logical_and(net_to_num_pins <= st::setting.ignore_net_degree,
                                  net_to_num_pins >= 2);  // 0: ignore, 1: consider in wirelength calculation

    torch::Tensor w = torch::ones(net_to_num_pins.sizes()[0]);
    pin_num_count = torch::zeros(net_to_num_pins.max().item<int>() + 1);
    pin_num_count.scatter_add_(0, torch::_cast_Long(net_to_num_pins), w);

#ifdef DEBUG
    if (true) {
        plt::backend("Agg");
        int n = min(101, net_to_num_pins.max().item<int>() + 1);
        std::vector<double> x(n), y(n);
        for (int i = 0; i < n - 1; ++i) {
            x.at(i) = i;
            y.at(i) = pin_num_count[i].item<int>();
        }
        x.at(n - 1) = n - 1;
        y.at(n - 1) = pin_num_count.index({Slice(100, None)}).sum().item<int>();

        plt::plot(x, y);

        std::filesystem::path current_dir(std::filesystem::current_path());
        std::filesystem::path result_dir(st::setting.result_dir);
        std::filesystem::path exp_id(st::setting.exp_id);
        std::filesystem::path res_root = current_dir / result_dir / exp_id;
        std::string fig_path = res_root.string() + "/" + "_net_pin_num.png";
        plt::save(fig_path);
        plt::close();
    }
    plt::close();
#endif  // DEBUG

    // obj related
    auto [mov_lhs, mov_rhs] = movable_index;
    mov_cell_area = torch::prod(node_size.index({torch::indexing::Slice(mov_lhs, mov_rhs), "..."}), 1);
    __total_mov_area_without_filler__ = torch::sum(mov_cell_area).item<float>();
    bin_area = torch::prod(unit_len).item<float>();
}  // END MODULE

//---------------------------------------------------------------------

bool NodeData::init_fence_region() { return true; }

void NodeData::logging_statistics() {
    logger.info("=========================================================");
    logger.info("#nodes = %d, #nets = %d, #pins = %d", num_nodes, num_nets, num_pins);
    int num_conmov_nodes = get<1>(node_type_indices[0]) - get<0>(node_type_indices[0]);
    int num_fltmov_nodes = get<1>(node_type_indices[1]) - get<0>(node_type_indices[1]);
    int num_confix_nodes = get<1>(node_type_indices[2]) - get<0>(node_type_indices[2]);
    int num_fltfix_nodes = get<1>(node_type_indices[6]) - get<0>(node_type_indices[6]);
    int num_coniopin = get<1>(node_type_indices[3]) - get<0>(node_type_indices[3]);
    int num_fltiopin = get<1>(node_type_indices[5]) - get<0>(node_type_indices[5]);
    int num_blkg = get<1>(node_type_indices[4]) - get<0>(node_type_indices[4]);
    logger.info("#ConnMov = %d, #FloatMov = %d, #ConnFix = %d, #FloatFix = %d, #ConnIOPin = %d, #FloatIOPin = %d",
                num_conmov_nodes, num_fltmov_nodes, num_confix_nodes, num_fltfix_nodes, num_coniopin, num_fltiopin);
    int lx = die_info[0].item().toInt();
    int hx = die_info[1].item().toInt();
    int ly = die_info[2].item().toInt();
    int hy = die_info[3].item().toInt();
    logger.info("Die Info %d %d %d %d", lx, hx, ly, hy);
    logger.info("Core Info %d %d %d %d", core_info[0].item().toInt(), core_info[1].item().toInt(),
                core_info[2].item().toInt(), core_info[3].item().toInt());
    logger.info("Site Width = %d, Row Height = %d", site_width, site_height);
    logger.info("#Bins = (%d, %d), UnitLen = (%.2f, %.2f)", num_bin_x, num_bin_y, unit_len[0].item<float>(),
                unit_len[1].item<float>());
    logger.info("target density = %.2f", target_density);
    logger.info("cell area = %.2f", __total_mov_area_without_filler__);
    logger.info("node[0] = (%.2f, %.2f)", node_size[0][0].item<float>(), node_size[0][1].item<float>());
    for (auto node_type_indice : node_type_indices) {
        auto [a, b, c] = node_type_indice;
        logger.info("node type %s: %d %d", c.c_str(), a, b);
    }

    logger.info("#Bonding = (%d)", num_bondings);
    logger.info("Bonding Size (%d, %d, %d)", bondingInfo[0].item().toInt(), bondingInfo[1].item().toInt(),
                bondingInfo[2].item().toInt());
    logger.info("Utils (%.2f, %.2f, %.2f)", maxUtilM[0].item<float>(), maxUtilM[1].item<float>(),
                mov_cell_util.item<float>());
    logger.info("NumRows = (%d, %d)", numRows[0].item<int>(), numRows[1].item<int>());

    logger.info("=========================================================");
}  // END MODULE

//---------------------------------------------------------------------

void NodeData::preprocess() {
    backup_ori_var();
    preshift();
    prescale_by_site_width();
    if (st::setting.scale_design) prescale();
    check_design();
    pre_compute_var();
    init_fence_region();
    logging_statistics();
}  // END MODULE

//---------------------------------------------------------------------

tuple<at::Tensor, at::Tensor, at::Tensor> NodeData::get_mov_node_info_dummy_via() {
    /* update hyperedge lists */
    /* via_mov_node | cell_node */
    // pin_id2node_id -> cells_pin | vias_pin
    // hyperedge_list -> net_pin | via
    // hyperedge_list_end -> hyperedge_list_end + 1

    pin_id2node_id = torch::zeros(num_pins + num_nets, torch::dtype(pin_id2node_id.dtype()));
    hyperedge_list = torch::zeros(num_pins + num_nets, torch::dtype(hyperedge_list.dtype()));
    hyperedge_list_end = torch::zeros_like(hyperedge_list_end);
    /* update hyperedge lists */
    index_type ptr = 0;
    index_type last_idx = 0;
    for (int i = 0; i < num_nets; ++i) {
        index_type start_idx = 0;
        if (i != 0) {
            start_idx = __ori_hyperedge_list_end__[i - 1].item<long>();
        }
        index_type end_idx = __ori_hyperedge_list_end__[i].item<long>();
        /* add cell pin */
        for (index_type idx = start_idx; idx < end_idx; idx++) {
            hyperedge_list[ptr] = __ori_hyperedge_list__[idx];
            ptr++;
        }
        /* add via */
        hyperedge_list[ptr] = num_pins + i;
        ptr++;
        last_idx += end_idx - start_idx + 1;
        hyperedge_list_end[i] = last_idx;
    }
    /* update pin_id */
    /* #pins -> #nets + #nodes */
    for (int i = 0; i < num_pins + num_nets; ++i) {
        if (i < num_pins) {
            /* net pins */
            pin_id2node_id[i] = __ori_pin_id2node_id__[i];  // FIXME: list length
        } else {
            /* via pins */
            pin_id2node_id[i] = (i - num_pins + num_nodes);
        }
    }
    auto via_pin_rel_cpos = torch::zeros({num_nets, 2}, torch::dtype(pin_rel_cpos.dtype()));
    pin_rel_cpos = torch::cat({pin_rel_cpos, via_pin_rel_cpos}, 0);

    /* move node info */
    auto [mov_lhs, mov_rhs] = movable_index;
    at::Tensor via_node_pos = torch::zeros({num_nets, 2}, torch::dtype(node_size.dtype()));
    at::Tensor via_node_size = torch::ones({num_nets, 2}, torch::dtype(node_size.dtype()));
    via_node_size.index({"...", 0}) *= (bondingInfo[0] + bondingInfo[2]);
    via_node_size.index({"...", 1}) *= (bondingInfo[1] + bondingInfo[2]);
    at::Tensor via_node_area = torch::prod(via_node_size, 1).unsqueeze(1);
    at::Tensor via_node_to_num_pins = torch::ones({num_nets, 1}, torch::dtype(mov_node_to_num_pins.dtype()));

    at::Tensor mov_node_pos = torch::cat({node_pos.index({Slice(mov_lhs, mov_rhs)}).clone(), via_node_pos}, 0);
    at::Tensor mov_node_size = torch::cat({node_size.index({Slice(mov_lhs, mov_rhs)}).clone(), via_node_size}, 0);
    mov_node_area = torch::cat({node_area.index({Slice(mov_lhs, mov_rhs)}).clone(), via_node_area}, 0);
    mov_node_to_num_pins =
        torch::cat({node_to_num_pins.index({Slice(mov_lhs, mov_rhs)}).clone(), via_node_to_num_pins}, 0);
    movable_index = make_tuple(mov_lhs, mov_rhs + num_nets);  // FIXME:

    at::Tensor scale = (die_ur - die_ll) * 0.001;
    at::Tensor loc = (die_ur + die_ll) * 0.5;
    mov_node_pos = torch::randn_like(mov_node_pos) * scale + loc;

    // rand mov node pos
    std::tie(mov_lhs, mov_rhs) = movable_index;
    if (st::setting.use_filler) {
        at::Tensor filler_pos = torch::rand({__num_fillers__, 2}, torch::dtype(mov_node_size.dtype()));
        at::Tensor scale = die_ur - die_ll;
        at::Tensor shift = die_ll;
        filler_pos = filler_pos * scale + shift;

        mov_node_pos = torch::cat({mov_node_pos, filler_pos}, 0);
        mov_node_size = torch::cat({mov_node_size, filler_size}, 0);
    }
    if (st::setting.noise_ratio > 0) {
        at::Tensor noise = torch::rand_like(mov_node_pos, torch::dtype(mov_node_size.dtype()));
        noise = noise.sub(0.5).mul(mov_node_size).mul(st::setting.noise_ratio);
        mov_node_pos += noise;
    }

    // at::Tensor expand_ratio = mov_node_pos.new_ones((mov_node_pos.sizes()[0]));
    at::Tensor expand_ratio = torch::ones((mov_node_pos.size(0)), torch::dtype(mov_node_size.dtype()));
    if (st::setting.clamp_node) {
        auto [mov_lhs, mov_rhs] = movable_index;
        at::Tensor __mov_node_area__ = torch::prod(mov_node_size, 1);
        at::Tensor clamp_mov_node_size = mov_node_size.clamp(unit_len * sqrt(2));
        at::Tensor clamp_mov_node_area = torch::prod(clamp_mov_node_size, 1);
        // update
        expand_ratio = __mov_node_area__ / clamp_mov_node_area;
        mov_node_size = clamp_mov_node_size;
        // mov_node_size.index({Slice(cell_mov_rhs, mov_rhs)}) *= 1;
    }

    /* fillers */
    // mov_node_area = node_area.index({Slice(mov_lhs, mov_rhs)}).clone();
    // mov_node_to_num_pins = node_to_num_pins.index({Slice(mov_lhs, mov_rhs)}).clone();
    if (use_filler) {
        at::Tensor filler_area = torch::prod(filler_size, 1).unsqueeze(1);
        mov_node_area = torch::cat({mov_node_area, filler_area}, 0);
        at::Tensor filler_to_num_pins = mov_node_to_num_pins.new_zeros({__num_fillers__, 1});
        assert(filler_to_num_pins.sizes() == filler_area.sizes());
        mov_node_to_num_pins = torch::cat({mov_node_to_num_pins, filler_to_num_pins}, 0);
    }

    /* sortep maps */
    auto [tmp, mov_sorted_map] = torch::sort(mov_node_area.flatten(), 0, true);
    mov_sorted_map = mov_sorted_map.contiguous();
    at::Tensor mov_conn_sorted_map = mov_sorted_map;
    if (use_filler) {
        auto [mov_lhs, mov_rhs] = movable_index;
        auto [tmp1, mov_conn_sorted_map] =
            torch::sort(mov_node_area.index({Slice(mov_lhs, mov_rhs)}).flatten(), 0, true);
        auto [tmp2, filler_sorted_map] = torch::sort(mov_node_area.index({Slice(mov_rhs)}).flatten(), 0, true);
        mov_conn_sorted_map = mov_conn_sorted_map.contiguous();
        filler_sorted_map = filler_sorted_map.contiguous();
        sorted_maps =
            make_tuple(mov_sorted_map.to(device), mov_conn_sorted_map.to(device), filler_sorted_map.to(device));
    } else {
        auto [mov_lhs, mov_rhs] = movable_index;
        auto [tmp1, mov_conn_sorted_map] =
            torch::sort(mov_node_area.index({Slice(mov_lhs, mov_rhs)}).flatten(), 0, true);
        at::Tensor filler_sorted_map;
        mov_conn_sorted_map = mov_conn_sorted_map.contiguous();
        sorted_maps = make_tuple(mov_sorted_map.to(device), mov_conn_sorted_map.to(device), filler_sorted_map);
    }

    mov_node_weight = torch::ones({mov_node_pos.size(0)}, dtype(mov_node_pos.dtype()));
    mov_node_weight.index({Slice(cell_mov_rhs, mov_rhs)}) *= 0;

    node_type_indices.emplace_back(std::make_tuple(cell_mov_rhs, mov_rhs, "Via"));

    return make_tuple(mov_node_pos, mov_node_size, expand_ratio);
}  // END MODULE

void NodeData::init_shape_params(torch::Tensor move_node_size)
{
    // torch::Tensor node_size_flat;
    // torch::Tensor node_size_tall;
    // torch::Tensor node_orientation_flat;
    // torch::Tensor node_orientation_tall;
    // torch::Tensor pin_rel_cpos_flat;
    // torch::Tensor pin_rel_cpos_tall;
    node_size_tall = move_node_size.clone().cpu();
    node_size_flat = move_node_size.clone().cpu();
    pin_rel_cpos_tall = pin_rel_cpos.clone().cpu();
    pin_rel_cpos_flat = pin_rel_cpos.clone().cpu();
    node_orientation_tall = node_orient_top.clone().cpu();
    node_orientation_flat = node_orient_top.clone().cpu();

    auto device_node2pin_list = node2pin_list.device();
    auto device_node2pin_list_end = node2pin_list_end.device();

    node2pin_list=node2pin_list.cpu().contiguous();
    node2pin_list_end=node2pin_list_end.cpu().contiguous();

    auto node2pin_list_a = node2pin_list.accessor<int64_t, 1>();
    auto node2pin_list_end_a = node2pin_list_end.accessor<int64_t, 1>();

    for(int i = 0;i<macro_list.size();i++)
    {
        int macro_id = macro_list[i];
        float width = node_size[macro_id][0].item<float>();
        float height = node_size[macro_id][1].item<float>();
        if(height > width)// 本来是tall,只改flat
        {
            node_orientation_flat[macro_id] = (node_orient_top[macro_id] + 1)%4;
            // change shape
            node_size_flat[macro_id][0] = node_size_tall[macro_id][1];
            node_size_flat[macro_id][1] = node_size_tall[macro_id][0];
            // change pin_rel_pos
            int start_idx=0;
            if(macro_id>0)
            {
                start_idx = node2pin_list_end_a[macro_id-1];
            }
            int end_idx = node2pin_list_end_a[macro_id];
            for (int j=start_idx;j<end_idx;j++) {
                int pin_id = node2pin_list_a[j];
                pin_rel_cpos_flat[pin_id][0] = -pin_rel_cpos_tall[pin_id][1];
                pin_rel_cpos_flat[pin_id][1] = pin_rel_cpos_tall[pin_id][0];
            } 
        }else{// 本来是flat,只改tall
            node_orientation_tall[macro_id] = (node_orient_top[macro_id] + 1) % 4;
            // change shape
            node_size_tall[macro_id][0] = node_size_flat[macro_id][1];
            node_size_tall[macro_id][1] = node_size_flat[macro_id][0];
            // change pin_rel_pos
            int start_idx=0;
            if(macro_id>0)
            {
                start_idx = node2pin_list_end_a[macro_id-1];
            }
            int end_idx = node2pin_list_end_a[macro_id];
            for (int j=start_idx;j<end_idx;j++) {
                int pin_id = node2pin_list_a[j];
                pin_rel_cpos_tall[pin_id][0] = -pin_rel_cpos_flat[pin_id][1];
                pin_rel_cpos_tall[pin_id][1] = pin_rel_cpos_flat[pin_id][0];
            } 
        }
    }
    macro_shape_ratio.resize(macro_list.size());
    for(int i=0;i<macro_shape_ratio.size();i++)
    {
        macro_shape_ratio[i]=0.5;
    }
    
    node2pin_list=node2pin_list.to(device_node2pin_list);
    node2pin_list_end=node2pin_list_end.to(device_node2pin_list_end);
}

void NodeData::updata_shape_by_density_grad(torch::Tensor density_grad_4part, torch::Tensor& mov_node_size, int iteration)
{
    if(iteration<=500||iteration>1000)
    {
        return;
    }
    /*
    pin_rel_cpos=pin_rel_cpos.to(device_origin_pin_rel_cpos);
    pin_rel_cpos_top=pin_rel_cpos_top.to(device_origin_pin_rel_cpos_top);
    pin_rel_cpos_bot=pin_rel_cpos_bot.to(device_origin_pin_rel_cpos_bot);
    pin_id2node_id=pin_id2node_id.to(device_origin_pin_id2node_id);

    node2pin_list=node2pin_list.to(device_node2pin_list);
    node2pin_list_end=node2pin_list_end.to(device_node2pin_list_end);
    */
    auto device_origin_pin_rel_cpos = pin_rel_cpos.device();
    auto device_origin_pin_rel_cpos_top = pin_rel_cpos_top.device();
    auto device_origin_pin_rel_cpos_bot = pin_rel_cpos_bot.device();
    auto device_origin_pin_id2node_id = pin_id2node_id.device();
    auto device_node2pin_list = node2pin_list.device();
    auto device_node2pin_list_end = node2pin_list_end.device();
    auto device_node_size_bot = node_size_bot.device();
    auto device_node_size_top = node_size_top.device();

    
    pin_rel_cpos=pin_rel_cpos.cpu().contiguous();
    pin_rel_cpos_top=pin_rel_cpos_top.cpu().contiguous();
    pin_rel_cpos_bot=pin_rel_cpos_bot.cpu().contiguous();
    pin_id2node_id=pin_id2node_id.cpu().contiguous();
    node2pin_list=node2pin_list.cpu().contiguous();
    node2pin_list_end=node2pin_list_end.cpu().contiguous();
    node_size_top=node_size_top.cpu().contiguous();
    node_size_bot=node_size_bot.cpu().contiguous();

    auto pin_rel_cpos_a = pin_rel_cpos.accessor<float, 2>();
    auto pin_rel_cpos_top_a = pin_rel_cpos_top.accessor<float, 2>();
    auto pin_rel_cpos_bot_a = pin_rel_cpos_bot.accessor<float, 2>();
    auto pin_id2node_id_a = this->pin_id2node_id.accessor<int64_t, 1>();
    auto node2pin_list_a = node2pin_list.accessor<int64_t, 1>();
    auto node2pin_list_end_a = node2pin_list_end.accessor<int64_t, 1>();
    auto node_size_top_a = node_size_top.accessor<float, 2>();
    auto node_size_bot_a = node_size_bot.accessor<float, 2>();

    for(int i=0;i<macro_list.size();i++)
    {
        int macro_id = macro_list[i];
        float left_force = density_grad_4part[macro_id][0].item<float>();
        float right_force = density_grad_4part[macro_id][1].item<float>();
        float down_force = density_grad_4part[macro_id][2].item<float>();
        float top_force = density_grad_4part[macro_id][3].item<float>();
        float force_x_enlarge = (left_force + top_force - right_force - down_force);// 这个数越大越扁
        float ratio_change = force_x_enlarge/150;
        if(ratio_change>1)
        {
            ratio_change=1;
        }
        if(ratio_change<-1)
        {
            ratio_change=-1;
        }
        ratio_change*=0.02;
        macro_shape_ratio[i] += ratio_change;
        //1: 扁, 0: 高
        //可以用朝上和朝下的模板来做?然后再用pin位置调整?
        //大于0.5一个方向,小于0.5一个方向
        int start_legalize_index = 850;
        if(iteration>start_legalize_index)
        {
            float legalization_force = 0;
            if(macro_shape_ratio[i]>0.5)
            {
                legalization_force += (iteration-start_legalize_index)/100;
            }else{
                legalization_force -= (iteration-start_legalize_index)/100;
            }
            if(legalization_force < -1)
            {
                legalization_force = -1;
            }
            if(legalization_force > 1)
            {
                legalization_force = 1;
            }
            macro_shape_ratio[i] += legalization_force*0.02;
        }
        //@@ macro_shape_ratio[i] trunc to 0-1
        if(macro_shape_ratio[i]>1)
        {
            macro_shape_ratio[i]=1;
        }
        if(macro_shape_ratio[i]<0)
        {
            macro_shape_ratio[i]=0;
        }

        // 0.5: pin在center, 1:template_bian 0:template_tall;
        // shape = shape_bian * ratio + shape_tall * (1 - ratio);
        // pin_rel_pos =  pin_rel_pos_bian * ratio + pin_rel_pos_bian * (1 - ratio);
        // pin_rel_pos *= abs(ratio-0.5)*2;
        // orientation: 大于0.5: orientation_bian, 小于0.5: orientation_tall

        //sumup: shape_bian, shape_tall, pin_rel_pos_bian, pin_rel_pos_tall, orientation_bian, orientation_tall
        mov_node_size[macro_id] = node_size_flat[macro_id] * macro_shape_ratio[i] + node_size_tall[macro_id] * (1 - macro_shape_ratio[i]);
        // pin_rel_cpos[macro_id] =  pin_rel_cpos_flat[macro_id] * macro_shape_ratio[i] + pin_rel_cpos_tall[macro_id] * (1 - macro_shape_ratio[i]);
        // pin_rel_cpos[macro_id] *= abs(macro_shape_ratio[i]-0.5)*2;
        int start_idx=0;
        if(macro_id>0)
        {
            start_idx = node2pin_list_end_a[macro_id-1];
        }
        int end_idx = node2pin_list_end_a[macro_id];

        for (int j=start_idx;j<end_idx;j++) {
            int pin_id = node2pin_list_a[j];
            pin_rel_cpos[pin_id] =  pin_rel_cpos_flat[pin_id] * macro_shape_ratio[i] + pin_rel_cpos_tall[pin_id] * (1 - macro_shape_ratio[i]);
            pin_rel_cpos[pin_id] *= abs(macro_shape_ratio[i]-0.5)*2;
        }   

        int original_orient = node_orient_top[macro_id].item<int>();
        int orient = 0;
        
        if(macro_shape_ratio[i]>0.5)
        {
            orient = (node_orientation_flat[macro_id].item<int>()-original_orient+40)%4;
            node_orient_top[macro_id] = node_orientation_flat[macro_id]; // 此处没有修改真实shape
            node_orient_bot[macro_id] = node_orientation_flat[macro_id];
        }
        else{
            orient = (node_orientation_tall[macro_id].item<int>()-original_orient+40)%4;
            node_orient_top[macro_id] = node_orientation_tall[macro_id];
            node_orient_bot[macro_id] = node_orientation_tall[macro_id];
        }
        
        // update params of _bot and _top ///////////////////////////////////////////////////////////////////////////////////////////
        if(orient==1||orient==3)
        {
            float tmp_float = node_size_top_a[macro_id][1];
            node_size_top_a[macro_id][1] = node_size_top_a[macro_id][0];
            node_size_top_a[macro_id][0] = tmp_float;
    
            tmp_float = node_size_bot_a[macro_id][1];
            node_size_bot_a[macro_id][1] = node_size_bot_a[macro_id][0];
            node_size_bot_a[macro_id][0] = tmp_float;
        }
        //
        if(orient==1)
        {
            float tmp_float = pin_rel_pos_mean[i].first;
            pin_rel_pos_mean[i].first = -pin_rel_pos_mean[i].second;
            pin_rel_pos_mean[i].second = tmp_float;
        }
        if(orient==2)
        {
            pin_rel_pos_mean[i].first=-pin_rel_pos_mean[i].first;
            pin_rel_pos_mean[i].second=-pin_rel_pos_mean[i].second;
        }
        if(orient==3)
        {
            float tmp_float = pin_rel_pos_mean[i].first;
            pin_rel_pos_mean[i].first = pin_rel_pos_mean[i].second;
            pin_rel_pos_mean[i].second = -tmp_float;
        }
        start_idx=0;
        if(macro_id>0)
        {
            start_idx = node2pin_list_end_a[macro_id-1];
        }
        end_idx = node2pin_list_end_a[macro_id];
        for (int j=start_idx;j<end_idx;j++) {
            int pin_id = node2pin_list_a[j];
            //auto& node = nodes[pin.getParNodeId()];
            if(orient<=0) continue;
            //@FIX ME: here not confirm orient 1 and 3 who is lockwise and who iscounterclockwise
            if(orient==1)
            {
                // float tmp_float = pin_rel_cpos_a[pin_id][0];
                // pin_rel_cpos_a[pin_id][0] = -pin_rel_cpos_a[pin_id][1];
                // pin_rel_cpos_a[pin_id][1] = tmp_float;
                
                float tmp_float = pin_rel_cpos_top_a[pin_id][0];
                pin_rel_cpos_top_a[pin_id][0] = -pin_rel_cpos_top_a[pin_id][1];
                pin_rel_cpos_top_a[pin_id][1] = tmp_float;
                
                tmp_float = pin_rel_cpos_bot_a[pin_id][0];
                pin_rel_cpos_bot_a[pin_id][0] = -pin_rel_cpos_bot_a[pin_id][1];
                pin_rel_cpos_bot_a[pin_id][1] = tmp_float;
            }else if(orient==2)
            {
                // pin_rel_cpos_a[pin_id][0] = -pin_rel_cpos_a[pin_id][0];
                // pin_rel_cpos_a[pin_id][1] = -pin_rel_cpos_a[pin_id][1];
    
                pin_rel_cpos_top_a[pin_id][0] = -pin_rel_cpos_top_a[pin_id][0];
                pin_rel_cpos_top_a[pin_id][1] = -pin_rel_cpos_top_a[pin_id][1];
    
                pin_rel_cpos_bot_a[pin_id][0] = -pin_rel_cpos_bot_a[pin_id][0];
                pin_rel_cpos_bot_a[pin_id][1] = -pin_rel_cpos_bot_a[pin_id][1];
            }else if(orient==3){
                // float tmp_float = pin_rel_cpos_a[pin_id][0];
                // pin_rel_cpos_a[pin_id][0] = pin_rel_cpos_a[pin_id][1];
                // pin_rel_cpos_a[pin_id][1] = -tmp_float;
                
                float tmp_float = pin_rel_cpos_top_a[pin_id][0];
                pin_rel_cpos_top_a[pin_id][0] = pin_rel_cpos_top_a[pin_id][1];
                pin_rel_cpos_top_a[pin_id][1] = -tmp_float;
                
                tmp_float = pin_rel_cpos_bot_a[pin_id][0];
                pin_rel_cpos_bot_a[pin_id][0] = pin_rel_cpos_bot_a[pin_id][1];
                pin_rel_cpos_bot_a[pin_id][1] = -tmp_float;
            }
        }

    }

    pin_rel_cpos=pin_rel_cpos.to(device_origin_pin_rel_cpos);
    pin_rel_cpos_top=pin_rel_cpos_top.to(device_origin_pin_rel_cpos_top);
    pin_rel_cpos_bot=pin_rel_cpos_bot.to(device_origin_pin_rel_cpos_bot);
    pin_id2node_id=pin_id2node_id.to(device_origin_pin_id2node_id);
    node2pin_list=node2pin_list.to(device_node2pin_list);
    node2pin_list_end=node2pin_list_end.to(device_node2pin_list_end);
    node_size_top = node_size_top.to(device_node_size_top);
    node_size_bot = node_size_bot.to(device_node_size_bot);
}

void NodeData::update_macro_orientation(at::Tensor new_node_orient)
{
    auto devide_origin_node_orient_top = node_orient_top.device();
    auto devide_origin_node_orient_bot = node_orient_bot.device();
    auto devide_origin_node_size = node_size.device();
    auto devide_origin_node_size_top = node_size_top.device();
    auto devide_origin_node_size_bot = node_size_bot.device();

    node_orient_top = node_orient_top.cpu().contiguous();
    node_orient_bot = node_orient_bot.cpu().contiguous();
    node_size = node_size.cpu().contiguous();
    node_size_top = node_size_top.cpu().contiguous();
    node_size_bot = node_size_bot.cpu().contiguous();

    auto node_orient_top_a = node_orient_top.accessor<int64_t, 1>();
    auto node_orient_bot_a = node_orient_bot.accessor<int64_t, 1>();
    auto node_size_a = node_size.accessor<float, 2>();
    auto node_size_top_a = node_size_top.accessor<float, 2>();
    auto node_size_bot_a = node_size_bot.accessor<float, 2>();

    std::unordered_map<int, int> macros_rotated_backup;
    for(int i=0;i<macro_list.size();i++)
    {
        int macro_id = macro_list[i];
        macros_rotated_backup[macro_id] = node_orient_top_a[macro_id];
        if(new_node_orient[macro_id].item<int>()!=node_orient_top_a[macro_id])
        {
            int orient = (new_node_orient[macro_id].item<int>() - macros_rotated_backup[macro_id] + 40) % 4;
            if(orient==1||orient==3)
            {
                float tmp_float = node_size_top_a[macro_id][1];
                node_size_top_a[macro_id][1] = node_size_top_a[macro_id][0];
                node_size_top_a[macro_id][0] = tmp_float;
    
                tmp_float = node_size_bot_a[macro_id][1];
                node_size_bot_a[macro_id][1] = node_size_bot_a[macro_id][0];
                node_size_bot_a[macro_id][0] = tmp_float;
    
                
                tmp_float = node_size_a[macro_id][1];
                node_size_a[macro_id][1] = node_size_a[macro_id][0];
                node_size_a[macro_id][0] = tmp_float;
            }
        }
    }
    auto device_origin_pin_rel_cpos = pin_rel_cpos.device();
    auto device_origin_pin_rel_cpos_top = pin_rel_cpos_top.device();
    auto device_origin_pin_rel_cpos_bot = pin_rel_cpos_bot.device();
    auto device_origin_pin_id2node_id = pin_id2node_id.device();

    pin_rel_cpos=pin_rel_cpos.cpu().contiguous();
    pin_rel_cpos_top=pin_rel_cpos_top.cpu().contiguous();
    pin_rel_cpos_bot=pin_rel_cpos_bot.cpu().contiguous();
    pin_id2node_id=pin_id2node_id.cpu().contiguous();

    auto pin_rel_cpos_a = pin_rel_cpos.accessor<float, 2>();
    auto pin_rel_cpos_top_a = pin_rel_cpos_top.accessor<float, 2>();
    auto pin_rel_cpos_bot_a = pin_rel_cpos_bot.accessor<float, 2>();
    auto pin_id2node_id_a = this->pin_id2node_id.accessor<int64_t, 1>();


    auto device_node2pin_list = node2pin_list.device();
    auto device_node2pin_list_end = node2pin_list_end.device();
    node2pin_list=node2pin_list.cpu().contiguous();
    node2pin_list_end=node2pin_list_end.cpu().contiguous();
    auto node2pin_list_a = node2pin_list.accessor<int64_t, 1>();
    auto node2pin_list_end_a = node2pin_list_end.accessor<int64_t, 1>();

    for(int i=0;i<macro_list.size();i++)
    {
        int macro_id = macro_list[i];
        int orient = (new_node_orient[macro_id].item<int>() - macros_rotated_backup[macro_id] + 40) % 4;
        if(orient<=0) continue;
        if(orient==1)
        {
            float tmp_float = pin_rel_pos_mean[i].first;
            pin_rel_pos_mean[i].first = -pin_rel_pos_mean[i].second;
            pin_rel_pos_mean[i].second = tmp_float;
        }
        if(orient==2)
        {
            pin_rel_pos_mean[i].first=-pin_rel_pos_mean[i].first;
            pin_rel_pos_mean[i].second=-pin_rel_pos_mean[i].second;
        }
        if(orient==3)
        {
            float tmp_float = pin_rel_pos_mean[i].first;
            pin_rel_pos_mean[i].first = pin_rel_pos_mean[i].second;
            pin_rel_pos_mean[i].second = -tmp_float;
        }
        int start_idx=0;
        if(macro_id>0)
        {
            start_idx = node2pin_list_end_a[macro_id-1];
        }
        int end_idx = node2pin_list_end_a[macro_id];
        for (int j=start_idx;j<end_idx;j++) {
            int pin_id = node2pin_list_a[j];
            //auto& node = nodes[pin.getParNodeId()];
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
    }    
        
    for(int i=0;i<macro_list.size();i++)
    {
        int macro_id = macro_list[i];
        node_orient_top_a[macro_id] = new_node_orient[macro_id].item<int>();
        node_orient_bot_a[macro_id] = new_node_orient[macro_id].item<int>();
    }

    node_orient_top = node_orient_top.to(devide_origin_node_orient_top);
    node_orient_bot = node_orient_bot.to(devide_origin_node_orient_bot);
    node_size = node_size.to(devide_origin_node_size);
    node_size_top = node_size_top.to(devide_origin_node_size_top);
    node_size_bot = node_size_bot.to(devide_origin_node_size_bot);
    
    pin_rel_cpos=pin_rel_cpos.to(device_origin_pin_rel_cpos);
    pin_rel_cpos_top=pin_rel_cpos_top.to(device_origin_pin_rel_cpos_top);
    pin_rel_cpos_bot=pin_rel_cpos_bot.to(device_origin_pin_rel_cpos_bot);
    pin_id2node_id=pin_id2node_id.to(device_origin_pin_id2node_id);

    node2pin_list=node2pin_list.to(device_node2pin_list);
    node2pin_list_end=node2pin_list_end.to(device_node2pin_list_end);
}

void NodeData::update_macro_orientaion_by_pin_std(torch::Tensor current_node_pos)
{
    at::TensorAccessor pin_rel_cpos_a = pin_rel_cpos.accessor<float, 2>();
    auto tensorr = current_node_pos.clone().cpu().contiguous();
    at::TensorAccessor current_node_pos_a = tensorr.accessor<float, 2>();
    auto orientation_new = node_orient_top.clone().contiguous();
    at::TensorAccessor orientation_new_a = orientation_new.accessor<int64_t, 1>();
    
    // cout<<"??????"<<endl;
    for(int i=0;i<macro_list.size();i++)
    {
        int macro_id = macro_list[i];
        int sample_time = 30;
        if(macro_neighbors[i].size()<2*sample_time)
        {
            sample_time = macro_neighbors[i].size()/2;
        }
        float distance0=0;
        float distance1=0;
        for(int time=0;time<sample_time;time++)
        {
            int sample_sequence = rand()%macro_neighbors[i].size();
            int sample_node_id = macro_neighbors[i][sample_sequence];


            float pin_x_0 = current_node_pos_a[macro_id][0]+pin_rel_pos_mean[i].first;
            float pin_y_0 = current_node_pos_a[macro_id][1]+pin_rel_pos_mean[i].second;
            float pin_x_1 = current_node_pos_a[macro_id][0]-pin_rel_pos_mean[i].first;
            float pin_y_1 = current_node_pos_a[macro_id][1]-pin_rel_pos_mean[i].second;

            float std_pos_x = current_node_pos_a[sample_node_id][0];
            float std_pos_y = current_node_pos_a[sample_node_id][1];

            distance0 += abs(std_pos_x-pin_x_0) + abs(std_pos_y-pin_y_0);
            distance1 += abs(std_pos_x-pin_x_1) + abs(std_pos_y-pin_y_1);
        }
        if(distance1<distance0)
        {
            // cout<<"rotating macro "<<macro_id<<endl;
            orientation_new[macro_id]=(orientation_new[macro_id]+2+40)%4;
        }
        // if(distance1>distance0)//@@@@@@DEBUGGGGG
        // {
        //     // cout<<"rotating macro "<<macro_id<<endl;
        //     orientation_new[macro_id]=(orientation_new[macro_id]+2+40)%4;
        // }
    }
    update_macro_orientation(orientation_new);
}

void NodeData::update_macro_orientaion_by_pin_std_false(torch::Tensor current_node_pos)
{
    at::TensorAccessor pin_rel_cpos_a = pin_rel_cpos.accessor<float, 2>();
    auto tensorr = current_node_pos.clone().cpu().contiguous();
    at::TensorAccessor current_node_pos_a = tensorr.accessor<float, 2>();
    auto orientation_new = node_orient_top.clone().contiguous();
    at::TensorAccessor orientation_new_a = orientation_new.accessor<int64_t, 1>();
    // cout<<"??????"<<endl;
    for(int i=0;i<macro_list.size();i++)
    {
        int macro_id = macro_list[i];
        cout<<"checking macro "<<macro_id<<endl;
        int sample_time = 30;
        if(macro_neighbors[i].size()<2*sample_time)
        {
            sample_time = macro_neighbors[i].size()/2;
        }
        float distance0=0;
        float distance1=0;
        cout<<"rel_pos: "<<pin_rel_pos_mean[i].first<<" "<<pin_rel_pos_mean[i].second<<endl;
        float std_pos_x_sum = 0;
        float std_pos_y_sum = 0;
        for(int time=0;time<sample_time;time++)
        {
            int sample_sequence = rand()%macro_neighbors[i].size();
            int sample_node_id = macro_neighbors[i][sample_sequence];


            float pin_x_0 = current_node_pos_a[macro_id][0]+pin_rel_pos_mean[i].first;
            float pin_y_0 = current_node_pos_a[macro_id][1]+pin_rel_pos_mean[i].second;
            float pin_x_1 = current_node_pos_a[macro_id][0]-pin_rel_pos_mean[i].first;
            float pin_y_1 = current_node_pos_a[macro_id][1]-pin_rel_pos_mean[i].second;

            float std_pos_x = current_node_pos_a[sample_node_id][0];
            float std_pos_y = current_node_pos_a[sample_node_id][1];

            distance0 += abs(std_pos_x-pin_x_0) + abs(std_pos_y-pin_y_0);
            distance1 += abs(std_pos_x-pin_x_1) + abs(std_pos_y-pin_y_1);
            std_pos_x_sum+=std_pos_x;
            std_pos_y_sum+=std_pos_y;
        }
        cout<<"std_pos_sum: "<<std_pos_x_sum/30<<" "<<std_pos_y_sum/30<<endl;
        cout<<"std_pos_mean: "<<std_pos_x_sum/30<<" "<<std_pos_y_sum/30<<endl;
        // if(distance1<distance0)
        // {
        //     // cout<<"rotating macro "<<macro_id<<endl;
        //     orientation_new[macro_id]=(orientation_new[macro_id]+2+40)%4;
        // }
        if(distance1>distance0)
        {
            cout<<"rotating macro "<<macro_id<<endl;
            orientation_new[macro_id]=(orientation_new[macro_id]+2+40)%4;
        }
    }
    update_macro_orientation(orientation_new);
}


void NodeData::update_macro_orientaion_by_pin_std_right(torch::Tensor current_node_pos)
{
    at::TensorAccessor pin_rel_cpos_a = pin_rel_cpos.accessor<float, 2>();
    auto tensorr = current_node_pos.clone().cpu().contiguous();
    at::TensorAccessor current_node_pos_a = tensorr.accessor<float, 2>();
    auto orientation_new = node_orient_top.clone().contiguous();
    at::TensorAccessor orientation_new_a = orientation_new.accessor<int64_t, 1>();
    // cout<<"??????"<<endl;
    for(int i=0;i<macro_list.size();i++)
    {
        int macro_id = macro_list[i];
        cout<<"checking: "<<macro_id<<endl;
        int sample_time = 30;
        if(macro_neighbors[i].size()<2*sample_time)
        {
            sample_time = macro_neighbors[i].size()/2;
        }
        float distance0=0;
        float distance1=0;
        cout<<"rel_pos: "<<pin_rel_pos_mean[i].first<<" "<<pin_rel_pos_mean[i].second<<endl;
        float std_pos_x_sum = 0;
        float std_pos_y_sum = 0;
        for(int time=0;time<sample_time;time++)
        {
            int sample_sequence = rand()%macro_neighbors[i].size();
            int sample_node_id = macro_neighbors[i][sample_sequence];


            float pin_x_0 = current_node_pos_a[macro_id][0]+pin_rel_pos_mean[i].first;
            float pin_y_0 = current_node_pos_a[macro_id][1]+pin_rel_pos_mean[i].second;
            float pin_x_1 = current_node_pos_a[macro_id][0]-pin_rel_pos_mean[i].first;
            float pin_y_1 = current_node_pos_a[macro_id][1]-pin_rel_pos_mean[i].second;

            float std_pos_x = current_node_pos_a[sample_node_id][0];
            float std_pos_y = current_node_pos_a[sample_node_id][1];

            distance0 += abs(std_pos_x-pin_x_0) + abs(std_pos_y-pin_y_0);
            distance1 += abs(std_pos_x-pin_x_1) + abs(std_pos_y-pin_y_1);
            std_pos_x_sum+=std_pos_x;
            std_pos_y_sum+=std_pos_y;
        }
        cout<<"std_pos_sum: "<<std_pos_x_sum/30<<" "<<std_pos_y_sum/30<<endl;
        cout<<"std_pos_mean: "<<std_pos_x_sum/30<<" "<<std_pos_y_sum/30<<endl;
        if(distance1<distance0)
        {
            cout<<"rotating macro "<<macro_id<<endl;
            orientation_new[macro_id]=(orientation_new[macro_id]+2+40)%4;
        }
        // if(distance1>distance0)
        // {
        //     // cout<<"rotating macro "<<macro_id<<endl;
        //     orientation_new[macro_id]=(orientation_new[macro_id]+2+40)%4;
        // }
    }
    update_macro_orientation(orientation_new);
}