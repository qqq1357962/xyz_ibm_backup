#include "partitioning_db.h"

namespace pt {

int floorDiv(float a, float b) { return std::floor(a / b); }

int ceilDiv(float a, float b) { return std::ceil(a / b); }

int floorDivRound(float a, float b, int prec) { return std::floor(std::round(a * prec) / std::round(b * prec)); }

int ceilDivRound(float a, float b, int prec) { return std::ceil(std::round(a * prec) / std::round(b * prec)); }

int roundDiv(float a, float b) { return std::round(a / b); }

PartitionDataTensor::PartitionDataTensor(NodeData& data, torch::Tensor node_pos_, torch::Tensor exact_node_size_) {
    /* process pos c->l */
    node_pos_init = node_pos_.clone();
    exact_node_size = exact_node_size_;
    node_die = data.node_die.clone();
    node_weight = data.node_die.clone();

    auto exact_node_size_x = exact_node_size.index({"...", 0});
    auto exact_node_size_y = exact_node_size.index({"...", 1});
    node_size_x = exact_node_size_x;
    node_size_y = exact_node_size_y;

    /* node pos */
    init_x = node_pos_init.index({"...", 0}).clone();
    init_y = node_pos_init.index({"...", 1}).clone();
    // x = node_pos_init.index({"...", 0}) - exact_node_size_x / 2;
    // y = node_pos_init.index({"...", 1}) - exact_node_size_y / 2;
    x = init_x.clone();   // FIXME: init -> center
    y = init_y.clone();

    /* process node sizes */
    auto via_node_size = exact_node_size.index({Slice(data.cell_mov_rhs, None)});
    auto node_size_bot = torch::cat({data.node_size_bot, via_node_size}, 0);
    auto node_size_top = torch::cat({data.node_size_top, via_node_size}, 0);
    node_sizes.push_back(node_size_bot);
    node_sizes.push_back(node_size_top);
    node_size_xs.push_back(node_size_bot.index({"...", 0}));
    node_size_xs.push_back(node_size_top.index({"...", 0}));
    node_size_ys.push_back(node_size_bot.index({"...", 1}));
    node_size_ys.push_back(node_size_top.index({"...", 1}));

    /* preprocess pin_pos */
    // pin_rel_cpos = data.pin_rel_cpos.clone();
    // // torch::Tensor pin_rel_lpos_bot = data.pin_rel_cpos_bot.clone();
    // // torch::Tensor pin_rel_lpos_top = data.pin_rel_cpos_top.clone(); // FIXME: fixed pin_rel_pos
    // torch::Tensor pin_rel_lpos_bot = data.pin_rel_cpos.clone();
    // torch::Tensor pin_rel_lpos_top = data.pin_rel_cpos.clone(); // FIXME: fixed pin_rel_pos

    // torch::Tensor pin_rel_lpos = data.pin_rel_cpos.clone();
    // for (int64_t i = 0; i < data.num_pins; i++) {
    //     int64_t node_id = data.pin_id2node_id[i].item<int64_t>();
    //     pin_rel_lpos_bot[i] += node_size_bot[node_id] / 2;
    //     pin_rel_lpos_top[i] += node_size_top[node_id] / 2;
    //     pin_rel_lpos[i] += exact_node_size[node_id] / 2;
    // }
    // pin_rel_lposes.push_back(pin_rel_lpos_bot);
    // pin_rel_lposes.push_back(pin_rel_lpos_top);

    // pin_offset_x = pin_rel_lpos.index({"...", 0});
    // pin_offset_y = pin_rel_lpos.index({"...", 1});

    pin_rel_cpos = data.pin_rel_cpos;
    pin_offset_x = pin_rel_cpos.index({"...", 0});
    pin_offset_y = pin_rel_cpos.index({"...", 1});
    pin_offset_xs.push_back(data.pin_rel_cpos_bot.index({"...", 0}));
    pin_offset_xs.push_back(data.pin_rel_cpos_top.index({"...", 0}));
    pin_offset_ys.push_back(data.pin_rel_cpos_bot.index({"...", 1}));
    pin_offset_ys.push_back(data.pin_rel_cpos_top.index({"...", 1}));

    /* cell number info */
    num_pins = data.num_pins;
    int num_nets = data.num_nets;
    int num_nodes = data.num_nodes;

    /* update node -> pin map */
    torch::Tensor via_node2pin_list_end =
        torch::arange(num_pins + 1, num_pins + num_nets + 1, torch::dtype(torch::kInt64));
    torch::Tensor node2pin_list_end = torch::cat({data.node2pin_list_end, via_node2pin_list_end}, 0);
    flat_node2pin_start_map = torch::cat({torch::zeros({1}, torch::dtype(torch::kInt64)), node2pin_list_end}, 0);
    torch::Tensor via_node2pin_list = torch::arange(num_pins, num_pins + num_nets, torch::dtype(torch::kInt64));
    flat_node2pin_map = torch::cat({data.node2pin_list, via_node2pin_list}, 0);
    pin2node_map = data.pin_id2node_id.clone();

    /* update net -> pin map / hyperedge_list*/
    flat_net2pin_start_map = torch::cat({torch::zeros({1}, torch::dtype(torch::kInt64)), data.hyperedge_list_end}, 0);
    flat_net2pin_map = data.hyperedge_list;
    pin2net_map = data.pin_id2net_id.clone();

    // net_mask = torch::ones(num_nets, torch::dtype(torch::kInt)); // TODO:
    net_mask = torch::_cast_Int(data.net_to_num_pins > 2);
}


PartitionData::PartitionData(NodeData& data, PartitionDataTensor& at_db)
    : x(at_db.x.accessor<float, 1>()),
      y(at_db.y.accessor<float, 1>()),
      init_x(at_db.init_x.accessor<float, 1>()),
      init_y(at_db.init_y.accessor<float, 1>()),
      node_size_x(at_db.node_size_x.accessor<float, 1>()),
      node_size_y(at_db.node_size_y.accessor<float, 1>()),
      pin_offset_x(at_db.pin_offset_x.accessor<float, 1>()),
      pin_offset_y(at_db.pin_offset_y.accessor<float, 1>()),
      flat_node2pin_start_map(at_db.flat_node2pin_start_map.accessor<int64_t, 1>()),
      flat_node2pin_map(at_db.flat_node2pin_map.accessor<int64_t, 1>()),
      pin2node_map(at_db.pin2node_map.accessor<int64_t, 1>()),
      flat_net2pin_start_map(at_db.flat_net2pin_start_map.accessor<int64_t, 1>()),
      flat_net2pin_map(at_db.flat_net2pin_map.accessor<int64_t, 1>()),
      pin2net_map(at_db.pin2net_map.accessor<int64_t, 1>()),
      net_mask(at_db.net_mask.accessor<int, 1>()),
      node_die(at_db.node_die.accessor<int, 1>()),
      node_weight(at_db.node_weight.accessor<int, 1>()) {
    /* construct dp info form placedb */
    num_threads = std::max(st::setting.num_threads, 1);
    num_nodes = at_db.node_pos_init.size(0);  // FIXME: #nodes = cell+via #node
    num_movable_nodes = data.num_nodes;       // FIXME: #mov_nodes = cell #node
    num_nets = data.num_nets;
    num_pins = data.num_pins;

    xl = data.die_info[0].item<float>();
    xh = data.die_info[1].item<float>();
    yl = data.die_info[2].item<float>();
    yh = data.die_info[3].item<float>();

    node_size_xs.push_back(at_db.node_size_xs[0].accessor<float, 1>());
    node_size_xs.push_back(at_db.node_size_xs[1].accessor<float, 1>());
    node_size_ys.push_back(at_db.node_size_ys[0].accessor<float, 1>());
    node_size_ys.push_back(at_db.node_size_ys[1].accessor<float, 1>());

    pin_offset_xs.push_back(at_db.pin_offset_xs[0].accessor<float, 1>());
    pin_offset_xs.push_back(at_db.pin_offset_xs[1].accessor<float, 1>());
    pin_offset_ys.push_back(at_db.pin_offset_ys[0].accessor<float, 1>());
    pin_offset_ys.push_back(at_db.pin_offset_ys[1].accessor<float, 1>());

    // debug --> hpwl_kernel info
    pin_id2node_id = data.pin_id2node_id;
    pin_rel_cpos = at_db.pin_rel_cpos;
    pin_rel_cpos_bot = data.pin_rel_cpos_bot.clone();
    pin_rel_cpos_top = data.pin_rel_cpos_top.clone();

    logger.info("#node: %d, #mov_node: %d", num_nodes, num_movable_nodes);
}  // END MODULE

}  // namespace pt