#include "detailed_place_db.h"

namespace dp {

int floorDiv(float a, float b) { return std::floor(a / b); }

int ceilDiv(float a, float b) { return std::ceil(a / b); }

int floorDivRound(float a, float b, int prec) { return std::floor(std::round(a * prec) / std::round(b * prec)); }

int ceilDivRound(float a, float b, int prec) { return std::ceil(std::round(a * prec) / std::round(b * prec)); }

int roundDiv(float a, float b) { return std::round(a / b); }

DetailedPlaceDataTensor::DetailedPlaceDataTensor(NodeData& data, torch::Tensor node_pos_, torch::Tensor node_size_) {
    /* process pos c->l */
    node_pos_init = node_pos_.clone();
    node_size = node_size_;
    node_size_x = node_size.index({"...", 0});
    node_size_y = node_size.index({"...", 1});
    init_x = node_pos_init.index({"...", 0}) - node_size_x / 2;
    init_y = node_pos_init.index({"...", 1}) - node_size_y / 2;
    x = init_x.clone();  // FIXME: center pos -> lower-left
    y = init_y.clone();

    /* preprocess pin_pos */
    torch::Tensor pin_rel_lpos = data.pin_rel_cpos.clone();
    for (int64_t i = 0; i < pin_rel_lpos.size(0); i++) {
        int64_t node_id = data.pin_id2node_id[i].item<int64_t>();
        pin_rel_lpos[i] += node_size[node_id] / 2;
    }

    pin_offset_x = pin_rel_lpos.index({"...", 0});
    pin_offset_y = pin_rel_lpos.index({"...", 1});

    /* cell number info */
    int num_pins = data.num_pins;
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

    net_mask = torch::ones(num_nets, torch::dtype(torch::kInt));
}

DetailedPlaceData::DetailedPlaceData(NodeData& data,
                                     DetailedPlaceDataTensor& at_db,
                                     int num_sites_y_,
                                     float row_height_)
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
      node_weight(at_db.node_weight.accessor<int, 1>()),
      node_die(at_db.node_die.accessor<int, 1>())
    {
    /* construct dp info form placedb */
    row_height = row_height_;
    num_sites_y = num_sites_y_;

    row_width = (data.die_info[1] - data.die_info[0]).item<float>();
    row_size_x = (data.die_info[1] - data.die_info[0]).item<float>();
    site_width = data.site_width;
    // site_width = 1;                                 // FIXME: pre_scale by site_width
    site_width_safe_divide = int(1);  // TODO:
    row_shift = data.die_info[2].item<float>();

    num_sites_x = int(row_size_x / site_width);
    num_threads = std::max(st::setting.num_threads, 1);
    num_nodes = at_db.node_pos_init.size(0);  // FIXME: #nodes = cell+via #node
    num_movable_nodes = data.num_nodes;       // FIXME: #mov_nodes = cell #node
    num_nets = data.num_nets;

    xl = data.die_info[0].item<float>();
    xh = data.die_info[1].item<float>();
    yl = data.die_info[2].item<float>();
    yh = data.die_info[3].item<float>();

    this->node_id2node_name = data.node_id2node_name;

    logger.info(
        "#siteX: %d, #siteY: %d, #node: %d,#mov_node: %d", num_sites_x, num_sites_y, num_nodes, num_movable_nodes);

    /* preprocess nets */

}  // END MODULE

//---------------------------------------------------------------------

void DetailedPlaceData::make_row2node_map(const torch::TensorAccessor<float, 1> vx,
                                          const torch::TensorAccessor<float, 1> vy,
                                          std::vector<std::vector<int> >& row2node_map) {
    // distribute cells to rows
    for (int i = i_bgn; i < i_end; ++i) {  // FIXME: exclude vias
        if (node_weight[i] == 0) continue;
        float node_yl = vy[i];
        float node_yh = node_yl + node_size_y[i];

        // FIXME: float divide
        // int row_idxl = floorDiv(node_yl - yl, row_height);
        int row_idxl = floorDivRound(node_yl - yl, row_height, site_width_safe_divide);
        // int row_idxh = ceilDiv(node_yh - yl, row_height);
        int row_idxh = ceilDivRound(node_yh - yl, row_height, site_width_safe_divide);
        row_idxl = std::max(row_idxl, 0);
        row_idxh = std::min(row_idxh, num_sites_y);

        for (int row_id = row_idxl; row_id < row_idxh; ++row_id) {
            float row_yl = yl + row_id * row_height;
            float row_yh = row_yl + row_height;

            if (node_yl < row_yh && node_yh > row_yl)  // overlap with row
            {
                row2node_map[row_id].push_back(i);
            }
        }
    }

#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (int i = 0; i < num_sites_y; ++i) {
        auto& row2nodes = row2node_map[i];
        // sort cells within rows according to left edges
        std::sort(row2nodes.begin(), row2nodes.end(), [&](int node_id1, int node_id2) {
            float x1 = vx[node_id1];
            float x2 = vx[node_id2];
            return x1 < x2 || (x1 == x2 && node_id1 < node_id2);
        });
        // After sorting by left edge,
        // there is a special case for fixed cells where
        // one fixed cell is completely within another in a row.
        // This will cause failure to detect some overlaps.
        // We need to remove the "small" fixed cell that is inside another.
        if (!row2nodes.empty()) {
            std::vector<int> tmp_nodes;
            tmp_nodes.reserve(row2nodes.size());
            tmp_nodes.push_back(row2nodes.front());
            for (int j = 1, je = row2nodes.size(); j < je; ++j) {
                int node_id1 = row2nodes.at(j - 1);
                int node_id2 = row2nodes.at(j);
                // two fixed cells
                if (node_id1 >= i_end && node_id2 >= i_end) {
                    float xl1 = vx[node_id1];
                    float xl2 = vx[node_id2];
                    float width1 = node_size_x[node_id1];
                    float width2 = node_size_x[node_id2];
                    float xh1 = xl1 + width1;
                    float xh2 = xl2 + width2;
                    // only collect node_id2 if its right edge is righter than node_id1
                    if (xh1 < xh2) {
                        tmp_nodes.push_back(node_id2);
                    }
                } else {
                    tmp_nodes.push_back(node_id2);
                }
            }
            row2nodes.swap(tmp_nodes);

            // sort according to center
            std::sort(row2nodes.begin(), row2nodes.end(), [&](int node_id1, int node_id2) {
                float x1 = vx[node_id1] + node_size_x[node_id1] / 2;
                float x2 = vx[node_id2] + node_size_x[node_id2] / 2;
                return x1 < x2 || (x1 == x2 && node_id1 < node_id2);
            });
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

void DetailedPlaceData::make_bin2node_map(const torch::TensorAccessor<float, 1> host_x,
                                          const torch::TensorAccessor<float, 1> host_y,
                                          const torch::TensorAccessor<float, 1> host_node_size_x,
                                          const torch::TensorAccessor<float, 1> host_node_size_y,
                                          std::vector<std::vector<int> >& bin2node_map,
                                          std::vector<BinMapIndex>& node2bin_map) {
    // construct bin2node_map
    for (int i = i_bgn; i < i_end; ++i) {
        if (node_weight[i] == 0) continue;
        int node_id = i;
        float node_x = host_x[node_id] + host_node_size_x[node_id] / 2;
        float node_y = host_y[node_id] + host_node_size_y[node_id] / 2;

        int bx = std::min(std::max((int)floorDiv(node_x - xl, bin_size_x), 0), num_bins_x - 1);
        int by = std::min(std::max((int)floorDiv(node_y - yl, bin_size_y), 0), num_bins_y - 1);
        int bin_id = bx * num_bins_y + by;
        // int sub_id = bin2node_map.at(bin_id).size();
        bin2node_map.at(bin_id).push_back(node_id);
    }
    // construct node2bin_map
    for (unsigned int bin_id = 0; bin_id < bin2node_map.size(); ++bin_id) {
        for (unsigned int sub_id = 0; sub_id < bin2node_map[bin_id].size(); ++sub_id) {
            int node_id = bin2node_map[bin_id][sub_id];
            BinMapIndex& bm_idx = node2bin_map.at(node_id);
            bm_idx.bin_id = bin_id;
            bm_idx.sub_id = sub_id;
        }
    }
}  // END MODULE

//--------------------------------------------------------------------

float DetailedPlaceData::compute_total_hpwl() {
    partial_hpwl = torch::zeros({num_nets, 2}, torch::dtype(torch::kFloat));
    float total_hpwl = 0;
    for (int net_id = 0; net_id < num_nets; ++net_id) {
        total_hpwl += compute_net_hpwl(net_id);
    }
    return total_hpwl;
}

//--------------------------------------------------------------------

float DetailedPlaceData::compute_total_hpwl_concurrent(const torch::TensorAccessor<float, 1> x,
                                                       const torch::TensorAccessor<float, 1> y,
                                                       float* net_hpwls) {
    partial_hpwl = torch::zeros({num_nets, 2}, torch::dtype(torch::kFloat));
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_nets; ++i) {
        net_hpwls[i] = compute_net_hpwl(i);
    }
    float hpwl = 0;
    // I found OpenMP reduction cannot guarantee run-to-run determinism
    //#pragma omp parallel for num_threads(state.num_threads) default(shared)
    // reduction(+:hpwl)
    for (int i = 0; i < num_nets; ++i) {
        hpwl += net_hpwls[i];
    }

    // printf("Torch calculated HPWL: %.2f\n", torch::sum(partial_hpwl.sum(1)).item<float>());
    return hpwl;
}  // END MODULE

//---------------------------------------------------------------------

float DetailedPlaceData::compute_net_hpwl(int net_id) const {
    float hpwl;
    if (!via_dp) {
        Box box(std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max());
        for (int net2pin_id = flat_net2pin_start_map[net_id]; net2pin_id < flat_net2pin_start_map[net_id + 1];
            ++net2pin_id) {
            int net_pin_id = flat_net2pin_map[net2pin_id];
            int other_node_id = pin2node_map[net_pin_id];
            if (node_die[other_node_id] == 0) continue;
            box.xl = std::min(box.xl, x[other_node_id] + pin_offset_x[net_pin_id]);
            box.xh = std::max(box.xh, x[other_node_id] + pin_offset_x[net_pin_id]);
            box.yl = std::min(box.yl, y[other_node_id] + pin_offset_y[net_pin_id]);
            box.yh = std::max(box.yh, y[other_node_id] + pin_offset_y[net_pin_id]);
        }
        if (box.xl == std::numeric_limits<float>::max() || box.yl == std::numeric_limits<float>::max()) {
            return (float)0;
        }
        partial_hpwl[net_id][0] = box.xh - box.xl;
        partial_hpwl[net_id][1] = box.yh - box.yl;
        hpwl = (box.xh - box.xl) + (box.yh - box.yl);
    } else{
        vector<Box> boxs;
        boxs.emplace_back(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
        boxs.emplace_back(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
        bool node_exist_flag[2] = {false, false};
        for (int net2pin_id = flat_net2pin_start_map[net_id]; net2pin_id < flat_net2pin_start_map[net_id + 1];
            ++net2pin_id) {
            int net_pin_id = flat_net2pin_map[net2pin_id];
            int other_node_id = pin2node_map[net_pin_id];
            if ((other_node_id < num_movable_nodes) ||
                        ((other_node_id >= num_movable_nodes) && (node_weight[other_node_id] == 1))) {
                int c_id = node_die[other_node_id];
                float xxl = x[other_node_id] + pin_offset_x[net_pin_id];
                float yyl = y[other_node_id] + pin_offset_y[net_pin_id];
                if (c_id != 2) {
                    node_exist_flag[c_id] = true;
                    boxs[c_id].xl = std::min(boxs[c_id].xl, xxl);
                    boxs[c_id].xh = std::max(boxs[c_id].xh, xxl);
                    boxs[c_id].yl = std::min(boxs[c_id].yl, yyl);
                    boxs[c_id].yh = std::max(boxs[c_id].yh, yyl);
                } else {
                    for (c_id = 0; c_id < 2; c_id++) {
                        node_exist_flag[c_id] = true;
                        boxs[c_id].xl = std::min(boxs[c_id].xl, xxl);
                        boxs[c_id].xh = std::max(boxs[c_id].xh, xxl);
                        boxs[c_id].yl = std::min(boxs[c_id].yl, yyl);
                        boxs[c_id].yh = std::max(boxs[c_id].yh, yyl);
                    }
                }
            }
        }
        for (int c_id = 0; c_id < 2; c_id++) {
            if (!node_exist_flag[c_id]) {
                boxs[c_id].xl = 0;
                boxs[c_id].xh = 0;
                boxs[c_id].yl = 0;
                boxs[c_id].yh = 0;
            }
        }
        hpwl = (boxs[0].xh - boxs[0].xl + boxs[0].yh - boxs[0].yl) +
                (boxs[1].xh - boxs[1].xl + boxs[1].yh - boxs[1].yl);
    }

    return hpwl;
}  // END MODULE

//---------------------------------------------------------------------

Box DetailedPlaceData::compute_optimal_region(int node_id) const {
    Box box(std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max());
    for (int node2pin_id = flat_node2pin_start_map[node_id]; node2pin_id < flat_node2pin_start_map[node_id + 1];
         ++node2pin_id) {
        int node_pin_id = flat_node2pin_map[node2pin_id];
        int net_id = pin2net_map[node_pin_id];
        if (net_mask[net_id]) {
            for (int net2pin_id = flat_net2pin_start_map[net_id]; net2pin_id < flat_net2pin_start_map[net_id + 1];
                 ++net2pin_id) {
                int net_pin_id = flat_net2pin_map[net2pin_id];
                int other_node_id = pin2node_map[net_pin_id];
                if ((node_id != other_node_id) && (node_die[other_node_id] == 1)) {
                    box.xl = std::min(box.xl, x[other_node_id] + pin_offset_x[net_pin_id]);
                    box.xh = std::max(box.xh, x[other_node_id] + pin_offset_x[net_pin_id]);
                    box.yl = std::min(box.yl, y[other_node_id] + pin_offset_y[net_pin_id]);
                    box.yh = std::max(box.yh, y[other_node_id] + pin_offset_y[net_pin_id]);
                }
            }
        }
    }
    shift_box_to_layout(box);

    return box;
}  // END MODULE

//---------------------------------------------------------------------

}  // namespace dp