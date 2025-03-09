#pragma once

#include "global.h"
#include "parser/db/Database.h"
#include "placer/database.h"

class NodeData3D : public PlaceData {
public:
    NodeData3D(NodeData& data_);
    void preprocess();
    void backup_ori_var();
    void preshift();
    void prescale_by_site_width();
    void prescale();
    bool check_design();
    void pre_compute_var();
    bool init_fence_region();
    void logging_statistics();

    using PlaceData::get_mov_node_info;
    using PlaceData::init_filler;
    /* initializer */
    void init_filler();
    void compute_filler();
    void compute_precond_var();
    void compute_sorted_node_map();
    /* get mov_node info */
    tuple<at::Tensor, at::Tensor, at::Tensor> get_mov_node_info();
    tuple<at::Tensor, at::Tensor, at::Tensor> get_mov_node_info_with_via();
    void update_net_weight(torch::Tensor box_len);

public:
    torch::Device device = torch::kCPU;
    int num_bin_z;

    /* site info */
    int __ori_die_lx__;
    int __ori_die_hx__;
    int __ori_die_ly__;
    int __ori_die_hy__;
    int __ori_die_lz__;
    int __ori_die_hz__;

    int site_depth;
    /* area */
    torch::Tensor node_area;
    torch::Tensor mov_cell_area;
    float __total_mov_area_without_filler__;
    float bin_area;

    torch::Tensor bondingInfo;
    torch::Tensor node_util_weight;
    torch::Tensor node_util_weight_x;
    torch::Tensor node_util_weight_y;
    torch::Tensor ratio_difference;

    torch::Tensor rowHeights;
    torch::Tensor maxUtilM;

    torch::Tensor sidelines_ll;
    torch::Tensor sidelines_ur;

    torch::Tensor total_mov_cell_areas;
    vector<torch::Tensor> mov_node_weights;

    torch::Tensor via_node_size;

    float shrink_size = 1;
    torch::Tensor actualUtilM;
    torch::Tensor node_wgt_grad;
    int num_fillers_single_chip = 0;
    int num_fillers_cross_chip = 0;

    torch::Tensor macro_mask;

public:
    void to(torch::Device device_) {
        macro_mask = macro_mask.to(device_);
        node_size = node_size.to(device_);
        pin_rel_cpos = pin_rel_cpos.to(device_);
        pin_rel_cpos_top = pin_rel_cpos_top.to(device_);
        pin_rel_cpos_bot = pin_rel_cpos_bot.to(device_);
        pin_id2node_id = pin_id2node_id.to(device_);
        node2pin_list = node2pin_list.to(device_);
        node2pin_list_end = node2pin_list_end.to(device_);
        hyperedge_list = hyperedge_list.to(device_);
        hyperedge_list_end = hyperedge_list_end.to(device_);
        die_ll = die_ll.to(device_);
        die_ur = die_ur.to(device_);

        die_info = die_info.to(device_);
        core_info = core_info.to(device_);
        node_pos = node_pos.to(device_);
        pin_id2net_id = pin_id2net_id.to(device_);
        __die_shift__ = __die_shift__.to(device_);
        __die_scale__ = __die_scale__.to(device_);
        node_to_num_pins = node_to_num_pins.to(device_);
        unit_len = unit_len.to(device_);
        net_mask = net_mask.to(device_);
        hpwl_scale = hpwl_scale.to(device_);

        node_util_weight = node_util_weight.to(device_);
        node_util_weight_x = node_util_weight_x.to(device_);
        ratio_difference = ratio_difference.to(device_);
        node_util_weight_y = node_util_weight_y.to(device_);

        rowHeights = rowHeights.to(device_);
        maxUtilM = maxUtilM.to(device_);
        sidelines_ll = sidelines_ll.to(device_);
        sidelines_ur = sidelines_ur.to(device_);
        net_weight = net_weight.to(device_);
        mov_node_weight = mov_node_weight.numel() ? mov_node_weight.to(device_) : mov_node_weight;
        mov_node_weights[0] = mov_node_weights[0].numel() ? mov_node_weights[0].to(device_) : mov_node_weights[0];
        mov_node_weights[1] = mov_node_weights[1].numel() ? mov_node_weights[1].to(device_) : mov_node_weights[1];

        init_density_map = init_density_map.numel() ? init_density_map.to(device_) : init_density_map;
        node_die = node_die.numel() ? node_die.to(device_) : node_die;
    }
};