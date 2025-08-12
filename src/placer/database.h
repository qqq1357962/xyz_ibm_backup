#pragma once

#include "global.h"
#include "torch/script.h"
#include "utils/GlobalParser.h"

class PlaceData {
public:
    PlaceData() {}

public:
    /* dataset format */
    string dataset_format;
    string dataset;
    string design_name;

    /* for GPU.ver */
    torch::Device device = torch::kCPU;

    /* General info */
    int num_nodes;
    int num_nets;
    int num_pins;

    /* ID & Node type */
    vector<string> node_id2node_name;

    /* Die/Core/Site info */
    torch::Tensor die_info;
    torch::Tensor die_info_back_up;
    torch::Tensor core_info;
    torch::Tensor die_ur;
    torch::Tensor die_ll;
    torch::Tensor core_ur;
    torch::Tensor core_ll;
    torch::Tensor __die_scale__;
    torch::Tensor __die_shift__;
    tuple<int, int> site_info;  // TODO:
    int site_width;
    int site_height;
    int site_width_current;

    /* Node & Pin */
    torch::Tensor node_pos;
    //torch::Tensor node_rotate;
    torch::Tensor node_size;
    torch::Tensor node_orient_bot;
    torch::Tensor node_orient_top;
    torch::Tensor node_type;
    torch::Tensor pin_rel_cpos;
    torch::Tensor pin_rel_cpos_top;
    torch::Tensor pin_rel_cpos_bot;
    torch::Tensor pin_size;

    /* Hyperedge */
    torch::Tensor pin_id2node_id;
    torch::Tensor hyperedge_index;
    torch::Tensor hyperedge_list;
    torch::Tensor hyperedge_list_end;
    torch::Tensor pin_id2net_id;

    /* Node To Pin */
    torch::Tensor node2pin_index;
    torch::Tensor node2pin_list;
    torch::Tensor node2pin_list_end;

    /* Net info */
    torch::Tensor net_mask;
    torch::Tensor net_weight;

    torch::Tensor hpwl_scale;
    torch::Tensor net_to_num_pins;

    /* Node info */
    tuple<torch::Tensor, torch::Tensor, torch::Tensor> sorted_maps;
    torch::Tensor node_to_num_pins;
    torch::Tensor node_area;
    /* MovNode info */
    torch::Tensor mov_node_to_num_pins;
    torch::Tensor mov_node_area;

    /* Bin info */
    int num_bin_x;
    int num_bin_y;
    double __total_mov_area_without_filler__;
    double bin_area;
    double target_density;
    torch::Tensor unit_len;

    /* Row info */
    torch::Tensor row_height;
    torch::Tensor numRows;

    /* Node index */
    vector<string> all_node_types;
    vector<tuple<gp::index_type, gp::index_type, string>> node_type_indices;
    tuple<int, int> movable_index;
    tuple<int, int> cell_movable_index;
    // Mov + FloatMov + Fix + IOPin
    tuple<int, int> connected_index;
    // Fix + IOPin + Blkg + FloatIOPin + FloatFix
    tuple<int, int> fixed_index;
    tuple<int, int> movable_connected_index;
    tuple<int, int> fixed_connected_index;
    tuple<int, int> fixed_unconnected_index;

    /* Fence */
    bool enable_fence;
    int num_regions;

    /* backup info */
    torch::Tensor __ori_die_info__;
    torch::Tensor __ori_core_info__;
    torch::Tensor __ori_node_pos__;
    //torch::Tensor __ori_node_rotate__;
    torch::Tensor __ori_node_size__;
    torch::Tensor __ori_pin_rel_cpos__;
    torch::Tensor __ori_pin_size__;
    torch::Tensor __ori_region_boxes__;

    torch::Tensor __ori_pin_id2node_id__;
    torch::Tensor __ori_pin_id2net_id__;
    torch::Tensor __ori_hyperedge_list__;
    torch::Tensor __ori_hyperedge_list_end__;

    torch::Tensor __ori_node_size_bot__;
    torch::Tensor __ori_node_size_top__;
    torch::Tensor __ori_node_orient_bot__;
    torch::Tensor __ori_node_orient_top__;

    torch::Tensor __ori_node_size_norm_bot__;
    torch::Tensor __ori_node_size_norm_top__;
    torch::Tensor __ori_node_size_norm__;

    /* site info */
    int __ori_die_lx__;
    int __ori_die_hx__;
    int __ori_die_ly__;
    int __ori_die_hy__;

    /* bin info */
    bool clamp_node;

public:
    /* Filler & Initalizer */
    bool use_filler = true;
    int __num_fillers__ = 0;
    torch::Tensor init_density_map;
    torch::Tensor single_filler_size;
    torch::Tensor filler_size;

    /* partition info */
    torch::Tensor node_die;
    torch::Tensor mov_cell_area;
    torch::Tensor mov_cell_util;

    vector<vector<int>> pin_pairs;
    vector<int> bond_nets;

    int num_bonds;
    torch::Tensor bonding_map;
    torch::Tensor pin_num_count;
    torch::Tensor pin_num_count_cut;
    torch::Tensor node_naive_flag;

    int cell_mov_lhs;
    int cell_mov_rhs;
    int iopin_mov_lhs;
    int iopin_mov_rhs;
    vector<int> macro_list;
    vector<float> macro_shape_ratio;
    vector<vector<int> > macro_neighbors;
    vector<int> pin_direction;
    vector<pair<float,float> > pin_rel_pos_mean;
    torch::Tensor mov_node_weight;

    torch::Tensor mov_node_sideline_ll;
    torch::Tensor mov_node_sideline_ur;

public:
    /* initializer */
    virtual void init_filler() { printlog(LOG_INFO, "Defined elsewhere in a specified placedata"); }
    void compute_filler();
    void compute_precond_var();
    void compute_sorted_node_map();
    /* get mov_node info */
    virtual tuple<at::Tensor, at::Tensor, at::Tensor> get_mov_node_info() {
        printlog(LOG_INFO, "Defined elsewhere in a specified placedata");
        torch::Tensor tmp;
        return {tmp, tmp, tmp};
    }

};  // END MODULE FIXME: const vars

//---------------------------------------------------------------------

class NodeData : public PlaceData {
public:
    NodeData() {}
    NodeData(Dict& design_info);
    NodeData(Dict& design_info, torch::Device device_);

    torch::Tensor to(torch::Tensor at_T, torch::Device device_) {
        return at_T.numel() ? at_T.to(device_) : at_T;
    }

    void to(torch::Device device_) {
        node_size = node_size.to(device_);
        // node_orient = node_orient.to(device_);
        // bondingInfo = bondingInfo.to(device_);
        // node_size_bot = node_size_bot.to(device_);
        // node_size_top = node_size_top.to(device_);
        // pin_rel_cpos_bot = pin_rel_cpos_bot.to(device_);
        // pin_rel_cpos_top = pin_rel_cpos_top.to(device_);
        
        to(bondingInfo, device_);
        to(node_size_bot, device_);
        to(node_size_top, device_);
        to(node_orient_bot, device_);
        to(node_orient_top, device_);
        to(pin_rel_cpos_top, device_);
        to(pin_rel_cpos_bot, device_);
        to(bondingInfo, device_);
        to(bondingInfo, device_);

        macro_mask = macro_mask.to(device_);
        pin_rel_cpos = pin_rel_cpos.to(device_);
        pin_id2node_id = pin_id2node_id.to(device_);
        node2pin_list = node2pin_list.to(device_);
        node2pin_list_end = node2pin_list_end.to(device_);
        hyperedge_list = hyperedge_list.to(device_);
        hyperedge_list_end = hyperedge_list_end.to(device_);
        die_ll = die_ll.to(device_);
        die_ur = die_ur.to(device_);

        die_info = die_info.to(device_);
        core_info = core_info.to(device_);
        rowHeights = rowHeights.to(device_);
        numRows = numRows.to(device_);
        node_pos = node_pos.to(device_);
        //node_rotate = node_rotate.to(device_);
        hyperedge_index = hyperedge_index.to(device_);
        pin_id2net_id = pin_id2net_id.to(device_);
        __die_shift__ = __die_shift__.to(device_);
        __die_scale__ = __die_scale__.to(device_);
        node_to_num_pins = node_to_num_pins.to(device_);
        unit_len = unit_len.to(device_);
        net_mask = net_mask.to(device_);
        hpwl_scale = hpwl_scale.to(device_);

        mov_node_weight = mov_node_weight.numel() ? mov_node_weight.to(device_) : mov_node_weight;
        init_density_map = init_density_map.numel() ? init_density_map.to(device_) : init_density_map;
        init_density_maps = init_density_maps.numel() ? init_density_maps.to(device_) : init_density_maps;

        hyperedge_list_cc[0] = hyperedge_list_cc[0].numel() ? hyperedge_list_cc[0].to(device_) : hyperedge_list_cc[0];
        hyperedge_list_cc[1] = hyperedge_list_cc[1].numel() ? hyperedge_list_cc[1].to(device_) : hyperedge_list_cc[1];
        hyperedge_list_end_cc[0] =
            hyperedge_list_end_cc[0].numel() ? hyperedge_list_end_cc[0].to(device_) : hyperedge_list_end_cc[0];
        hyperedge_list_end_cc[1] =
            hyperedge_list_end_cc[1].numel() ? hyperedge_list_end_cc[1].to(device_) : hyperedge_list_end_cc[1];
        // mov_node_weights[0] = mov_node_weights[0].numel() ? mov_node_weights[0].to(device_) : mov_node_weights[0];
        // mov_node_weights[1] = mov_node_weights[1].numel() ? mov_node_weights[1].to(device_) : mov_node_weights[1];
        // mov_node_weights[2] = mov_node_weights[2].numel() ? mov_node_weights[2].to(device_) : mov_node_weights[2];
        mov_node_weights = mov_node_weights.numel() ? mov_node_weights.to(device_) : mov_node_weights;
        mov_node_sideline = mov_node_sideline.numel() ? mov_node_sideline.to(device_) : mov_node_sideline;
        mov_node_sideline_ll = mov_node_sideline_ll.numel() ? mov_node_sideline_ll.to(device_) : mov_node_sideline_ll;
        mov_node_sideline_ur = mov_node_sideline_ur.numel() ? mov_node_sideline_ur.to(device_) : mov_node_sideline_ur;


        node_die = node_die.numel() ? node_die.to(device_) : node_die;
        mov_cell_areas = mov_cell_areas.numel() ? mov_cell_areas.to(device_) : mov_cell_areas;
        max_mov_cell_areas = max_mov_cell_areas.numel() ? max_mov_cell_areas.to(device_) : max_mov_cell_areas;
    }

    void reset_net_node() {
        pin_id2node_id = __ori_pin_id2node_id__.clone();
        pin_id2net_id = __ori_pin_id2net_id__.clone();
        hyperedge_list = __ori_hyperedge_list__.clone();
        hyperedge_list_end = __ori_hyperedge_list_end__.clone();

        movable_index = make_tuple(cell_mov_lhs, cell_mov_rhs);
        node_size = node_size.index({Slice(cell_mov_lhs, cell_mov_rhs)});
    }

    void reset() {
        pin_rel_cpos = __ori_pin_rel_cpos__.clone();
        node_size_bot = __ori_node_size_bot__.clone();
        node_size_top = __ori_node_size_top__.clone();
        node_orient_bot = __ori_node_orient_bot__.clone();
        node_orient_top = __ori_node_orient_top__.clone();

        reset_net_node();
    }

public:
    /* Region */
    torch::Tensor node_id2region_id;
    torch::Tensor region_boxes;
    torch::Tensor region_boxes_end;

public:
    /* preprocess */
    // torch::Tensor die_shift;
    // torch::Tensor die_scale;

public:
    // TODO:
    // void to(torch::Device device);
    /* preprocess */
    void preprocess();
    void backup_ori_var();
    void preshift();
    void prescale_by_site_width();
    void postscale_by_site_width();
    void prescale();
    bool check_design();
    void pre_compute_var();
    bool init_fence_region();
    void logging_statistics();
    void setMacroOrient();
    void setMacroOrient_vertical();
    void setMacroOrient(torch::Tensor node_rotate);
    void setMacroOrient_default();
    void setMacroOrient_ilp();
    void update_macro_orientation(at::Tensor new_node_orient);
    void updata_shape_by_density_grad(torch::Tensor density_grad_4part, torch::Tensor& mov_node_size, int iteration);
    void update_macro_orientaion_by_pin_std(torch::Tensor current_node_pos);
    void update_macro_orientaion_by_pin_std_false(torch::Tensor current_node_pos);
    void update_macro_orientaion_by_pin_std_right(torch::Tensor current_node_pos);
    void init_shape_params(torch::Tensor move_node_size);

    using PlaceData::get_mov_node_info;
    using PlaceData::init_filler;
    /* initializer */
    void init_filler();
    void compute_filler();
    void compute_precond_var();
    void compute_sorted_node_map();
    /* get mov_node info */
    tuple<at::Tensor, at::Tensor, at::Tensor> get_mov_node_info();
    tuple<at::Tensor, at::Tensor, at::Tensor> get_mov_node_info_dummy_via();
    tuple<at::Tensor, at::Tensor, at::Tensor> get_mov_node_info_cross_chip(bool init_macro=true, bool init_stdcell=true, bool move_macro=true);
    tuple<at::Tensor, at::Tensor, at::Tensor> get_mov_node_info_cross_chip_with_via();

public:
    /* iccad2022 */
    torch::Tensor macro_mask;
    int numTechlibs;
    torch::Tensor maxUtilM;
    torch::Tensor rowHeights;
    torch::Tensor bondingInfo;
    int bondingCost;
    torch::Tensor numRows;

    /* node info */
    torch::Tensor node_size_bot;
    torch::Tensor pin_size_bot;

    torch::Tensor node_size_top;
    torch::Tensor pin_rel_cpos_top;
    torch::Tensor pin_rel_cpos_bot;

    torch::Tensor node_size_flat;
    torch::Tensor node_size_tall;
    torch::Tensor node_orientation_flat;
    torch::Tensor node_orientation_tall;
    torch::Tensor pin_rel_cpos_flat;
    torch::Tensor pin_rel_cpos_tall;

    torch::Tensor pin_size_top;

    torch::Tensor aspect_ratio;
    torch::Tensor stack_cells;
    torch::Tensor node_size_selector;

    /* bondings areas */
    int num_bondings;
    torch::Tensor bonding_pos;
    torch::Tensor bonding_size;

    /* cell areas */
    long mov_cell_area_bot_bound;
    long mov_cell_area_top_bound;
    torch::Tensor mov_cell_areas;
    torch::Tensor max_mov_cell_areas;

    /* partitioner info */
    at::Tensor tech_ratio;
    at::Tensor node_ratio;
    float area_ratio;
    float area_ratio_bot;
    float area_ratio_top;
    int site_width_keep = 0;
    int site_height_keep = 0;

    at::Tensor node_area_bot;
    at::Tensor node_area_top;

    torch::Tensor net_wgt_grad;
    torch::Tensor node_wgt_grad;
    torch::Tensor node_size_exact;
    torch::Tensor mov_node_die;
    // vector<torch::Tensor> mov_node_weights;
    torch::Tensor mov_node_weights;
    vector<torch::Tensor> net_masks;

    torch::Tensor init_density_maps;

    /* multi circuit node info */
    torch::Tensor mov_node_sideline;
    // torch::Tensor mov_node_sideline_ll;
    // torch::Tensor mov_node_sideline_ur;

    vector<torch::Tensor> mov_node_sizes;
    vector<torch::Tensor> idx_cc;
    vector<torch::Tensor> hyperedge_list_cc;
    vector<torch::Tensor> hyperedge_list_end_cc;
    vector<vector<index_type>> hyperedge_list_raw;
    vector<vector<index_type>> hyperedge_list_end_raw;  // FIXME: vector data blob_from will release

    /* precond calc */
    vector<torch::Tensor> mov_node_sizes_array;
    vector<torch::Tensor> mov_node_areas_array;
    vector<torch::Tensor> mov_node_to_num_pins_array;

    torch::Tensor actualUtilM;
    torch::Tensor upper_lower_bound_ratio;

    tuple<at::Tensor, at::Tensor, at::Tensor> get_mov_node_info_cross_chip_with_via(
        PlaceData& via_data,
        tuple<torch::Tensor, torch::Tensor, torch::Tensor> mov_node_info,
        tuple<torch::Tensor, torch::Tensor, torch::Tensor> via_mov_node_info);
};

tuple<Dict, shared_ptr<db::Database>, shared_ptr<gp::GPDatabase>> load_dataset();
