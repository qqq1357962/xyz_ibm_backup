#pragma once

#include "utils/argparse.hpp"
#include "utils/reflect.h"
#include <torch/torch.h>

namespace st {

class Setting {
public:
    void parse(argparse::ArgumentParser args);

public:
    int mode;
    int numPart;
    int minGPStep;
    int min_gp3d_record_step;
    bool withMacro;
    bool use_floorplan;
    bool use_greedy_place_in_fp;
    bool half_filler_height = false;
    bool is_fp_permit_change_cross_chip;
    bool adjust_macro;
    bool move_macro_3d;
    double gp_padding;
    double macro_hpwl_scale;
    std::string dataset;
    std::string design_name;
    std::string input_path;
    std::string output_path;
    bool load_from_raw = false;
    int gpu = -1;
    int num_threads;

    bool verbose = false;
    bool draw_placement = false;
    bool enable_rotate_in_gp = false;
    std::string result_dir;
    std::string log_dir;
    std::string log_name;
    std::string exp_id;
    std::string load_stage;
    std::string load_file;
    std::string load_json;
    std::string load_def_template;
    std::string output_def;
    int log_freq;

    /* model params */
    double lr;
    int inner_iter;
    int inner_iter_gp3d;
    torch::Tensor cache_density_grad_4part;
    torch::Tensor cache_macro_mask;
    double cache_density_weight = 1;
    double cache_density_weight_xy = 1;
    int num_bin_x;
    int num_bin_y;
    int num_bin_z;
    double target_density;
    bool use_filler = true;
    double noise_ratio;
    int ignore_net_degree;
    bool scale_design = true;
    bool clamp_node = false;
    std::string loss_type;
    double via_target_density;
    std::string net_type;
    std::string den_type;
    bool block_row;
    int site_width;
    double sideline;

public:
    /* params scheduler */
    bool step_precond_coef;
    int magic_hpwl;
    double density_weight;
    double density_weight_coef;
    double wa_coeff;
    double stop_overflow;
    double stop_overflow_via;
    bool use_precond;
    bool enable_skip_update;
    bool quad_penalty;
    double quad_coeff;
    bool early_stop_check_plateau;

public:
    // flow
    bool gp;

    // pt
    std::string partitioner;
    int num_cuts;
    double num_folds;
    double num_grids;
    int slice_direction;
    double soft_margin;
    bool pin_propagation;
    std::string pt_model;
    double pt_imbl;
    bool dynamic_ratio;
    bool mononlithic;
    int global_const;
    double die_diff;
    double patoh_guide_ratio;
    bool skip_2d;

    bool rotate_180 = false;
    bool rotate_90 = false;
    std::string rotate_type;
    double rotate_thre;
    int rotate_coef;

    // dp
    bool save_model;
    std::string gp_model;
    std::string lg_model;
    bool eval_params;
    bool lg;
    bool dp;
    bool pp;
    bool rf;
    bool sw;
    bool via_dp;
    double disp_coef;
    int lg_ver;
    bool rf_flag = false;
    bool worse_legalize;

    // 3D GP params
    bool pt;
    bool use_filler_3d = true;
    int cut_net_thres;
    double net_weight_coef;
    double net_weight_offset;
    int num_den_layer;
    bool use_pre_gp;
    bool use_pre_pt;
    double shrink_size;
    bool match_2pin_nets;
    double force_coeff_2d;
    double wa_coeff_wa_z;
    double wa_coeff_wa_xy;
    bool correlate_bbox;
    double bbox_correlation;
    std::string wa_z_model;
    double top_util_filler;
    bool clamp_util;
    double stop_overflow_3d;
    int round_recursion;
    int net_cut_str_thrs;
    double strengthen_via_density;
    double strengthen_net_wa_coef;
    bool select_nets;
    double weaken_net_size;
    double weaken_net_wa_coef;
    std::string net_weight_type;
    std::string filler_type;
    int kernel_size;
    int num_bin_3d;
    bool visualize_curve;
    int fmwl_iter;
    double fmwl_area_coef;
    bool local_density_weight;
    int stack_cells;
    bool skip_gp3d;
    bool skip_hpwl_fm; 
    bool skip_draw;
    int first_magic_hpwl; 
    int second_num_bin_x;
    int second_num_bin_y;

    // other
    double draw_mat_size;
    int omni_int = 0;
    double omni_float = 0;

    bool log_verbose;

    int min_stop_iter = 100;

public:
    // history
    int iteration = 0;
    int __ori_magic_hpwl__;
    double __ori_density_weight__;
    double __ori_density_weight_coef__;
    double __ori_wa_coeff__;
    bool __ori_quad_penalty__;
};

extern Setting setting;

}  // namespace st


BOOST_FUSION_ADAPT_STRUCT(st::Setting,
                          lr,
                          num_bin_x,
                          num_bin_y,
                          num_bin_z,
                          target_density,
                          use_filler,
                          noise_ratio,
                          ignore_net_degree,
                          scale_design,
                          loss_type,
                          net_type,
                          step_precond_coef,
                          magic_hpwl,
                          density_weight,
                          density_weight_coef,
                          wa_coeff,
                          stop_overflow,
                          stop_overflow_via,
                          use_precond,
                          enable_skip_update,
                          quad_penalty,
                          quad_coeff,
                          early_stop_check_plateau,
                          partitioner,
                          num_cuts,
                          num_grids,
                          slice_direction,
                          soft_margin,
                          pin_propagation,
                          dynamic_ratio,
                          mononlithic,
                          global_const,
                          use_filler_3d,
                          cut_net_thres,
                          net_weight_coef,
                          net_weight_offset,
                          num_den_layer,
                          use_pre_gp,
                          shrink_size,
                          force_coeff_2d,
                          wa_coeff_wa_z,
                          wa_coeff_wa_xy,
                          correlate_bbox,
                          bbox_correlation,
                          wa_z_model,
                          top_util_filler,
                          clamp_util,
                          stop_overflow_3d,
                          round_recursion,
                          net_cut_str_thrs,
                          strengthen_via_density,
                          strengthen_net_wa_coef,
                          select_nets,
                          weaken_net_size,
                          weaken_net_wa_coef,
                          net_weight_type,
                          filler_type,
                          num_bin_3d,
                          omni_int,
                          omni_float)