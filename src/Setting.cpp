#include "Setting.h"

namespace st {

std::string str2lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

bool str2bool(std::string str) {
    std::string lower_str = str2lower(str);
    if (lower_str == "true" || lower_str == "t" || lower_str == "yes" || lower_str == "y" || lower_str == "1") {
        return true;
    }
    return false;
}

void Setting::parse(argparse::ArgumentParser args) {
    if(args.get<std::string>("--version")=="23")
    {
         withMacro=1;
    }
    load_stage = args.get<std::string>("--load_stage");
    load_file = args.get<std::string>("--load_file");
    load_json = args.get<std::string>("--load_json");
    load_def_template = args.get<std::string>("--load_def_template");
    output_def = args.get<std::string>("--output_def");
    if(args.get<std::string>("--fp")=="true")
    {
         use_floorplan=1;
    }
    if(args.get<std::string>("--use_greedy_place_in_fp")=="true")
    {
         use_greedy_place_in_fp=1;
    }
    if(args.get<std::string>("--half_filler_height")=="true")
    {
         half_filler_height=1;
    }
    if(args.get<std::string>("--is_fp_permit_change_cross_chip")=="true")
    {
         is_fp_permit_change_cross_chip=1;
    }
    if(args.get<std::string>("--adjust_macro")=="true")
    {
        adjust_macro=1;
    }
    if(args.get<std::string>("--is_move_macro_3d")=="true")
    {
         move_macro_3d=1;
    }
    /* general setting */
    dataset = args.get<std::string>("--dataset");
    design_name = args.get<std::string>("--design_name");
    num_threads = args.get<int>("--num_threads");
    numPart = args.get<int>("--num_part");
    minGPStep = args.get<int>("--min_gp_step");
    min_gp3d_record_step = args.get<int>("--min_gp3d_record_step");
    gpu = args.get<int>("--gpu");

    result_dir = args.get<std::string>("--result_dir");
    result_dir = "results/" + result_dir;  // TODO: merge logs

    log_dir = args.get<std::string>("--log_dir");
    log_name = args.get<std::string>("--log_name");
    exp_id = args.get<std::string>("--exp_id");
    log_freq = args.get<int>("--log_freq");

    load_from_raw = str2bool(args.get<std::string>("--load_from_raw"));
    verbose = str2bool(args.get<std::string>("--verbose"));
    draw_placement = str2bool(args.get<std::string>("--draw_placement"));
    enable_rotate_in_gp = str2bool(args.get<std::string>("--enable_rotate_in_gp"));

    /* model params */
    lr = args.get<double>("--lr");
    inner_iter = args.get<int>("--inner_iter");
    inner_iter_gp3d = args.get<int>("--inner_iter_gp3d");
    num_bin_x = args.get<int>("--num_bin_x");
    num_bin_y = args.get<int>("--num_bin_y");
    num_bin_z = args.get<int>("--num_bin_z");

    target_density = args.get<double>("--target_density");
    via_target_density = args.get<double>("--via_target_density");
    use_filler = str2bool(args.get<std::string>("--use_filler"));
    noise_ratio = args.get<double>("--noise_ratio");
    gp_padding = args.get<double>("--gp_padding");
    macro_hpwl_scale = args.get<double>("--macro_hpwl_scale");
    ignore_net_degree = args.get<int>("--ignore_net_degree");
    scale_design = str2bool(args.get<std::string>("--scale_design"));
    clamp_node = str2bool(args.get<std::string>("--clamp_node"));
    loss_type = args.get<std::string>("--loss_type");
    net_type = args.get<std::string>("--net_type");
    den_type = args.get<std::string>("--den_type");
    block_row = str2bool(args.get<std::string>("--block_row"));
    site_width = args.get<int>("--site_width");
    sideline = args.get<double>("--sideline");

    /* params scheduler */
    magic_hpwl = args.get<int>("--magic_hpwl");
    step_precond_coef = str2bool(args.get<std::string>("--step_precond_coef"));
    density_weight = args.get<double>("--density_weight");
    density_weight_coef = args.get<double>("--density_weight_coef");
    wa_coeff = args.get<double>("--wa_coeff");
    stop_overflow = args.get<double>("--stop_overflow");
    stop_overflow_via = args.get<double>("--stop_overflow_via");
    use_precond = str2bool(args.get<std::string>("--use_precond"));
    enable_skip_update = str2bool(args.get<std::string>("--enable_skip_update"));
    quad_penalty = str2bool(args.get<std::string>("--quad_penalty"));
    quad_coeff = args.get<double>("--quad_coeff");
    early_stop_check_plateau = str2bool(args.get<std::string>("--early_stop_check_plateau"));

    // TODO: ICCAD 2022 only, remove me after the contest
    input_path = args.get<std::string>("--input_path");
    output_path = args.get<std::string>("--output_path");
    if (input_path != "null") {
        dataset = "iccad2022";
        design_name = input_path;  // TODO:will init it later
        std::size_t pos = design_name.find("B_");
        std::string tmp = design_name.substr(pos + 2);
        std::size_t pos1 = tmp.find(".");
        design_name = tmp.substr(0, pos1);
    }

    /* main flow */
    gp = str2bool(args.get<std::string>("--gp"));

    /* partitioner arguments */
    partitioner = args.get<std::string>("partitioner");
    num_cuts = args.get<int>("--num_cuts");
    num_folds = args.get<double>("--num_folds");
    num_grids = args.get<double>("--num_grids");
    slice_direction = args.get<int>("--slice_direction");
    pt_imbl = args.get<double>("--pt_imbl");
    pin_propagation = str2bool(args.get<std::string>("--pin_propagation"));
    dynamic_ratio = str2bool(args.get<std::string>("--dynamic_ratio"));
    pt_model = args.get<std::string>("--pt_model");
    mononlithic = str2bool(args.get<std::string>("--mononlithic"));
    global_const = args.get<int>("--global_const");

    /* detailed placer */
    save_model = str2bool(args.get<std::string>("--save_model"));
    gp_model = args.get<std::string>("--gp_model");
    lg_model = args.get<std::string>("--lg_model");

    eval_params = str2bool(args.get<std::string>("--eval_params"));
    lg = str2bool(args.get<std::string>("--lg"));
    dp = str2bool(args.get<std::string>("--dp"));
    pp = str2bool(args.get<std::string>("--pp"));
    rf = str2bool(args.get<std::string>("--rf"));
    sw = str2bool(args.get<std::string>("--sw"));
    via_dp = str2bool(args.get<std::string>("--via_dp"));
    disp_coef = args.get<double>("--disp_coef");
    soft_margin = args.get<double>("--soft_margin");
    lg_ver = args.get<int>("--lg_ver");
    worse_legalize = str2bool(args.get<std::string>("--worse_legalize"));

    /* 3D global placement */
    pt = str2bool(args.get<std::string>("--pt"));
    use_filler_3d = str2bool(args.get<std::string>("--use_filler_3d"));
    cut_net_thres = args.get<int>("--cut_net_thres");
    net_weight_coef = args.get<double>("--net_weight_coef");
    net_weight_offset = args.get<double>("--net_weight_offset");
    num_den_layer = args.get<int>("--num_den_layer");
    use_pre_gp = str2bool(args.get<std::string>("--use_pre_gp"));
    use_pre_pt = str2bool(args.get<std::string>("--use_pre_pt"));
    shrink_size = args.get<double>("--shrink_size");
    match_2pin_nets = str2bool(args.get<std::string>("--match_2pin_nets"));
    force_coeff_2d = args.get<double>("--force_coeff_2d");
    wa_coeff_wa_z = args.get<double>("--wa_coeff_wa_z");
    wa_coeff_wa_xy = args.get<double>("--wa_coeff_wa_xy");
    correlate_bbox = str2bool(args.get<std::string>("--correlate_bbox"));
    bbox_correlation = args.get<double>("--bbox_correlation");
    wa_z_model = args.get<std::string>("wa_z_model");
    top_util_filler = args.get<double>("--top_util_filler");
    clamp_util = str2bool(args.get<std::string>("--clamp_util"));
    stop_overflow_3d = args.get<double>("--stop_overflow_3d");
    round_recursion = args.get<int>("--round_recursion");
    net_cut_str_thrs = args.get<int>("--net_cut_str_thrs");
    strengthen_via_density = args.get<double>("--strengthen_via_density");
    strengthen_net_wa_coef = args.get<double>("--strengthen_net_wa_coef");
    select_nets = str2bool(args.get<std::string>("--select_nets"));
    weaken_net_size = args.get<double>("--weaken_net_size");
    weaken_net_wa_coef = args.get<double>("--weaken_net_wa_coef");
    net_weight_type = args.get<std::string>("net_weight_type");
    filler_type = args.get<std::string>("filler_type");
    kernel_size = args.get<int>("--kernel_size");
    num_bin_3d = args.get<int>("--num_bin_3d");
    visualize_curve = str2bool(args.get<std::string>("--visualize_curve"));
    fmwl_iter = args.get<int>("--fmwl_iter");
    fmwl_area_coef = args.get<double>("--fmwl_area_coef");
    local_density_weight = str2bool(args.get<std::string>("--local_density_weight"));
    stack_cells = args.get<int>("--stack_cells");
    skip_gp3d = str2bool(args.get<std::string>("--skip_gp3d"));
    skip_hpwl_fm = str2bool(args.get<std::string>("--skip_hpwl_fm"));
    skip_draw = str2bool(args.get<std::string>("--skip_draw"));
    first_magic_hpwl = args.get<int>("--first_magic_hpwl");
    second_num_bin_x = args.get<int>("--second_num_bin_x");
    second_num_bin_y = args.get<int>("--second_num_bin_y");
    die_diff = args.get<double>("--die_diff");
    patoh_guide_ratio = args.get<double>("--patoh_guide_ratio");

    
    rotate_180 = str2bool(args.get<std::string>("--rotate_180"));
    rotate_90 = str2bool(args.get<std::string>("--rotate_90"));
    rotate_type = args.get<std::string>("--rotate_type");
    rotate_thre = args.get<double>("--rotate_thre");
    rotate_coef = args.get<int>("--rotate_coef");

    /* others */
    draw_mat_size = args.get<double>("--draw_mat_size");
    omni_int = args.get<int>("--omni_int");
    omni_float = args.get<double>("--omni_float");

    log_verbose = str2bool(args.get<std::string>("--log_verbose"));
    if (!log_verbose) {
        draw_placement = false;
    }

    __ori_magic_hpwl__ = magic_hpwl;
    __ori_density_weight__ = density_weight;
    __ori_density_weight_coef__ = density_weight_coef;
    __ori_wa_coeff__ = wa_coeff;
    __ori_quad_penalty__ = quad_penalty;
}

Setting setting;

}  // namespace st