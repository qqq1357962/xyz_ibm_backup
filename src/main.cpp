#include "global.h"
#include "placer/placer.h"

void signalHandler(int signum) {
    std::cout << "Signal (" << signum << ") received. Exiting...\n";
    std::exit(signum);
}

inline const char* const BoolToString(bool b) { return b ? "true" : "false"; }

void get_iccad22_option(argparse::ArgumentParser& parser) {
    parser.add_argument("input_path").default_value(string("null")).help("input file path");
    parser.add_argument("output_path").default_value(string("null")).help("output file path");
}

argparse::ArgumentParser get_option(int argc, char* argv[]) {
    argparse::ArgumentParser parser("place");
    parser.add_argument("--load_stage").default_value(string("")).help("version");
    parser.add_argument("--load_file").default_value(string("")).help("version");
    parser.add_argument("--load_json").default_value(string("")).help("version");
    parser.add_argument("--version").default_value(string("22")).help("version");
    parser.add_argument("--fp").default_value(string("false")).help("wherther to use floorplan (true/false)");
    parser.add_argument("--num_part").default_value(int(50)).help("-num_part").scan<'i', int>();
    parser.add_argument("--min_gp_step").default_value(int(800)).help("-min_gp_step").scan<'i', int>();
    parser.add_argument("--min_gp3d_record_step").default_value(int(400)).help("-min_gp_step").scan<'i', int>();
    parser.add_argument("--is_fp_permit_change_cross_chip").default_value(string("false")).help("true/false");
    parser.add_argument("--is_move_macro_3d").default_value(string("false")).help("true/false");
    parser.add_argument("--gp_padding").default_value(double(0.06)).help("gp_padding").scan<'g', double>();
    parser.add_argument("--macro_hpwl_scale").default_value(double(1)).help("gp_padding").scan<'g', double>();
    parser.add_argument("--adjust_macro").default_value(string("false")).help("true/false");
    parser.add_argument("--use_greedy_place_in_fp").default_value(string("false")).help("true/false");
    parser.add_argument("--half_filler_height").default_value(string("false")).help("true/false");

    // TODO: ICCAD 2022 only, remove me after contest
    get_iccad22_option(parser);
    // general setting
    parser.add_argument("--dataset").default_value(string("ispd2005")).help("dataset name");
    parser.add_argument("--design_name").default_value(string("adaptec1")).help("design name");
    parser.add_argument("--load_from_raw").default_value(string("true")).help("load from given design");
    parser.add_argument("--gpu").default_value(int(0)).help("gpu id").scan<'i', int>();
    parser.add_argument("--num_threads").default_value(int(2)).help("#threads").scan<'i', int>();

    // logging and saver
    parser.add_argument("--log_verbose").default_value(string("true")).help("log to file");
    parser.add_argument("--verbose").default_value(string("true")).help("verbose");
    parser.add_argument("--result_dir").default_value(string("test")).help("log root directory");
    parser.add_argument("--log_dir").default_value(string("log")).help("log directory");
    parser.add_argument("--log_name").default_value(string("test.log")).help("log file name");
    parser.add_argument("--log_freq").default_value(int(100)).help("#log freq").scan<'i', int>();
    parser.add_argument("--draw_placement").default_value(string("false")).help("draw_placement");
    parser.add_argument("--enable_rotate_in_gp").default_value(string("false")).help("enable_rotate_in_gp");

    // model params
    parser.add_argument("--lr").default_value(double(0.01)).help("learning rate").scan<'g', double>();
    parser.add_argument("--inner_iter").default_value(int(3000)).help("#inner iters").scan<'i', int>();
    parser.add_argument("--inner_iter_gp3d").default_value(int(3000)).help("#inner iters").scan<'i', int>();
    parser.add_argument("--num_bin_x").default_value(int(512)).help("#binX").scan<'i', int>();
    parser.add_argument("--num_bin_y").default_value(int(512)).help("#binY").scan<'i', int>();
    parser.add_argument("--num_bin_z").default_value(int(10)).help("#binY").scan<'i', int>();
    parser.add_argument("--target_density").default_value(double(2)).help("placement target density").scan<'g', double>();
    parser.add_argument("--via_target_density").default_value(double(1)).help("placement target density").scan<'g', double>();
    parser.add_argument("--use_filler").default_value(string("true")).help("use filler");
    parser.add_argument("--noise_ratio").default_value(double(0.025)).help("noise ratio for init").scan<'g', double>();
    parser.add_argument("--ignore_net_degree").default_value(int(100)).help("threshold of net degree to ignore in wl calculation").scan<'i', int>();
    parser.add_argument("--scale_design").default_value(string("false")).help("scale_design");
    parser.add_argument("--use_precond").default_value(string("true")).help("apply precond");
    parser.add_argument("--enable_skip_update").default_value(string("true")).help("enable skip update");
    parser.add_argument("--clamp_node").default_value(string("true")).help("enable clamp_node");
    parser.add_argument("--loss_type").default_value(string("direct")).help("Loss Type");
    parser.add_argument("--net_type").default_value(string("monon")).help("Net Type");
    parser.add_argument("--den_type").default_value(string("step")).help("density map type");
    parser.add_argument("--block_row").default_value(string("false")).help("");
    parser.add_argument("--site_width").default_value(int(-1)).help("").scan<'i', int>();
    parser.add_argument("--sideline").default_value(double(0)).help("").scan<'g', double>();

    // params scheduler
    parser.add_argument("--magic_hpwl").default_value(int(350000)).help("").scan<'i', int>();
    parser.add_argument("--step_precond_coef").default_value(string("true")).help("");
    parser.add_argument("--density_weight").default_value(double(8e-5)).help("the weight of density loss").scan<'g', double>();
    parser.add_argument("--density_weight_coef").default_value(double(1.05)).help("the ratio of density_weight").scan<'g', double>();
    parser.add_argument("--wa_coeff").default_value(double(4)).help("wa coeff").scan<'g', double>();
    parser.add_argument("--stop_overflow").default_value(double(0.07)).help("stop overflow").scan<'g', double>();
    parser.add_argument("--stop_overflow_via").default_value(double(0.01)).help("stop overflow via").scan<'g', double>();
    parser.add_argument("--use_precond").default_value(string("true")).help("apply precond");
    parser.add_argument("--enable_skip_update").default_value(string("true")).help("enable skip update'");
    parser.add_argument("--quad_penalty").default_value(string("false")).help("quadratic penalty to accelerate gp");
    parser.add_argument("--quad_coeff").default_value(double(2000)).help("quad penalty").scan<'g', double>();
    parser.add_argument("--early_stop_check_plateau").default_value(string("true")).help("check plateau");

    // flow params
    parser.add_argument("--gp").default_value(string("true")).help("global placement");

    // partition params
    parser.add_argument("--partitioner").default_value(string("gp2d_grid")).help("1.fm; 2.patoh");
    parser.add_argument("--num_cuts").default_value(int(-1)).help("#cuts required").scan<'i', int>();
    parser.add_argument("--num_folds").default_value(double(2)).help("# folds in slicing").scan<'g', double>();
    parser.add_argument("--num_grids").default_value(double(2)).help("# grids in slicing").scan<'g', double>();
    parser.add_argument("--slice_direction").default_value(int(2)).help("0.x; 1.y").scan<'i', int>();
    parser.add_argument("--soft_margin").default_value(double(1)).help("soft margin for fm").scan<'g', double>();
    parser.add_argument("--pin_propagation").default_value(string("false")).help("add external pins at grids");
    parser.add_argument("--pt_imbl").default_value(double(0.03)).help("imbalance for partition").scan<'g', double>();
    parser.add_argument("--pt_model").default_value(string("null")).help("gp2d gp model");
    parser.add_argument("--dynamic_ratio").default_value(string("false")).help("dynamical pt ratio");
    parser.add_argument("--mononlithic").default_value(string("false")).help("global pt with multi-constraints");
    parser.add_argument("--global_const").default_value(int(1)).help("#global const: 0/1").scan<'i', int>();

    // dp
    parser.add_argument("--save_model").default_value(string("false")).help("save to .pt");
    parser.add_argument("--gp_model").default_value(string("null")).help("load from .pt");
    parser.add_argument("--lg_model").default_value(string("null")).help("load from .pt");
    parser.add_argument("--eval_params").default_value(string("false")).help("");
    parser.add_argument("--lg").default_value(string("true")).help("legalization");
    parser.add_argument("--dp").default_value(string("true")).help("detailed placement");
    parser.add_argument("--pp").default_value(string("true")).help("post process");
    parser.add_argument("--rf").default_value(string("false")).help("gp refinments");
    parser.add_argument("--sw").default_value(string("false")).help("gp swap");
    parser.add_argument("--via_dp").default_value(string("true")).help("via lg/dp");
    parser.add_argument("--disp_coef").default_value(double(2)).help("displacement coeff").scan<'g', double>();
    parser.add_argument("--lg_ver").default_value(int(2)).help("legalizer version").scan<'i', int>();
    parser.add_argument("--round_recursion").default_value(int(0)).help("").scan<'i', int>();
    parser.add_argument("--net_cut_str_thrs").default_value(int(0)).help("").scan<'i', int>();
    parser.add_argument("--strengthen_via_density").default_value(double(-1)).help("displacement coeff").scan<'g', double>();
    parser.add_argument("--strengthen_net_wa_coef").default_value(double(0)).help("displacement coeff").scan<'g', double>();
    parser.add_argument("--worse_legalize").default_value(string("false")).help("");

    // 3D GP params
    parser.add_argument("--pt").default_value(string("true")).help("partition");
    parser.add_argument("--use_filler_3d").default_value(string("false")).help("use filler");
    parser.add_argument("--cut_net_thres").default_value(int(2)).help("thres for wa_z weight").scan<'i', int>();
    parser.add_argument("--net_weight_offset").default_value(double(0)).help("wa_z weight offset").scan<'g', double>();
    parser.add_argument("--net_weight_coef").default_value(double(1)).help("wa_z weight").scan<'g', double>();
    parser.add_argument("--num_den_layer").default_value(int(3)).help("#deensity layer in gp").scan<'i', int>();
    parser.add_argument("--use_pre_gp").default_value(string("false")).help("whether load previous sol");
    parser.add_argument("--use_pre_pt").default_value(string("false")).help("whether load previous pt");
    parser.add_argument("--shrink_size").default_value(double(1)).help("shrink coef in 3d GP").scan<'g', double>();
    parser.add_argument("--match_2pin_nets").default_value(string("false")).help("stack 2-pin nets");
    parser.add_argument("--force_coeff_2d").default_value(double(0)).help("force_coeff in x/y").scan<'g', double>();
    parser.add_argument("--wa_coeff_wa_z").default_value(double(1)).help("wa grad in z").scan<'g', double>();
    parser.add_argument("--wa_coeff_wa_xy").default_value(double(1)).help("wa grad in xy").scan<'g', double>();
    parser.add_argument("--correlate_bbox").default_value(string("false")).help("");
    parser.add_argument("--bbox_correlation").default_value(double(20)).help("").scan<'g', double>();
    parser.add_argument("--wa_z_model").default_value(string("xy")).help("wa hpwl z model");
    parser.add_argument("--top_util_filler").default_value(double(0.00)).help("top extra density").scan<'g', double>();
    parser.add_argument("--clamp_util").default_value(string("true")).help("");
    parser.add_argument("--stop_overflow_3d").default_value(double(0.15)).help("").scan<'g', double>();
    parser.add_argument("--select_nets").default_value(string("false")).help("");
    parser.add_argument("--weaken_net_size").default_value(double(0.1)).help("").scan<'g', double>();
    parser.add_argument("--weaken_net_wa_coef").default_value(double(0.1)).help("").scan<'g', double>();
    parser.add_argument("--net_weight_type").default_value(string("step")).help("");
    parser.add_argument("--filler_type").default_value(string("center")).help("");
    parser.add_argument("--kernel_size").default_value(int(2)).help("Gaussian kernel size").scan<'i', int>();
    parser.add_argument("--num_bin_3d").default_value(int(1)).help("").scan<'i', int>();
    parser.add_argument("--visualize_curve").default_value(string("false")).help("");
    parser.add_argument("--fmwl_iter").default_value(int(1)).help("").scan<'i', int>();
    parser.add_argument("--fmwl_area_coef").default_value(double(10)).help("").scan<'g', double>();
    parser.add_argument("--local_density_weight").default_value(string("false")).help("");
    parser.add_argument("--stack_cells").default_value(int(-1)).help("").scan<'i', int>();
    parser.add_argument("--skip_gp3d").default_value(string("false")).help("");
    parser.add_argument("--skip_hpwl_fm").default_value(string("false")).help("");
    parser.add_argument("--skip_draw").default_value(string("false")).help("");
    parser.add_argument("--first_magic_hpwl").default_value(int(350000)).help("").scan<'i', int>();
    parser.add_argument("--second_num_bin_x").default_value(int(512)).help("").scan<'i', int>();
    parser.add_argument("--second_num_bin_y").default_value(int(1024)).help("").scan<'i', int>();


    // others
    parser.add_argument("--draw_mat_size").default_value(double(1)).help("all use").scan<'g', double>();
    parser.add_argument("--omni_int").default_value(int(0)).help("all use'").scan<'i', int>();
    parser.add_argument("--omni_float").default_value(double(1)).help("all use").scan<'g', double>();

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    // cout << parser["--load_from_raw"];
    // cout << parser["--dataset"];
    // cout << parser.get("--dataset") << endl;
    // if (parser["--load_from_raw"] == true) {
    //     cout << "fucker\n";
    // }
    std::string input_path = parser.get<std::string>("input_path");
    std::string design_name;

    std::string num_bin_3d = std::to_string(parser.get<int>("num_bin_3d"));
    std::string wa_z_model = parser.get<std::string>("wa_z_model");

    double num_grids_raw = parser.get<double>("num_grids");
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << num_grids_raw;
    std::string num_grids = stream.str();

    double force_coeff_2d_raw = parser.get<int>("cut_net_thres");
    std::stringstream stream1;
    stream1 << std::fixed << std::setprecision(2) << force_coeff_2d_raw;
    std::string force_coeff_2d = stream1.str();

    double wa_coeff_wa_xy_raw = parser.get<double>("net_weight_coef");
    std::stringstream stream2;
    stream2 << std::fixed << std::setprecision(2) << wa_coeff_wa_xy_raw;
    std::string wa_coeff_wa_xy = stream2.str();

    double wa_coeff_wa_z_raw = parser.get<double>("net_weight_offset");
    std::stringstream stream3;
    stream3 << std::fixed << std::setprecision(2) << wa_coeff_wa_z_raw;
    std::string wa_coeff_wa_z = stream3.str();

    if (input_path != "null") {  // TODO:
        std::size_t pos = input_path.find("B_");
        std::string tmp = input_path.substr(pos + 2);
        std::size_t pos1 = tmp.find(".txt");
        design_name = tmp.substr(0, pos1);
    } else
        design_name = parser.get<std::string>("--design_name");
    std::time_t rawtime;
    std::tm* timeinfo;
    char buffer[80];
    std::time(&rawtime);
    timeinfo = std::localtime(&rawtime);
    std::strftime(buffer, 80, "%Y_%m%d_%H%M%S", timeinfo);
    std::string stamp(buffer);
    std::string design_id = stamp + "_" + design_name + "_" + num_grids + "_" + wa_z_model + "_" + num_bin_3d + "_" + force_coeff_2d + "_" + wa_coeff_wa_xy + "_" + wa_coeff_wa_z;
    parser.add_argument("--exp_id").default_value(design_id).help("experiment id");

    st::setting.parse(parser);
    if (st::setting.verbose) utils::verbose_parser_log = true;

    return parser;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    std::cout << std::boolalpha;  // set std::boolalpha to std::cout
    srand(0);
    // torch::manual_seed(3407);
    torch::manual_seed(0);

    argparse::ArgumentParser args = get_option(argc, argv);
    logger.setup_logger(args);

    string cmd = "";
    for (int i = 0; i < argc; ++i) {
        cmd += argv[i];
        cmd += " ";
    }
    logger.info("\n%s", cmd.c_str());

    // std::cout << st::setting << "\n";
    logger.dump_setting();

    logger.info("------------------------------------------------------------------------------");
    logger.info("                               Global Placement                               ");
    logger.info("------------------------------------------------------------------------------");
    // run_placement_main()
    run_placement_main_multi_circuit();
    // if (st::setting.eval_params)
    //     run_placement_main_nesterov();
    // else if (st::setting.dataset == "iccad2022") {
    //     run_placement_main_multi_circuit();
    // } else
    //     run_placement_main_nesterov();
    logger.info("------------------------------------------------------------------------------");
    logger.info("                               Terminate...                                   ");
    logger.info("------------------------------------------------------------------------------");

    return 0;
}