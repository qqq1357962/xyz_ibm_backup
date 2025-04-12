
#include "gp3d/core3d/core.h"
#include "gp3d/optim3d/Nesterov.h"
#include "gp3d/optim3d/ParamScheduler.h"
#include "gp3d/placer3d/calculator.h"
#include "gp3d/placer3d/database3d.h"
#include "gp3d/placer3d/evaluator.h"
#include "gp3d/placer3d/initializer.h"
#include "partition.h"
#include "patoh.h"
#include "placer/visualization.h"

void saveTensorToTxt(const torch::Tensor& tensor, const std::string& filename) {
    // 确保Tensor是CPU上的
    torch::Tensor cpu_tensor = tensor.to(torch::kCPU);

    // 打开文件
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // 获取Tensor的尺寸
    auto sizes = cpu_tensor.sizes();
    int num_dims = sizes.size();

    // 写入Tensor的尺寸信息
    file << "Shape: ";
    for (int i = 0; i < num_dims; ++i) {
        file << sizes[i] << " ";
    }
    file << "\n";

    // 写入Tensor的数据
    file << "Data:\n";
    auto accessor = cpu_tensor.accessor<float, 2>(); // 假设Tensor是1维的
    for (int i = 0; i < sizes[0]; ++i) {
        for (int j = 0; j < sizes[1]; ++j) {
            file << accessor[i][j] << " ";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Tensor saved to " << filename << std::endl;
}

tuple<torch::Tensor, torch::Tensor> Partitioner::run_gp3d(NodeData& data_2d) {
    logger.info("============= Running GP3D ============");

    // ======================================================================================================
    //
    //                                            GP3D
    //
    // ======================================================================================================
    /* initialization */
    auto device = data.device;
    st::setting.early_stop_check_plateau = false;
    torch::Tensor node_pos;

    // if (st::setting.force_coeff_2d > 0) st::setting.net_weight_coef = 0;

    st::setting.density_weight = 1e-5;
    // st::setting.density_weight_coef = 1.05;
    st::setting.wa_coeff = 2.5;
    st::setting.quad_penalty = false;
    // st::setting.magic_hpwl = st::setting.magic_hpwl / 2;

    logger.info("Optimizer info: density_weight: %3E | wa_coef %.2f | magic_hpwl %d",
                st::setting.density_weight,
                st::setting.wa_coeff,
                st::setting.magic_hpwl);

    NodeData3D data(data_2d);

    data.preprocess();
    data.init_filler();

    torch::Tensor mov_node_pos;
    torch::Tensor mov_node_size;
    torch::Tensor expand_ratio;
    if (st::setting.mononlithic)
        std::tie(mov_node_pos, mov_node_size, expand_ratio) = data.get_mov_node_info_with_via();
    else
        std::tie(mov_node_pos, mov_node_size, expand_ratio) = data.get_mov_node_info();
    auto [mov_lhs, mov_rhs] = data.movable_index;
    auto init_density_map = data.init_density_map.to(device);

#ifdef DEBUG
    /* Cell Alignment */
    if (data_2d.node_wgt_grad.numel()) {
        int min_cell_mov_rhs = data.cell_mov_rhs;
        logger.info("load from prev sol of size %d...", min_cell_mov_rhs);
        mov_node_pos.index({Slice(0, min_cell_mov_rhs), Slice(0, 2)})
            .data()
            .copy_(data_2d.node_pos.index({Slice(0, min_cell_mov_rhs), Slice(0, 2)}).data());
        logger.info("load from prev partition ...");
        for (int i = 0; i < min_cell_mov_rhs; i++) {
            mov_node_pos[i][2] = data.__ori_die_hz__ / 2 * (0.5 + data_2d.node_die[i]);  // TODO:;
        }

        auto cell_mov_node_weight = torch::_cast_Float(data_2d.node_wgt_grad > 0.45);
        for (int i = 0; i < min_cell_mov_rhs; i++) {
            if (cell_mov_node_weight[i].item<int>() == 1) mov_node_pos[i][2] = data.__ori_die_hz__ / 2 * (1);  // TODO:;
        }

        logger.info("%d cells are re-placed", cell_mov_node_weight.sum().item<int>());
        auto node_weight = torch::ones({mov_node_pos.size(0)}, dtype(mov_node_pos.dtype()));
        data.node_wgt_grad = torch::cat({cell_mov_node_weight, node_weight.index({Slice(data.cell_mov_rhs, None)})}, 0);

        data.mov_node_weights[0] =
            torch::cat({cell_mov_node_weight, data.mov_node_weights[0].index({Slice(data.cell_mov_rhs, None)})}, 0);
        auto fix_node_weight = (1 - data.mov_node_weights[0]);

        auto init_node_size = data.node_size.clone();
        init_node_size.index({"...", 1}) *= data.node_util_weight_y.index_select(0, node_die);
        init_node_size.index({"...", 0}) *= data.node_util_weight_x.index_select(0, node_die);

        torch::Tensor zeros_density_map =
            torch::zeros({data.num_bin_x, data.num_bin_y, data.num_bin_z}, torch::dtype(init_node_size.dtype()))
                .to(device);
        init_density_map = GP3D::density_map_forward_naive(mov_node_pos.to(device),
                                                           init_node_size.to(device),
                                                           fix_node_weight.to(device),
                                                           data.unit_len.to(device),
                                                           zeros_density_map.to(device),
                                                           data.num_bin_x,
                                                           data.num_bin_y,
                                                           data.num_bin_z,
                                                           mov_node_pos.size(0),
                                                           -1.0,
                                                           -1.0,
                                                           1e-4,
                                                           false)
                               .to(device);
        logger.info("Cell preplaced and fixed on one chip");
    }

    /* 2D placement net bounding box information */
    if (data_2d.net_wgt_grad.numel()) {
        int count = 0;
        for (int i = 0; i < num_nets; i++)
            if (data_2d.net_wgt_grad[i].item<float>() < st::setting.strengthen_via_density &&
                data.net_to_num_pins[i].item<int>() <= 3) {
                data.net_weight[i] = st::setting.strengthen_via_density;
                count++;
            }
        logger.info("via density less than %.3f are strengthened to %.3f, %d nets are strengthened",
                    st::setting.strengthen_via_density,
                    st::setting.strengthen_via_density,
                    count);
    }
#endif  // DEBUG

    /* select nets */
    torch::Tensor box_len = torch::zeros({num_nets, 2}, dtype(torch::kFloat));
    auto net_size_x = torch::_cast_Float(torch::zeros(data.net_to_num_pins.max().item<int>() + 1));
    auto net_size_y = torch::_cast_Float(torch::zeros(data.net_to_num_pins.max().item<int>() + 1));
    if (st::setting.select_nets && node_pos_2d_ground.numel()) {
        torch::Tensor pin_pos = wa_wirelength_hpwl::nodePosToPinPos(
            node_pos_2d_ground.index({Slice(data_2d.cell_mov_lhs, data.cell_mov_rhs)}).to(device),
            data_2d.pin_id2node_id.to(device),
            data_2d.pin_rel_cpos.to(device));
        box_len = wa_wirelength_hpwl::get_hpwl(data, pin_pos).to(torch::kCPU);
        box_len.index({"...", 0}) /= data.die_info[1].item<float>();
        box_len.index({"...", 1}) /= data.die_info[3].item<float>();
        const torch::TensorAccessor<float, 2> box_len_raw = box_len.accessor<float, 2>();
        float max_box = box_len.max().item<float>();

        for (int i = 0; i < data.num_nets; i++) {
            float box_w = box_len_raw[i][0] + box_len_raw[i][1];
            int num_pins = net_to_num_pins[i].item<int>();
            float prob_not_cut = pow((float)0.5, num_pins - 1);
            float prob_cut = 1 - prob_not_cut;

            // data.net_weight[i] = (box_w * prob_cut) * st::setting.net_weight_coef;
            data.net_weight[i] *= (box_w * prob_cut);

            // if (num_pins >= st::setting.cut_net_thres) data.net_weight[i] = st::setting.net_weight_coef;
        }
        logger.info("Net weight updated, avg weight: %.4f", data.net_weight.mean().item<float>());
    }

    // if (!data_2d.node_wgt_grad.numel() && (st::setting.use_pre_gp || node_pos_2d_ground.numel())) {
    //     int min_cell_mov_rhs = min(node_pos_2d_ground.size(0), mov_node_pos.size(0));
    //     logger.info("load from prev sol of size %d...", min_cell_mov_rhs);
    //     mov_node_pos.index({Slice(0, min_cell_mov_rhs), Slice(0, 2)})
    //         .data()
    //         .copy_(node_pos_2d_ground.index({Slice(0, min_cell_mov_rhs), Slice(0, 2)}).data());

        // if (node_pos_2d_ground.size(1) == 3) {
        //     logger.info("load from prev sol[2] ...");
        //     mov_node_pos.index({Slice(0, min_cell_mov_rhs), 2})
        //         .data()
        //         .copy_(node_pos_2d_ground.index({Slice(0, min_cell_mov_rhs), 2}).data());
        // } else if (data_2d.node_die.numel()) {
        //     logger.info("load from prev partition ...");
        //     for (int i = 0; i < min_cell_mov_rhs; i++) {
        //         // mov_node_pos[i][2] = data.__ori_die_hz__ / 2 * (0.5 + data_2d.node_die[i]);  // TODO:; //1/4 or 3/4
        //         if (data_2d.macro_mask[i].item<int>() == 1) {
        //             mov_node_pos[i][2] = data.__ori_die_hz__ / 2 * (0.5 + 1);
        //             // logger.info("macro %d on die %d", i, data_2d.node_die[i].item<int>());
        //         }
        //     }
        // }
    // }

    // int min_cell_mov_rhs = min(node_pos_2d_ground.size(0), mov_node_pos.size(0));
    // mov_node_pos.index({Slice(0, min_cell_mov_rhs), Slice(0, 1)}) = data.__ori_die_hx__ / 2;
    // mov_node_pos.index({Slice(0, min_cell_mov_rhs), Slice(1, 2)}) = data.__ori_die_hy__ / 2;
    // mov_node_pos.index({Slice(0, min_cell_mov_rhs), Slice(2, 3)}) = data.__ori_die_hz__ / 2;
    // auto rand_shift = torch::randn_like(mov_node_pos);
    // rand_shift .index({Slice(0, min_cell_mov_rhs), Slice(0, 1)}) *= data.__ori_die_hx__ * 0.001;
    // rand_shift .index({Slice(0, min_cell_mov_rhs), Slice(1, 2)}) *= data.__ori_die_hy__ * 0.001;
    // rand_shift .index({Slice(0, min_cell_mov_rhs), Slice(2, 3)}) *= data.__ori_die_hz__ * 0.001;
    // mov_node_pos = mov_node_pos + rand_shift;

    // if (st::setting.stack_cells >= 0) {
    //     for (int i = 0; i < data_2d.num_nodes; i++)
    //         if (data_2d.macro_mask[i].item<int>()==0&&data_2d.stack_cells[i].item<bool>())
    //             mov_node_pos[i][2] = data.__ori_die_hz__ / 2 * (0.5 + st::setting.stack_cells);  // TODO:;
    // }

    mov_node_pos = mov_node_pos.to(data.device).requires_grad_(true);
    mov_node_size = mov_node_size.to(data.device);
    expand_ratio = expand_ratio.to(data.device);
    data.to(data.device);

    /* trunc nodes to core */  // TODO:
    torch::Tensor node_pos_lb = mov_node_size / 2 + data.sidelines_ll + 1e-4;
    torch::Tensor node_pos_ub = data.sidelines_ur - mov_node_size / 2 + data.die_ll - 1e-4;
    // if (st::setting.stack_cells >= 0) {
    //     for (int i = 0; i < data_2d.num_nodes; i++)
    //         if (data_2d.macro_mask[i].item<int>()==0&&data_2d.stack_cells[i].item<bool>()) {
    //             node_pos_lb[i][2] =
    //                 data.__ori_die_hz__ / 2 * (0.5 / data.shrink_size + st::setting.stack_cells);  // TODO:;
    //             node_pos_ub[i][2] =
    //                 data.__ori_die_hz__ / 2 * (0.5 / data.shrink_size + st::setting.stack_cells);  // TODO:;
    //         }
    // }
    std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn = [&node_pos_lb, &node_pos_ub](torch::Tensor x) {
        x.data().clamp_(node_pos_lb, node_pos_ub);
        return x;
    };

    vector<GP3D::ElectronicDensityLayer> density_map_layers;
    /* overflow function */
    std::function<torch::Tensor(torch::Tensor)> overflow_fn = [&data](torch::Tensor mov_density_map) {
        torch::Tensor overflow_ori =
            ((mov_density_map - data.target_density) * data.bin_area).clamp_(0.0).sum(0).sum(0);
        // overflow_ori: 10 dimensions
        torch::Tensor overflows = torch::zeros({2}, torch::dtype(mov_density_map.dtype()));
        overflows[0] = overflow_ori.index({Slice(0, data.num_bin_z / 2)}).sum();
        overflows[1] = overflow_ori.index({Slice(data.num_bin_z / 2, None)}).sum();

        torch::Tensor overflow_all =
            (mov_density_map * data.bin_area).clamp_(0.0).sum(0).sum(0);
        // overflow_ori: 10 dimensions
        torch::Tensor overflows_all = torch::zeros({2}, torch::dtype(mov_density_map.dtype()));
        overflows_all[0] = overflow_all.index({Slice(0, data.num_bin_z / 2)}).sum();
        overflows_all[1] = overflow_all.index({Slice(data.num_bin_z / 2, None)}).sum();

        float target_density_xy = 2;
        float height_cells = data.num_bin_z / 2;
        auto density_map_xy = mov_density_map.sum(2);
        torch::Tensor overflow_xy =
            ((density_map_xy/height_cells - target_density_xy) * data.bin_area * height_cells).clamp_(0.0).sum(0).sum(0);
        overflow_xy/=(data.total_mov_cell_areas[0]*2);
        //@@ cout<<"overflow_xy: "<<overflow_xy.item<float>()<<endl;

        return (overflows / overflows_all);
    };

    GP3D::ParamScheduler ps = GP3D::ParamScheduler(data);

    auto overflow_helper = make_tuple(mov_lhs, mov_rhs, overflow_fn);
    vector<int> macro_list;
    for (int i = 0; i < data.cell_mov_rhs; i++) {
        if (data.macro_mask[i].item<int>() == 1) {
            macro_list.push_back(i);
        }
    }
    auto density_map_layer = GP3D::ElectronicDensityLayer(data.unit_len,
                                                          data.num_bin_x,
                                                          data.num_bin_y,
                                                          data.num_bin_z,
                                                          device,
                                                          overflow_helper,
                                                          expand_ratio,
                                                          data.sorted_maps,
                                                          data.macro_mask);
    density_map_layer.die_info = data.die_info;
    density_map_layer.node_util_weight = data.node_util_weight;  // TODO:
    density_map_layer.node_util_weight_x = data.node_util_weight_x;
    density_map_layer.node_util_weight_y = data.node_util_weight_y;
    density_map_layer.ratio_difference = data.ratio_difference;//@@@@@@@@@@
    density_map_layer.density_weight_local = ps.density_weight_local.to(device);
    // density_map_layer.node_die = node_die;
    density_map_layers.push_back(density_map_layer);

    /* parameteer scheduler */
    // ps.density_weight_map = torch::ones_like(init_density_map); // TODO:
    ps.min_stop_iter = st::setting.min_stop_iter;
    float step_ovfl = 1;

    /* objective function */
    torch::Tensor conn_fix_node_pos = data.node_pos.new_empty({0, 3});
    torch::Tensor init_fix_node_pos = data.node_pos.index({Slice(data.iopin_mov_lhs, data.iopin_mov_rhs), "..."}).clone();
    init_fix_node_pos.select(1, 0).fill_((data.__ori_die_lx__ + data.__ori_die_hx__) / 2);
    init_fix_node_pos.select(1, 1).fill_((data.__ori_die_ly__ + data.__ori_die_hy__) / 2);
    if (data.iopin_mov_lhs < data.iopin_mov_rhs) {
        auto lhs = data.iopin_mov_lhs;
        auto rhs = data.iopin_mov_rhs;
        conn_fix_node_pos = data.node_pos.index({Slice(lhs, rhs), "..."}) * min((1 - step_ovfl) * 2, static_cast<float>(1)) + init_fix_node_pos * max(1 - (1 - step_ovfl) * 2, static_cast<float>(0));
    }
    conn_fix_node_pos = conn_fix_node_pos.detach();
    auto mov_node_size_top = data_2d.node_size_top.to(data.device);
    auto mov_node_size_bot = data_2d.node_size_bot.to(data.device);
    auto node_slide_grad =
        torch::zeros(mov_node_size_top.size(0), torch::dtype(data.node_pos.dtype()).device(data.device));
    auto node_slide_state = torch::zeros(mov_node_size_top.size(0), torch::dtype(data.node_pos.dtype()).device(data.device)) + data.macro_mask.to(data.node_pos.dtype()) * 0.5;
    std::function<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor)> obj_and_grad_fn =
        [&trunc_node_pos_fn,
         &mov_node_size,
         &init_density_map,
         &density_map_layers,
         &conn_fix_node_pos,
         &ps,
         &data,
         &macro_list,
         &data_2d,
         &mov_node_size_top,
         &mov_node_size_bot,
         &node_slide_grad,
         &node_slide_state
         ](at::Tensor mov_node_pos) {
            auto node_pos_channel = mov_node_pos.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs), 2});
            auto ratio = node_pos_channel / float(data.__ori_die_hz__);
            ratio = (-(1.5 + 2 * data.macro_mask) + ratio * (4 + data.macro_mask * 4)).unsqueeze(1);;
            ratio = ratio.clamp(0,1);
            auto current_mov_node_size = mov_node_size_top * ratio + mov_node_size_bot * (1-ratio);
            auto current_mov_node_length =
                torch::sqrt(current_mov_node_size.select(1, 0) * current_mov_node_size.select(1, 1));
            auto [loss, grad, slide_grad] = GP3D::calc_obj_and_grad(mov_node_pos,
                                                        trunc_node_pos_fn,
                                                        mov_node_size,
                                                        init_density_map,
                                                        density_map_layers,
                                                        conn_fix_node_pos,
                                                        ps,
                                                        data,
                                                        (data_2d.node_die) * st::setting.patoh_guide_ratio,
                                                        node_slide_state);

            node_slide_grad = slide_grad / current_mov_node_length;
            if (!st::setting.move_macro_3d) {
                for (auto macro_id : macro_list) {
                    grad[macro_id] = 0;
                }
            }
            if (true) {
                grad.index({torch::indexing::Slice(data.cell_mov_rhs, data.cell_mov_rhs + static_cast<int>(data.__num_fillers__ * (1 - st::setting.die_diff))), torch::indexing::Slice(2, 3)}) = 0.0;
                grad.index({torch::indexing::Slice(data.iopin_mov_lhs, data.iopin_mov_rhs), torch::indexing::Slice(0, 2)}) = 0.0;
            }
            // for (auto macro_id : macro_list) {
            //     grad[macro_id][2] = 0;
            // }
            return std::make_tuple(loss, grad);
        };

    /* evaluation function */
    std::function<tuple<torch::Tensor, torch::Tensor, torch::Tensor>(torch::Tensor)> evaluator_fn =
        [&trunc_node_pos_fn, &mov_node_size, &init_density_map, &density_map_layers, &conn_fix_node_pos, &ps, &data, &node_slide_state](
            at::Tensor mov_node_pos) {
            return GP3D::fast_evaluator(mov_node_pos,
                                        trunc_node_pos_fn,
                                        mov_node_size,
                                        init_density_map,
                                        density_map_layers[0],
                                        conn_fix_node_pos,
                                        ps,
                                        data,
                                        node_slide_state);
        };
    // FIXME: for quitting gp3d when init_lr nan
    torch::Tensor node_pos_2d_backup = mov_node_pos.clone();
    /* Nesterov optimizer */
    auto optimizer =
        torch::optim::GP3D::Nesterov({mov_node_pos}, torch::optim::GP3D::NesterovOptions(0.0), obj_and_grad_fn);
    GP3D::init_params(mov_node_pos,
                      trunc_node_pos_fn,
                      mov_lhs,
                      data.iopin_mov_lhs,
                      conn_fix_node_pos,
                      density_map_layers,
                      mov_node_size,
                      init_density_map,
                      optimizer,
                      ps,
                      data,
                      (data_2d.node_die) * st::setting.patoh_guide_ratio,
                      node_slide_state);
    // FIXME
    double init_lr = 1e5;
    if(!st::setting.skip_gp3d)
    {
        init_lr =
            GP3D::estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos, st::setting.lr);
    }
    if (init_lr < -0.5) {
        logger.info("Bug not fixed: init learning rate nan");
        // return node_pos_2d_ground.clone();
        // return node_pos_2d_backup;
    }
    logger.info("Init learning rate %.3E", init_lr);
    for (auto& group : optimizer.param_groups()) {
        auto& options = static_cast<torch::optim::GP3D::NesterovOptions&>(group.options());
        options.set_lr(init_lr);
    }

    if (!st::setting.skip_gp3d) {
        auto [hpwl, overflows, mov_density_map] = evaluator_fn(mov_node_pos);
        logger.info(
            "iter: %-5d | masked_hpwl/2D: %.2E, %.2E | overflows:<%.4f, %.4f> "
            "density_weight: %.4E wa_coeff: %.4E",
            0,
            // hpwl.index({"...", 2}).sum().item<float>(),
            hpwl.sum().item<float>(),
            hpwl.index({"...", Slice(0, 2)}).sum().item<float>(),
            overflows[0].item<float>(),
            overflows[1].item<float>(),
            ps.density_weight,
            ps.wa_coeff);
    }

    int& iteration = st::setting.iteration;
    auto via_node_size_bkup = mov_node_size.index({Slice(data.cell_mov_rhs, mov_rhs)}).clone();
    logger.info("=========================================");
    logger.info("start gp 3d");
    iteration = 0;  // FIXME: 0 ? 1
    int roll_back_cnt = 0;
    // FIXME: skip this loop when init_lr nan

    if (true) {
        // auto nos = 1 - density_map_layers[0].node_die.index({Slice(data.cell_mov_lhs,
        // data.cell_mov_rhs)}).clone().to(torch::kCPU);

        auto node_pos = mov_node_pos.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)}).to(torch::kCPU);
        auto true_node_pos = mov_node_pos.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs), Slice(0, 2)}).to(torch::kCPU);
        auto node_shift = (data.__die_shift__.index({Slice(0, 2)}) / data.__die_scale__.index({Slice(0, 2)})).expand_as(true_node_pos).to(torch::kCPU);
        true_node_pos = true_node_pos + node_shift;
        // data.to(torch::kCPU);

        /* dump to partition result */
        int dir = st::setting.slice_direction;
        auto node_pos_channel = node_pos.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs), dir});  // fetch z-cord
        auto slicer = data.die_info[dir * 2 + 1].to(torch::kCPU) * 0.5;                              // FIXME:
        auto parter = (node_pos_channel > slicer);
        auto nos = torch::_cast_Int(parter);

        torch::Tensor node_size_bot = data_2d.node_size_bot * (1 - nos).unsqueeze(1);
        torch::Tensor node_size_top = data_2d.node_size_top * (nos).unsqueeze(1);

        auto info1 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_init_3D_0");
        draw_fig_with_cairo_cpp(
            true_node_pos.to(torch::kCPU),
            node_size_bot,
            data,
            info1);
        auto info2 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_init_3D_1");
        draw_fig_with_cairo_cpp(
            true_node_pos.to(torch::kCPU),
            node_size_top,
            data,
            info2);
        auto info3 = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_3D_2");
        auto node_pos_draw_cp =
            torch::cat({true_node_pos,
                        true_node_pos},
                       0)
                .to(torch::kCPU);
        auto node_size_draw_cp = torch::cat({node_size_bot, node_size_top}, 0);
        draw_fig_with_cairo_cpp_cross_chip(node_pos_draw_cp, node_size_draw_cp, data, info3);
    }

    auto macro_mask_2d = data.macro_mask.unsqueeze(1);

    // FIXME: skip this loop when init_lr nan
    if (st::setting.skip_gp3d) {
        logger.warning("Skip gp3d.");
    }
    bool need_stop=false;
    double stop_threshold = 1e25;
    float mid_z = (data.die_info[5].item<float>() + data.die_info[4].item<float>()) / 2;

    for (iteration = 1; iteration < st::setting.inner_iter_gp3d && init_lr > 0 && !st::setting.skip_gp3d; iteration++) {
        // for (iteration = 1; iteration < 0 && init_lr > 0; iteration++) {
        torch::Tensor obj = optimizer.step();
        conn_fix_node_pos = data.node_pos.index({Slice(data.iopin_mov_lhs, data.iopin_mov_rhs), "..."}) * min((1 - step_ovfl) * 1.4, static_cast<double>(1)) + init_fix_node_pos * max(1 - (1 - step_ovfl) * 1.4, static_cast<double>(0));
        // cout << conn_fix_node_pos[0][1] << endl;
        conn_fix_node_pos = conn_fix_node_pos.detach();
        // auto mov_node_area = torch::prod(mov_node_size.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)}), 1) * expand_ratio.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)});
        // auto mask_bot = mov_node_pos.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)}).select(1, 2) < mid_z;
        // auto mask_top = mov_node_pos.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)}).select(1, 2) >= mid_z;

        // data.total_mov_cell_areas[0] = mov_node_area.masked_select(mask_bot).sum();
        // data.total_mov_cell_areas[1] = mov_node_area.masked_select(mask_top).sum();

        if (step_ovfl > 0.3) {
            node_slide_state = (node_slide_state + node_slide_grad).clamp(0, 1).detach();
        }
        else {
            auto node_slide_distance = node_slide_state - 0.5;
            auto node_slide_norm = (0.5 - torch::abs(node_slide_distance)).clamp(0, 0.5);
            node_slide_distance = (node_slide_distance + 1e-6) / (torch::abs(node_slide_distance) + 1e-6);
            node_slide_state = (node_slide_state + node_slide_grad * node_slide_norm + node_slide_distance * 0.08).clamp(0, 1).detach();
        }

        auto non_zero_indices = torch::nonzero(node_slide_grad);
        auto non_zero_num = torch::count_nonzero(node_slide_grad).item<int>();

        auto [hpwl, overflows, mov_density_map] = evaluator_fn(mov_node_pos);

        hpwl.index({"...", 2}) *= data.net_weight * ps.net_weight_coef;
        // float step_wl = hpwl.index({"...", Slice(0, 2)}).sum().item<float>();
        // float step_wl = hpwl.index({"...", 2}).sum().item<float>();
        float step_wl = hpwl.sum().item<float>();
        float step_wl_xy = hpwl.slice(1,0,2).sum().item<float>();
        step_ovfl = overflows.mean().item<float>();

        if (!ps.local_density_lock && st::setting.local_density_weight && step_ovfl < 0.2) {
            logger.info("Global density below threshold, start applying local density");
            st::setting.loss_type = "den_only";
            ps.reset();
            ps.local_density_lock = true;
        }
        ps.step(step_wl, step_wl_xy, step_ovfl, mov_node_pos);
        torch::Tensor node_density;
        if (ps.local_density_lock) {
            node_density = density_map_layers[0].density_grad(mov_node_pos.index({Slice(mov_lhs, mov_rhs), "..."}),
                                                              mov_node_size.index({Slice(mov_lhs, mov_rhs), "..."}),
                                                              mov_density_map);
            ps.step_local_density_weight(node_density);
        }

        // if (!roll_back_cnt && ps.need_to_early_stop()) {
        //     logger.info("Iteration %d, rolling back parameters...", iteration);
        //     ps.net_weight_coef = 1;
        //     st::setting.force_coeff_2d = 0;
        //     ps.reset();
        //     roll_back_cnt++;
        // }
        if(obj.item<float>()>stop_threshold || ps.density_weight>stop_threshold)
        {
            need_stop = true;
            logger.warning("stop because density weight or obj is too high obj: %.4E density weight:%.4E ",obj.item<float>(), ps.density_weight);
        }
        if (need_stop || (iteration - 1) % st::setting.log_freq == 0 || iteration == st::setting.inner_iter - 1 ||
            (iteration >= st::setting.minGPStep && ps.need_to_early_stop())) {
            logger.info(
                "iter: %-5d | masked_hpwl/2D: %.2E, %.2E | overflows:<%.4f, %.4f> obj: %.4E "
                "density_weight: %.4E wa_coeff: %.4E",
                iteration - 1,
                // hpwl.index({"..", 2}).sum().item<float>(),
                hpwl.sum().item<float>(),
                hpwl.index({"...", Slice(0, 2)}).sum().item<float>(),
                overflows[0].item<float>(),
                overflows[1].item<float>(),
                obj.item<float>(),
                ps.density_weight,
                ps.wa_coeff);
            
            auto node_pos = mov_node_pos.index({Slice(0, data.cell_mov_rhs)}).to(torch::kCPU);
            int dir = st::setting.slice_direction;
            auto node_pos_channel =
            node_pos.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs), dir});  // fetch z-cord
            auto slicer = data.die_info[dir * 2 + 1].to(torch::kCPU) * 0.5;          // FIXME:
            auto parter = (node_pos_channel > slicer);
            auto node_die = torch::_cast_Int(parter);

            auto [hpwl1, hpwl2, hpwl_ovlp] = evaluate_wl_cross_chip(node_pos.to(device), node_die.to(device), data);
            logger.info("bot: %f, top: %f, overlap: %f", 
            hpwl1.item<float>(), hpwl2.item<float>(), hpwl_ovlp.item<float>());
            
            if (need_stop || st::setting.draw_placement || ps.need_to_early_stop() || data_2d.node_wgt_grad.numel()
                || iteration == st::setting.inner_iter-1) {
                if (true) {
                    auto true_node_pos = mov_node_pos.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs), Slice(0, 2)}).to(torch::kCPU);
                    auto node_shift = (data.__die_shift__.index({Slice(0, 2)}) / data.__die_scale__.index({Slice(0, 2)})).expand_as(true_node_pos).to(torch::kCPU);
                    true_node_pos = true_node_pos + node_shift;
                    torch::Tensor node_size_bot = data_2d.node_size_bot * (1 - node_die).unsqueeze(1);
                    torch::Tensor node_size_top = data_2d.node_size_top * node_die.unsqueeze(1);

                    auto info1 = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_3D_0");
                    draw_fig_with_cairo_cpp(
                        true_node_pos.to(torch::kCPU),
                        node_size_bot,
                        data,
                        info1);
                    auto info2 = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_3D_1");
                    draw_fig_with_cairo_cpp(
                        true_node_pos.to(torch::kCPU),
                        node_size_top,
                        data,
                        info2);
                    auto info3 = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_3D_2");
                    auto node_pos_draw_cp =
                        torch::cat({true_node_pos,
                                    true_node_pos},
                                   0)
                            .to(torch::kCPU);
                    auto node_size_draw_cp = torch::cat({node_size_bot, node_size_top}, 0);
                    draw_fig_with_cairo_cpp_cross_chip(node_pos_draw_cp, node_size_draw_cp, data, info3);
                }

                // std::filesystem::path eval(std::string("eval"));
                // std::filesystem::path fig_root = logger.res_root / eval;
                // if (!std::filesystem::exists(fig_root)) {
                //     std::filesystem::create_directories(fig_root);
                // }
                // auto density_map = torch::rot90(mov_density_map.index({"...", 5}).to(torch::kCPU));
                // plot_pt(density_map,
                //         (char *)"imshow",
                //         (char *)fig_root.string().c_str(),
                //         (char *)("3d_density_" + std::to_string(iteration) + ".png").c_str());

                // auto density_map1 = torch::rot90((torch::abs(mov_density_map.index({"...", 5}) -
                // mov_density_map.index({"...", 4}))).to(torch::kCPU)); plot_pt(density_map1,
                //         (char *)"imshow",
                //         (char *)fig_root.string().c_str(),
                //         (char *)("3d_density_dif_" + std::to_string(iteration) + ".png").c_str());
            }
        }
        // if (roll_back_cnt && ps.need_to_early_stop()) {
        if (need_stop || (iteration >= st::setting.minGPStep && ps.need_to_early_stop())) {
            // torch::Tensor obj = optimizer.step();
            auto node_pos = mov_node_pos.index({Slice(0, data.cell_mov_rhs)}).to(torch::kCPU);
            int dir = st::setting.slice_direction;
            auto node_pos_channel =
            node_pos.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs), dir});  // fetch z-cord
            auto slicer = data.die_info[dir * 2 + 1].to(torch::kCPU) * 0.5;          // FIXME:
            auto parter = (node_pos_channel > slicer);
            // auto nos = torch::_cast_Int(parter)
            auto node_die = torch::_cast_Int(parter);
            if (true) {
                // auto nos = 1 - density_map_layers[0].node_die.index({Slice(data.cell_mov_lhs,
                // data.cell_mov_rhs)}).clone().to(torch::kCPU);

                // data.to(torch::kCPU);

                /* dump to partition result */

                // torch::Tensor node_size_bot = data_2d.node_size_bot * (nos).unsqueeze(1);
                // torch::Tensor node_size_top = data_2d.node_size_top * (1 - nos).unsqueeze(1);
                auto true_node_pos = mov_node_pos.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs), Slice(0, 2)}).to(torch::kCPU);
                auto node_shift = (data.__die_shift__.index({Slice(0, 2)}) / data.__die_scale__.index({Slice(0, 2)})).expand_as(true_node_pos).to(torch::kCPU);
                true_node_pos = true_node_pos + node_shift;
                torch::Tensor node_size_bot = data_2d.node_size_bot * (1 - node_die).unsqueeze(1);
                torch::Tensor node_size_top = data_2d.node_size_top * node_die.unsqueeze(1);

                auto info1 = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_3D_0");
                draw_fig_with_cairo_cpp(
                    true_node_pos.to(torch::kCPU),
                    node_size_bot,
                    data,
                    info1);
                auto info2 = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_3D_1");
                draw_fig_with_cairo_cpp(
                    true_node_pos.to(torch::kCPU),
                    node_size_top,
                    data,
                    info2);
                auto info3 = make_tuple(st::setting.round_recursion, iteration, data.design_name + "_3D_2");
                auto node_pos_draw_cp =
                    torch::cat({true_node_pos,
                                true_node_pos},
                                0)
                        .to(torch::kCPU);
                auto node_size_draw_cp = torch::cat({node_size_bot, node_size_top}, 0);
                draw_fig_with_cairo_cpp_cross_chip(node_pos_draw_cp, node_size_draw_cp, data, info3);
            }
            break;
        }
    }
    auto non_zero_indices = torch::nonzero(node_slide_grad);
    auto non_zero_num = torch::count_nonzero(node_slide_grad).item<int>();
    auto node_rotate = (node_slide_state * 2).to(torch::kInt32);
    for (int i = 0; i < non_zero_num; i++) {
        cout << node_rotate[non_zero_indices[i].item<int>()].item<int>() << endl;
    }

    // iteration = 1;
    /* retrieve best score and evaluate without filler */
    auto [best_sol, best_hpwl, best_overflow, best_iteration] = ps.get_best_solution();  //@@
    hpwl_state.hpwls[hpwl_state.hpwl_idx++] = best_hpwl;
    if (best_iteration > -1) {
        logger.info("GP Stop! #Iters %d masked_hpwl: %.4E overflow: %.4f", iteration, best_hpwl, best_overflow);
    } else {
        logger.info("GP Stop! cannot find best solution");
    }
    if (best_sol.numel() != 0) {
        mov_node_pos.data().copy_(best_sol);
    }

    if (st::setting.visualize_curve) ps.visualize();

    if (true) {
        torch::Tensor pin_pos = GP3D::wa_wirelength_hpwl::nodePosToPinPos(
            mov_node_pos.index({Slice(mov_lhs, data.cell_mov_rhs)}), data.pin_id2node_id, data.pin_rel_cpos);
        box_len = GP3D::wa_wirelength_hpwl::get_hpwl(data, pin_pos).to(torch::kCPU);
        box_len.index({"...", 0}) /= data.die_info[1].item<float>();
        box_len.index({"...", 1}) /= data.die_info[3].item<float>();
    }

    /* to CPU for postprocess */
    node_pos = mov_node_pos.index({Slice(mov_lhs, data.cell_mov_rhs)}).to(torch::kCPU);
    data.to(torch::kCPU);

    /* dump to partition result */
    int dir = st::setting.slice_direction;
    auto node_pos_channel = node_pos.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs), dir});  // fetch z-cord
    auto slicer = data.die_info[dir * 2 + 1] * 0.5;                                              // FIXME:

    auto [sorted_tensor, indices] = node_pos_channel.slice(0,0,data.cell_mov_rhs).sort();
    float area0_std = 0;
    float area1_std = 0;
    float area0_macro = 0;
    float area1_macro = 0;
    auto node_area_bot = torch::prod(data_2d.node_size_bot, 1);
    auto node_area_top = torch::prod(data_2d.node_size_top, 1);
    auto node_area_bot_at = node_area_bot.accessor<float, 1>();
    auto node_area_top_at = node_area_top.accessor<float, 1>();
    // for(int i=0;i<data.cell_mov_rhs;i++){
    //     if(data.macro_mask[i].item<int>()==1)
    //     {
    //         if(node_pos_channel[i].item<float>()>slicer.item<float>())
    //         {
    //             node_die[i]=1;
    //             area1_macro+=node_area_top_at[i];
    //         }else{
    //             area0_macro+=node_area_bot_at[i];
    //             node_die[i]=0;
    //         }
    //         continue;
    //     }
    //     else{
    //         node_die[i]=1;
    //         area1_std+=node_area_top_at[i];
    //     }
    // }
    // float bound0 = data_2d.max_mov_cell_areas[0].item<float>();
    // float bound1 = data_2d.max_mov_cell_areas[1].item<float>();
    // auto index_at = indices.accessor<long,1>();
    // int criteria=0;
    // for(int i=0;i<data.cell_mov_rhs;i++)
    // {
    //     int index = index_at[i];
    //     if(data.macro_mask[index].item<int>()==1)
    //     {
    //         continue;
    //     }
    //     else{
    //         float cell_area_top = node_area_top_at[index];
    //         float cell_area_bot = node_area_bot_at[index];
    //         if((area0_macro+area0_std+cell_area_bot)>bound0)
    //         {
    //             criteria=1;
    //         }
    //         if((criteria==0 && area0_std<area1_std)||(criteria==1 && (area0_macro+area0_std)<(area1_macro+area1_std)))
    //         {
    //             node_die[index] = 0;
    //             area0_std+=cell_area_bot;
    //             area1_std-=cell_area_top;
    //         }else{
    //             if(criteria==0 && (area1_macro+area1_std)>bound1)
    //             {
    //                 criteria=1;
    //             }
    //             else{
    //                 break;
    //             }
    //         }
    //     } 
    // }
    // logger.info("after slice, area0_macro=%f, area1_macro=%f",area0_macro,area1_macro);
    // logger.info("after slice, area0_std=%f, area1_std=%f",area0_std,area1_std);
    // logger.info("after slice, area0_total=%f, area1_total=%f",area0_std+area0_macro,area1_std+area1_macro);

    auto parter = (node_pos_channel > slicer);
    node_die = torch::_cast_Int(parter);

    // vector<float> macro_lx;
    // vector<float> macro_ly;
    // vector<float> macro_hx;
    // vector<float> macro_hy;
    // vector<int> macro_die;
    // for(auto macro_id:data_2d.macro_list){
    //     float size_x;
    //     float size_y;
    //     macro_die.push_back(node_die[macro_id].item<int>());
    //     if(node_die[macro_id].item<int>()==0)
    //     {
    //         size_x = data_2d.node_size_bot[macro_id][0].item<float>();
    //         size_y = data_2d.node_size_bot[macro_id][1].item<float>();
    //     }else{
    //         size_x = data_2d.node_size_top[macro_id][0].item<float>();
    //         size_y = data_2d.node_size_top[macro_id][1].item<float>();
    //     }
    //     macro_lx.push_back(node_pos[macro_id][0].item<float>()-size_x/2);
    //     macro_hx.push_back(node_pos[macro_id][0].item<float>()+size_x/2);
    //     macro_ly.push_back(node_pos[macro_id][1].item<float>()-size_y/2);
    //     macro_hy.push_back(node_pos[macro_id][1].item<float>()+size_y/2);
       
    // }
    // for(int j=0;j<macro_lx.size();j++)
    // {
    //     logger.info("id: %d, %f, %f, %f, %f", j, macro_lx[j], macro_hx[j], macro_ly[j], macro_hy[j]);
    // }
    // int change_cnt = 0;
    // for(int i=0;i<data.cell_mov_rhs;i++){
    //     if(data_2d.macro_mask[i].item<int>()==1)
    //     {
    //         continue;
    //     }
    //     float node_x = node_pos[i][0].item<float>();
    //     float node_y = node_pos[i][1].item<float>();
    //     float distance0=0;
    //     float distance1=0;
    //     for(int j=0;j<macro_lx.size();j++)
    //     {
    //         if(node_x>macro_lx[j]&&node_x<macro_hx[j]
    //            && node_y>macro_ly[j]&&node_y<macro_hy[j])
    //         {
    //             // logger.info("node %d overlap with macro %d", i, j);
    //             float distance_to_border_x = 
    //             min(abs(node_x-macro_lx[j]),abs(macro_hx[j]-node_x));
    //             float distance_to_border_y = 
    //             min(abs(node_y-macro_ly[j]),abs(macro_hy[j]-node_y));
    //             if(macro_die[j]==0)
    //             {
    //                 distance0 = min(distance_to_border_x, distance_to_border_y);
    //             }
    //             else{
    //                 distance1 = min(distance_to_border_x, distance_to_border_y);
    //             }
    //         }
    //     }
    //     if(distance1==0&&distance0==0) continue;
    //     int die_original = node_die[i].item<int>();
    //     // logger.info("id: %d, distance1: %f, distance0: %f", i, distance1, distance0);
    //     if(distance1>0&&distance0>0)
    //     {
    //         if(distance1<distance0)
    //         {
    //             node_die[i] = 1;
    //         }else{
    //             node_die[i] = 0;
    //         }
    //     }else if(distance1>0)
    //     {
    //         node_die[i] = 0;
    //     }else if(distance0>0)
    //     {
    //         node_die[i] = 1;
    //     }
    //     if(die_original!= node_die[i].item<int>())
    //     {
    //         change_cnt++;
    //     }
    // }
    // logger.info("%d changed thier die to get better border diatance", change_cnt);
    /*
    if (true) {
        auto node_z_plot = ((node_pos_channel / data.die_info[5]));
        auto [node_z_plot_sort, tmp] = torch::sort(node_z_plot, 0, false);

        std::filesystem::path eval(std::string("eval"));
        std::filesystem::path fig_root = logger.res_root / eval;
        if (!std::filesystem::exists(fig_root)) {
            std::filesystem::create_directories(fig_root);
        }
        plot_pt(node_z_plot_sort,
                (char *)"plot",
                (char *)fig_root.string().c_str(),
                (char *)("pin_pos.png"));
    }
    */

    /* legalize partition */
    mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    logger.info("num_nodes: %ld", num_nodes);
    int64_t max_area = 0;
    for (int i = 0; i < num_nodes; i++) {
        int group = node_die[i].item<int>();
        mov_cell_areas[group] += nodes[i]->sizes[group];
        if (max_area < nodes[i]->sizes[group]) {
            max_area = nodes[i]->sizes[group];
        }
    }
    logger.info("max area: %ld", max_area);
    logger.info("============ Original partition result ============");
    logger.info("Slice %.2f -> #Cells for each chip (%d, %d)",
                data_2d.tech_ratio.item<float>(),
                (1 - node_die).sum().item<int>(),
                node_die.sum().item<int>());
    logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(), (mov_cell_areas[1]).item<long>());
    logger.info("Utils for each chip (%.2f, %.2f)",
                (mov_cell_areas[0] / max_mov_cell_areas[0]).item<double>(),
                (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());

    // if (true)  { // TODO:
    //     mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    //     for (int i = 0; i < num_nodes; i++) {
    //         int group = node_die[i].item<int>();
    //         // TODO: force balance
    //         if(st::setting.clamp_util) {
    //             if ((mov_cell_areas[group] + nodes[i]->sizes[group] > max_mov_cell_areas[group]).item<bool>()) group
    //             = !group;
    //         }
    //         mov_cell_areas[group] += nodes[i]->sizes[group];
    //         nodes[i]->group = group;
    //         node_die[i] = group;
    //     }
    // }

    // data_2d.mov_cell_areas = mov_cell_areas.clone();
    // logger.info("============ Legalized partition result ============");
    // // rpt_cut_size();
    // logger.info("Slice %.2f -> #Cells for each chip (%d, %d)",
    //             data_2d.tech_ratio.item<float>(),
    //             (1 - node_die).sum().item<int>(),
    //             node_die.sum().item<int>());
    // logger.info("#Cells for each chip (%d, %d)", (1 - node_die).sum().item<int>(), node_die.sum().item<int>());
    // logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(),
    // (mov_cell_areas[1]).item<long>()); logger.info("Utils for each chip (%.2f, %.2f)",
    //             (mov_cell_areas[0] / max_mov_cell_areas[0]).item<double>(),
    //             (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());

    // if (st::setting.round_recursion == 0) {
    //     int count = 0;
    //     for (int i = 0; i < data.num_nodes; i++) {
    //         count += (node_die[i] != data_2d.node_die[i]).item<int>();
    //     }
    //     cout << count << " cells not in its region\n";
    // }
    data_2d.node_die = node_die.clone();

    if (true) {
        int count = 0;
        for (int i = 0; i < data_2d.num_nodes; i++)
            if (data_2d.aspect_ratio[i].item<float>() > 6)
                if (node_die[i].item<int>() == 1) count++;
        logger.info("[%d / %d] long cells at chip [0/1]",
                    torch::_cast_Float(data_2d.aspect_ratio > 6).sum().item<int>() - count,
                    count);
    }

    // if (ps.force_coeff) node_pos_2d_ground = run_gp2d_grid(data_2d);

    if (ps.force_coeff) {
        // torch::Tensor node_size_bot = data_2d.node_size_bot * (1 - node_die).unsqueeze(1);
        // torch::Tensor node_size_top = data_2d.node_size_top * node_die.unsqueeze(1);
        // auto filler_die = torch::ones({data.num_fillers_single_chip + data.num_fillers_cross_chip},
        //                               dtype(node_die.dtype()).device(torch::kCPU));
        auto filler_die = torch::ones({data.__num_fillers__ + data.num_fillers_cross_chip},
                                      dtype(node_die.dtype()).device(torch::kCPU));
        auto bot_filler = filler_die.clone();
        auto top_filler = filler_die.clone();
        bot_filler.index({Slice(mov_rhs + data.num_fillers_single_chip, None)}) *= 0;
        bot_filler = torch::cat({(1 - node_die), bot_filler});
        top_filler = torch::cat({node_die, top_filler});

        auto true_node_pos = mov_node_pos.index({"...", Slice(0, 2)}).to(torch::kCPU);
        auto node_shift = (data.__die_shift__.index({Slice(0, 2)}) / data.__die_scale__.index({Slice(0, 2)})).expand_as(true_node_pos).to(torch::kCPU);
        true_node_pos = true_node_pos + node_shift;

        torch::Tensor node_size_bot =
            mov_node_size.index({"...", Slice(0, 2)}).to(torch::kCPU) * bot_filler.unsqueeze(1);
        torch::Tensor node_size_top =
            mov_node_size.index({"...", Slice(0, 2)}).to(torch::kCPU) * top_filler.unsqueeze(1);

        auto info1 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_3D_PT_0");
        draw_fig_with_cairo_cpp(true_node_pos, node_size_bot, data, info1);
        auto info2 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_3D_PT_1");
        draw_fig_with_cairo_cpp(true_node_pos, node_size_top, data, info2);
        auto info3 = make_tuple(st::setting.round_recursion, 0, data.design_name + "_3D_PT_2");

        true_node_pos = node_pos.index({"...", Slice(0, 2)}).to(torch::kCPU);
        node_shift = (data.__die_shift__.index({Slice(0, 2)}) / data.__die_scale__.index({Slice(0, 2)})).expand_as(true_node_pos).to(torch::kCPU);
        true_node_pos = true_node_pos + node_shift;

        node_size_bot = data_2d.node_size_bot * (1 - node_die).unsqueeze(1);
        node_size_top = data_2d.node_size_top * node_die.unsqueeze(1);
        auto node_pos_draw_cp =
            torch::cat({true_node_pos, true_node_pos}, 0);
        auto node_size_draw_cp = torch::cat({node_size_bot, node_size_top}, 0);
        draw_fig_with_cairo_cpp_cross_chip(node_pos_draw_cp, node_size_draw_cp, data, info3);
    }

    if (ps.force_coeff) return {node_rotate.to(torch::kCPU), mov_node_pos.to(torch::kCPU)};

    // if (ps.force_coeff) node_pos_2d_ground = run_gp2d_grid(data_2d);
    node_pos_2d_ground = node_pos.index({Slice(data_2d.cell_mov_lhs, data_2d.cell_mov_rhs), Slice(0, 2)});
    /* hpwl-driven fm */
    if (st::setting.round_recursion) {
        logger.info("============== HPWL-FM round %d ==============", st::setting.round_recursion);
        data_2d.node_pos = node_pos.index({Slice(data_2d.cell_mov_lhs, data_2d.cell_mov_rhs), "..."});
        data_2d.node_die = data_2d.node_die.index({Slice(data_2d.cell_mov_lhs, data_2d.cell_mov_rhs)});
        data_2d.reset_net_node();
        run_fm_wl(data_2d, st::setting.skip_hpwl_fm);
    }
    st::setting.round_recursion--;
    st::setting.use_filler = true;

    torch::Tensor bonding_map;
    if (true) {
        /* reconstruct nets */
        torch::Tensor net_cut_info = torch::zeros({num_nets, 2}, torch::dtype(torch::kInt));
        const torch::TensorAccessor<int64_t, 1> pin_id2node_id_at = data.pin_id2node_id.accessor<int64_t, 1>();
        const torch::TensorAccessor<int64_t, 1> hyperedge_list_at = data.hyperedge_list.accessor<int64_t, 1>();
        const torch::TensorAccessor<int64_t, 1> hyperedge_list_end_at = data.hyperedge_list_end.accessor<int64_t, 1>();
        const torch::TensorAccessor<int, 1> cell_die_at = node_die.accessor<int, 1>();

        for (int i = 0; i != num_nets; i++) {
            int64_t start_idx = 0;
            if (i != 0) {
                start_idx = hyperedge_list_end_at[i - 1];
            }
            int64_t end_idx = hyperedge_list_end_at[i];
            if (end_idx != start_idx) {
                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    int64_t pin_id = hyperedge_list_at[idx];
                    int64_t node_id = pin_id2node_id_at[pin_id];
                    if (node_id > data.num_nodes) continue;

                    int c_id = cell_die_at[node_id];

                    if (c_id != 0 && c_id != 1) continue;

                    net_cut_info[i][c_id] += 1;
                }
            }
        }
        bonding_map = torch::_cast_Int(torch::prod(net_cut_info, 1) != 0);
        cutsize = (bonding_map).sum().item<int>();
        logger.info("============ Cutsize: %d ============", cutsize);

        torch::Tensor w = torch::ones(data.net_to_num_pins.sizes()[0]);
        torch::Tensor pin_num_count_cut = torch::_cast_Int(torch::zeros(data.net_to_num_pins.max().item<int>() + 1));
        pin_num_count_cut.scatter_add_(0, torch::_cast_Long(data.net_to_num_pins), torch::_cast_Int(bonding_map));

        auto net_size_cut_x = torch::_cast_Float(torch::zeros(data.net_to_num_pins.max().item<int>() + 1));
        auto net_size_cut_y = torch::_cast_Float(torch::zeros(data.net_to_num_pins.max().item<int>() + 1));
        net_size_cut_x.scatter_add_(
            0, torch::_cast_Long(data_2d.net_to_num_pins), box_len.index({"...", 0}) * bonding_map);
        net_size_cut_y.scatter_add_(
            0, torch::_cast_Long(data_2d.net_to_num_pins), box_len.index({"...", 1}) * bonding_map);

        auto net_size_x = torch::_cast_Float(torch::zeros(data.net_to_num_pins.max().item<int>() + 1));
        auto net_size_y = torch::_cast_Float(torch::zeros(data.net_to_num_pins.max().item<int>() + 1));
        net_size_x.scatter_add_(0, torch::_cast_Long(data_2d.net_to_num_pins), box_len.index({"...", 0}));
        net_size_y.scatter_add_(0, torch::_cast_Long(data_2d.net_to_num_pins), box_len.index({"...", 1}));

        int view = 20;
        if (net_size_x.size(0) < view) {
            view = net_size_x.size(0);
        }
        logger.info("======== Net bbox info ========");
        for (int i = 0; i < view - 1; ++i) {
            logger.info("%-5d / %-5d (%5.2f) %d-pin nets are cut, \taverage bbox (%5.3f / %5.3f)",
                        pin_num_count_cut[i].item<int>(),
                        data_2d.pin_num_count[i].item<int>(),
                        (pin_num_count_cut[i] / data_2d.pin_num_count[i]).item<float>(),
                        i,
                        ((net_size_cut_x[i] + net_size_cut_y[i]) / pin_num_count_cut[i]).item<float>(),
                        ((net_size_x[i] + net_size_y[i]) / data_2d.pin_num_count[i]).item<float>());
        }
        logger.info("%-5d / %-5d (%5.2f) %d-pin nets are cut, \taverage bbox (%5.3f / %5.3f)",
                    pin_num_count_cut.index({Slice(view, None)}).sum().item<int>(),
                    data_2d.pin_num_count.index({Slice(view, None)}).sum().item<int>(),
                    (pin_num_count_cut.index({Slice(view, None)}).sum() /
                     data_2d.pin_num_count.index({Slice(view, None)}).sum())
                        .item<float>(),
                    view,
                    ((net_size_cut_x + net_size_cut_y).index({Slice(view, None)}).sum() /
                     pin_num_count_cut.index({Slice(view, None)}).sum())
                        .item<float>(),
                    ((net_size_x + net_size_y).index({Slice(view, None)}).sum() /
                     data_2d.pin_num_count.index({Slice(view, None)}).sum())
                        .item<float>());
    }

    if (!st::setting.mononlithic) {
        auto via_pos = GP3D::wa_wirelength_hpwl::place_via_to_optim(node_pos.to(device),
                                                                    node_die.to(device),
                                                                    data.pin_id2node_id.to(device),
                                                                    data.pin_rel_cpos.to(device),
                                                                    data.hyperedge_list.to(device),
                                                                    data.hyperedge_list_end.to(device),
                                                                    (bonding_map == 1).to(device));

        auto node_die_tmp = torch::cat({node_die, torch::_cast_Int(bonding_map * 3 - 1)}, 0);
        std::tie(mov_node_pos, mov_node_size, expand_ratio) = data.get_mov_node_info_with_via();
        std::tie(mov_lhs, mov_rhs) = data.movable_index;

        logger.info("load from prev sol ...");
        mov_node_pos.index({Slice(0, data.cell_mov_rhs), Slice(0, 2)})
            .copy_(node_pos.index({Slice(0, data.cell_mov_rhs), Slice(0, 2)}));
        mov_node_pos.index({Slice(data.cell_mov_rhs, mov_rhs), Slice(0, 2)}).copy_(via_pos.index({"...", Slice(0, 2)}));

        data.hyperedge_list = data.hyperedge_list.to(device);
        data.hyperedge_list_end = data.hyperedge_list_end.to(device);
        auto [hpwl1, hpwl2] = GP3D::wa_wirelength_hpwl::get_hpwl_cross_chip(
            data, mov_node_pos.index({Slice(mov_lhs, mov_rhs)}).to(device), node_die_tmp.to(device));
        logger.info("3D Adding Vias, solution eval, exact HPWL [bot + top = total]: [%.3E, %.3E, %.3E]",
                    hpwl1.item<float>(),
                    hpwl2.item<float>(),
                    (hpwl1 + hpwl2).item<float>());

        node_pos = mov_node_pos.index({Slice(mov_lhs, mov_rhs)}).to(torch::kCPU);
    } else {
        node_pos = mov_node_pos;
    }

    if (st::setting.save_model) {
        std::filesystem::path res_root = logger.res_root;
        std::string model_dir = res_root.string() + "/gp3d.pt";
        logger.info("Save model to %s", model_dir.c_str());
        torch::save(node_pos, model_dir);
        std::string pt_dir = res_root.string() + "/pt3d.pt";
        torch::save(node_die, pt_dir);
    }

    return {node_rotate, node_pos};
}