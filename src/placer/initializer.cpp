#include "initializer.h"
#include <cuda_runtime.h>
#include <cuda.h>

torch::Tensor get_init_density_map(PlaceData& data) {
    auto [lhs, rhs] = data.fixed_index;
    torch::Device device = data.device;
    torch::Tensor zeros_density_map =
        torch::zeros({data.num_bin_x, data.num_bin_y}, torch::dtype(data.node_size.dtype())).to(device);
    if (lhs == rhs) {
        data.init_density_map = zeros_density_map.to(torch::kCPU);
        return zeros_density_map;
    }

    torch::Tensor node_pos = data.node_pos.index({Slice(lhs, rhs)});
    torch::Tensor node_size = data.node_size.index({Slice(lhs, rhs)});
    torch::Tensor node_weight = torch::ones(node_size.size(0), torch::dtype(torch::kFloat).device(device));

    // TODO: device
    auto init_density_map = density_map_forward_naive(
        node_pos.to(device), node_size.to(device), node_weight.to(device), data.unit_len.to(device),
        zeros_density_map.to(device), data.num_bin_x, data.num_bin_y, node_pos.sizes()[0], -1.0, -1.0, 1e-4, false);

    init_density_map = init_density_map.contiguous();
    if ((init_density_map > 1).sum().item().toInt() > 0) {
        logger.warning("Some bins in init_density_map are overflow. Clamp them.");
    }
    if ((init_density_map < 0).sum().item().toInt() > 0) {
        logger.error("init_density_map has negative value. Please check.");
    }


    init_density_map = init_density_map.clamp(0.0, 1.0).mul(st::setting.target_density);
    data.init_density_map = init_density_map.to(torch::kCPU);

    return init_density_map;
}

torch::Tensor get_init_density_map_cross_chip(NodeData& data) {
    torch::Device device = data.device;
    torch::Tensor init_density_maps =
        torch::zeros({2, data.num_bin_x, data.num_bin_y}, torch::dtype(data.node_size.dtype())).to(device);
    if(st::setting.block_row) {
        for (int i = 0; i < 2; i++) {
            torch::Tensor zeros_density_map =
                torch::zeros({data.num_bin_x, data.num_bin_y}, torch::dtype(data.node_size.dtype())).to(device);

            torch::Tensor node_pos = torch::zeros({data.numRows[i].item<int>(), 2}, torch::dtype(torch::kFloat).device(device));
            torch::Tensor node_size = torch::zeros({data.numRows[i].item<int>(), 2}, torch::dtype(torch::kFloat).device(device));
            torch::Tensor node_weight = torch::ones(data.numRows[i].item<int>(), torch::dtype(torch::kFloat).device(device));

            node_size.index_put_({"...", 0}, data.die_info[1]);
            node_size.index_put_({"...", 1}, data.rowHeights[i] * 0.1);

            node_pos.index_put_({"...", 0}, (data.die_info[0] + data.die_info[1]) * 0.5);
            for (int r = 0; r < data.numRows[i].item<int>(); r++) {
                node_pos[r][1] =  data.rowHeights[i] * (r + 1);
            }

            auto init_density_map = density_map_forward_naive(
                node_pos.to(device), node_size.to(device), node_weight.to(device), data.unit_len.to(device),
                zeros_density_map.to(device), data.num_bin_x, data.num_bin_y, node_pos.sizes()[0], -1.0, -1.0, 1e-4, false);

            init_density_maps[i] = init_density_map.clone();

            std::filesystem::path eval(std::string("eval"));
            std::filesystem::path fig_root = logger.res_root / eval;
            if (!std::filesystem::exists(fig_root)) {
                std::filesystem::create_directories(fig_root);
            }
            auto density_map = torch::rot90(init_density_map.to(torch::kCPU));
            // plot_pt(density_map,
            //         (char *)"imshow",
            //         (char *)fig_root.string().c_str(),
            //         (char *)("density_" + std::to_string(i) + ".png").c_str());

        }
        data.node_size_bot.index({"...", 1}) *= 0.9;
        data.node_size_top.index({"...", 1}) *= 0.9;
    }

    data.init_density_maps = init_density_maps.to(torch::kCPU);
    return init_density_maps;
}

void init_params(torch::Tensor mov_node_pos, std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn, int mov_lhs,
                 int mov_rhs, torch::Tensor conn_fix_node_pos, ElectronicDensityLayer& density_map_layer,
                 torch::Tensor mov_node_size, torch::Tensor init_density_map, torch::optim::Optimizer& optimizer,
                 ParamScheduler& ps, PlaceData& data) {
    mov_node_pos = trunc_node_pos_fn(mov_node_pos);
    torch::Tensor conn_node_pos = mov_node_pos.index({Slice(mov_lhs, mov_rhs), "..."});
    conn_node_pos = torch::cat({conn_node_pos, conn_fix_node_pos}, 0);

    // TODO: param scheduler
    auto wl_val_list = WAWirelengthLossAndHPWL::apply(conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos,
                                                      data.node2pin_list, data.node2pin_list_end,
                                                      data.hyperedge_list, data.hyperedge_list_end, data.net_mask,
                                                      // FIXME:
                                                      ps.wa_coeff, data.hpwl_scale, data.macro_list);
    auto wl_loss = wl_val_list[0];
    auto hpwl = wl_val_list[1];

    torch::Tensor node_weight;
    auto den_val_list = density_map_layer.forward(mov_node_pos, mov_node_size, init_density_map, data.mov_node_weight);
    auto density_loss = den_val_list[0];
    auto overflow = den_val_list[1];

    auto [wl_grad, density_grad] = calc_grad(optimizer, mov_node_pos, wl_loss, density_loss);
    double init_density_weight = (wl_grad.norm(1) / density_grad.norm(1)).detach().item().toDouble();

    logger.info("Init density weight %.3E", init_density_weight);

    ps.set_init_param(init_density_weight);
}

double estimate_initial_learning_rate(
    const std::function<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor)>& obj_and_grad_fn,
    std::function<torch::Tensor(torch::Tensor)> constraint_fn, torch::Tensor _x_k, double lr) {
    // create a new x_k
    torch::Tensor x_k = constraint_fn(_x_k).clone().detach().requires_grad_(true);
    auto [obj_k, g_k] = obj_and_grad_fn(x_k);
    torch::Tensor x_k_1 = (constraint_fn(x_k - lr * g_k)).clone().detach().requires_grad_(true);
    auto [obj_k_1, g_k_1] = obj_and_grad_fn(x_k_1);
    // FIXME:
    // return ((x_k - x_k_1).norm(2) / (g_k - g_k_1).norm(2)).item().toDouble();
    return ((lr * g_k).norm(2) / (g_k - g_k_1).norm(2)).item().toDouble();
}

void init_params_with_via(torch::Tensor mov_node_pos, std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn,
                          int mov_lhs, int mov_rhs, torch::Tensor conn_fix_node_pos,
                          vector<ElectronicDensityLayer>& density_map_layers, torch::Tensor mov_node_size,
                          torch::Tensor init_density_maps, torch::optim::Optimizer& optimizer, ParamScheduler& ps,
                          NodeData& data, int& c_id) {
    mov_node_pos = trunc_node_pos_fn(mov_node_pos);
    torch::Tensor conn_node_pos = mov_node_pos.index({Slice(mov_lhs, mov_rhs), "..."});
    conn_node_pos = torch::cat({conn_node_pos, conn_fix_node_pos}, 0);

    vector<torch::Tensor> wl_losses(2);
    vector<torch::Tensor> den_losses(3);
    torch::Tensor overflow = torch::tensor(0, torch::dtype(mov_node_pos.dtype()).device(mov_node_pos.device()));
    torch::Tensor hpwl = torch::tensor(0, torch::dtype(mov_node_pos.dtype()).device(mov_node_pos.device()));

    auto wl_val_list = WAWirelengthLossAndHPWL::apply(conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos,
                                                      data.node2pin_list, data.node2pin_list_end,
                                                      data.hyperedge_list_cc[c_id], data.hyperedge_list_end_cc[c_id],
                                                      data.net_mask, ps.wa_coeff, data.hpwl_scale, data.macro_list);

    torch::Tensor node_weight = data.mov_node_weights[c_id];
    auto den_val_list =
        density_map_layers[c_id].forward(mov_node_pos, mov_node_size, init_density_maps[c_id], node_weight, false);

    auto [wl_grad, density_grad] = calc_grad(optimizer, mov_node_pos, wl_val_list[0], den_val_list[0]);
    double init_density_weight = (wl_grad.norm(1) / density_grad.norm(1)).detach().item<double>();

    printlog(LOG_INFO, "Init density weight %.E", init_density_weight);

    ps.set_init_param(init_density_weight);
}

void init_params_multi_circuit(torch::Tensor mov_node_pos,
                               std::function<torch::Tensor(torch::Tensor)> trunc_node_pos_fn, int mov_lhs, int mov_rhs,
                               torch::Tensor conn_fix_node_pos, vector<ElectronicDensityLayer>& density_map_layers,
                               torch::Tensor mov_node_size, torch::Tensor init_density_maps,
                               torch::optim::Optimizer& optimizer, ParamScheduler& ps, NodeData& data) {
    // Assert input parameters are valid
    assert(mov_node_pos.defined() && "mov_node_pos tensor is undefined");
    assert(conn_fix_node_pos.defined() && "conn_fix_node_pos tensor is undefined");
    assert(mov_node_size.defined() && "mov_node_size tensor is undefined");
    assert(init_density_maps.defined() && "init_density_maps tensor is undefined");
    assert(mov_lhs >= 0 && mov_rhs >= mov_lhs && "Invalid mov_lhs/mov_rhs range");

    mov_node_pos = trunc_node_pos_fn(mov_node_pos);

    // Check bounds before slicing
    assert(mov_rhs <= mov_node_pos.size(0) && "mov_rhs exceeds tensor dimension");

    torch::Tensor conn_node_pos = mov_node_pos.index({Slice(mov_lhs, mov_rhs), "..."});
    conn_node_pos = torch::cat({conn_node_pos, conn_fix_node_pos}, 0);

    vector<torch::Tensor> wl_losses(2);
    vector<torch::Tensor> den_losses(3);
    torch::Tensor wl_loss;
    torch::Tensor den_loss;
    torch::Tensor overflow = torch::tensor(0, torch::dtype(mov_node_pos.dtype()).device(mov_node_pos.device()));
    torch::Tensor hpwl = torch::tensor(0, torch::dtype(mov_node_pos.dtype()).device(mov_node_pos.device()));

    

    if (st::setting.net_type == "monon") {
        auto wl_val_list = WAWirelengthLossAndHPWL::apply(conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos,
                                                        data.node2pin_list, data.node2pin_list_end,
                                                        data.hyperedge_list, data.hyperedge_list_end, data.net_mask,
                                                        ps.wa_coeff, data.hpwl_scale, data.macro_list, data.node_die);
        wl_loss = wl_val_list[0];
    } else {
        auto wl_val_list =
            WAWirelengthLossAndHPWL::apply(conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos, 
                                           data.node2pin_list, data.node2pin_list_end,
                                           data.hyperedge_list,
                                           data.hyperedge_list_end, data.net_mask, ps.wa_coeff, data.hpwl_scale, data.macro_list);
        wl_loss = wl_val_list[0];
    }



    /* 3 density layers: cell | cell | via */
    if (st::setting.skip_2_5d) {
        torch::Tensor node_weight = data.mov_node_weights[2];
        auto den_val_list =
            density_map_layers[2].forward(mov_node_pos, mov_node_size, init_density_maps[2], node_weight);
        den_loss = den_val_list[0];
        overflow += den_val_list[1];
    } else {
        for (int i = 0; i < st::setting.num_den_layer; i++) {  // TODO:
            torch::Tensor node_weight = data.mov_node_weights[i];
            auto den_val_list =
                density_map_layers[i].forward(mov_node_pos, mov_node_size, init_density_maps[i], node_weight);
            den_losses[i] = den_val_list[0];
            overflow += den_val_list[1] / 3;
        }
        den_loss = st::setting.num_den_layer == 3 ? (den_losses[0] + den_losses[1] + den_losses[2])
                                                  : (den_losses[0] + den_losses[1]);
    }

    std::cout << "st::setting.num_den_layer" << st::setting.num_den_layer <<std::endl;

    std::cout << "den_loss" << den_loss <<std::endl;

    std::cout << "wl_loss" << wl_loss <<std::endl;

    // Check for NaN/Inf in losses before gradient calculation
    if (torch::isnan(den_loss).any().item<bool>() || torch::isinf(den_loss).any().item<bool>()) {
        std::cout << "Error: den_loss contains NaN or Inf values!" << std::endl;
        assert(false && "den_loss contains invalid values");
    }
    if (torch::isnan(wl_loss).any().item<bool>() || torch::isinf(wl_loss).any().item<bool>()) {
        std::cout << "Error: wl_loss contains NaN or Inf values!" << std::endl;
        assert(false && "wl_loss contains invalid values");
    }

    // Synchronize CUDA and check for errors before gradient calculation
    if (mov_node_pos.is_cuda()) {
        cudaDeviceSynchronize();
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            std::cout << "CUDA error detected before calc_grad: " << cudaGetErrorString(cuda_err) << std::endl;
            std::cout << "This error may be from a previous CUDA operation" << std::endl;
            assert(false && "CUDA error before gradient calculation");
        }
    }

    auto [wl_grad, density_grad] = calc_grad(optimizer, mov_node_pos, wl_loss, den_loss);

    // Check for CUDA errors after gradient calculation
    if (mov_node_pos.is_cuda()) {
        cudaDeviceSynchronize();
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            std::cout << "CUDA error detected after calc_grad: " << cudaGetErrorString(cuda_err) << std::endl;
            assert(false && "CUDA error after gradient calculation");
        }
    }

    // Assert gradients are valid
    assert(wl_grad.defined() && "wl_grad is undefined");
    assert(density_grad.defined() && "density_grad is undefined");
    assert(!torch::isnan(wl_grad).any().item<bool>() && "wl_grad contains NaN values");
    assert(!torch::isinf(wl_grad).any().item<bool>() && "wl_grad contains Inf values");
    assert(!torch::isnan(density_grad).any().item<bool>() && "density_grad contains NaN values");
    assert(!torch::isinf(density_grad).any().item<bool>() && "density_grad contains Inf values");

    // Check gradient norms before division
    double density_grad_norm = density_grad.norm(1).item<double>();
    assert(density_grad_norm > 0.0 && "density_grad norm is zero - cannot divide by zero");

    double init_density_weight = (wl_grad.norm(1) / density_grad_norm).detach().item<double>();

    // Check for valid init_density_weight
    assert(!std::isnan(init_density_weight) && "init_density_weight is NaN");
    assert(!std::isinf(init_density_weight) && "init_density_weight is Inf");
    assert(init_density_weight >= 0.0 && "init_density_weight is negative");

    printlog(LOG_INFO, "Init density weight %.3E", init_density_weight);

    ps.set_init_param(init_density_weight);
}
