#include "electronic_density_layer.h"

ElectronicDensityLayer::ElectronicDensityLayer(at::Tensor _unit_len,
                                               int _num_bin_x,
                                               int _num_bin_y,
                                               torch::Device device,
                                               tuple<double, double, function<at::Tensor(at::Tensor)>> _overflow_helper,
                                               at::Tensor _expand_ratio,
                                               tuple<at::Tensor, at::Tensor, at::Tensor> _sorted_maps,
                                               torch::Tensor _macro_mask,
                                               bool _inference_mode,
                                               bool _scale_w_k) {
    num_bin_x = _num_bin_x;
    num_bin_y = _num_bin_y;
    inference_mode = _inference_mode;
    overflow_helper = _overflow_helper;
    expand_ratio = _expand_ratio;
    sorted_maps = _sorted_maps;
    scale_w_k = _scale_w_k;
    macro_mask = _macro_mask;

    // FIXME: is None -> !.numel()
    if (!_unit_len.numel())
        unit_len = at::tensor({1.0 / num_bin_x, 1.0 / num_bin_y}, torch::dtype(torch::kFloat).device(device));
    else
        unit_len = _unit_len;
    min_node_w = unit_len[0] * sqrt(2);
    min_node_h = unit_len[1] * sqrt(2);

    // Pre-compute some constants in FFT computation
    at::Tensor w_j = torch::arange(num_bin_x, torch::dtype(torch::kFloat).device(device))
                         .mul(2 * at::tensor(M_PI, torch::dtype(torch::kFloat).device(device)) / num_bin_x)
                         .reshape({num_bin_x, 1});
    at::Tensor w_k = torch::arange(num_bin_y, torch::dtype(torch::kFloat).device(device))
                         .mul(2 * at::tensor(M_PI, torch::dtype(torch::kFloat).device(device)) / num_bin_y)
                         .reshape({1, num_bin_y});
    // scale_w_k because the aspect ratio of a bin may not be 1
    // NOTE: we will not scale down w_k in NN since it may distrub the training
    if (scale_w_k) w_k.mul_(unit_len[0] / unit_len[1]);
    at::Tensor wj2_plus_wk2 = w_j.pow(2) + w_k.pow(2);
    wj2_plus_wk2[0][0] = 1.0;

    at::Tensor potential_scale = 1.0 / wj2_plus_wk2;
    potential_scale[0][0] = 0.0;

    at::Tensor force_x_scale = w_j * potential_scale * 0.5;
    at::Tensor force_y_scale = w_k * potential_scale * 0.5;

    at::Tensor force_x_coeff = (torch::tensor(-1, torch::dtype(torch::kFloat).device(device))
                                    .pow(torch::arange(num_bin_x, torch::dtype(torch::kFloat).device(device))))
                                   .unsqueeze(1);
    at::Tensor force_y_coeff = (torch::tensor(-1, torch::dtype(torch::kFloat).device(device))
                                    .pow(torch::arange(num_bin_y, torch::dtype(torch::kFloat).device(device))))
                                   .unsqueeze(0);

    at::Tensor potential_coeff = at::tensor(1.0, torch::dtype(torch::kFloat).device(device));
    fft_scale =
        make_tuple(potential_scale, potential_coeff, force_x_scale, force_y_scale, force_x_coeff, force_y_coeff);
    // TODO: cache
}  // END MODULE

//---------------------------------------------------------------------

at::Tensor ElectronicDensityLayer::get_cache_var(at::Tensor node_pos, at::Tensor node_size, at::Tensor node_weight) {
    if (has_setup_inference_var && node_weight.numel()) {
        cache_node_weight = node_weight;
        return node_weight;
    }
    if (inference_mode) {
        if (!has_setup_inference_var) {
            torch::NoGradGuard no_grad;
            preprocess_inference_var(node_pos, node_size, node_weight);
        }
        node_weight = cache_node_weight;
    }
    // FIXME: is None -> !.numel()
    if (!node_weight.numel()) {
        // node_weight = node_pos.new_ones(node_pos.sizes()[0]).detach();
        node_weight = torch::ones(node_pos.size(0), torch::dtype(node_size.dtype()).device(node_pos.device())).detach();
    }
    return node_weight;
}  // END MODULE

//---------------------------------------------------------------------

void ElectronicDensityLayer::preprocess_inference_var(at::Tensor node_pos,
                                                      at::Tensor node_size,
                                                      at::Tensor node_weight) {
    // precompute all common variables for faster inferencing
    assert(inference_mode);
    assert(!has_setup_inference_var);
    // FIXME: is None -> !.numel()
    if (!node_weight.numel()) {
        // generate all ones node_weight if not given node_weight
        node_weight =
            torch::ones(node_size.size(0), torch::dtype(node_size.dtype()).device(node_pos.device())).detach();
    }
    cache_node_weight = node_weight;
    has_setup_inference_var = true;
}  // END MODULE

//---------------------------------------------------------------------

tuple<torch::Tensor, torch::Tensor> ElectronicDensityLayer::direct_calc_overflow(torch::Tensor node_pos,
                                                           torch::Tensor node_size,
                                                           torch::Tensor init_density_map) {
    //    // FIXME: !=None -> !.numel()
    //    torch::Tensor node_weight) {
    auto [mov_lhs, mov_rhs, overflow_fn] = overflow_helper;
    // TODO: density cache
    torch::Tensor node_weight;
    auto [tmp1, mov_conn_sorted_map, tmp2] = sorted_maps;
    int num_mov_nodes = mov_rhs - mov_lhs;
    node_weight = get_cache_var(node_pos, node_size, node_weight);
    
    // cout << "-------- den -----\n";
    // cout << has_setup_inference_var << endl;
    // cout << inference_mode << endl;
    // cout << mov_lhs << " " << mov_rhs << endl;
    // cout << node_weight.index({Slice({mov_lhs, mov_rhs})}).sum().item<int>() << endl;

    torch::Tensor normalize_node_info = node_size.new_empty({num_mov_nodes, 5});  // x_l, x_h, y_l, y_h, weight
    normalize_node_info = density_map_normalize_node(node_pos.index({Slice({mov_lhs, mov_rhs})}),
                                                     node_size.index({Slice({mov_lhs, mov_rhs})}),
                                                     node_weight.index({Slice({mov_lhs, mov_rhs})}),
                                                     expand_ratio.index({Slice({mov_lhs, mov_rhs})}),
                                                     unit_len,
                                                     normalize_node_info,
                                                     num_bin_x,
                                                     num_bin_y,
                                                     num_mov_nodes);
    torch::Tensor aux_mat = init_density_map.clone();
    torch::Tensor mov_density_map =
        density_map_forward(normalize_node_info, mov_conn_sorted_map, aux_mat, num_bin_x, num_bin_y, num_mov_nodes);

    torch::Tensor overflow = overflow_fn(mov_density_map);
    return {overflow, mov_density_map};
}  // END MODULE

//---------------------------------------------------------------------

torch::Tensor ElectronicDensityLayer::density_grad(at::Tensor node_pos,
                                                      at::Tensor node_size,
                                                      at::Tensor node_weight,
                                                      at::Tensor density_map) {
    auto [mov_lhs, mov_rhs, overflow_fn] = overflow_helper;
    // TODO: density cache
    // torch::Tensor node_weight;
    // node_weight = get_cache_var(node_pos, node_size, node_weight);
    auto [tmp1, mov_conn_sorted_map, tmp2] = sorted_maps;
    int num_mov_nodes = mov_rhs - mov_lhs;

    torch::Tensor normalize_node_info = node_size.new_empty({num_mov_nodes, 5});  // x_l, x_h, y_l, y_h, weight
    normalize_node_info = density_map_normalize_node(node_pos.index({Slice({mov_lhs, mov_rhs})}),
                                                     node_size.index({Slice({mov_lhs, mov_rhs})}),
                                                     node_weight.index({Slice({mov_lhs, mov_rhs})}),
                                                     expand_ratio.index({Slice({mov_lhs, mov_rhs})}),
                                                     unit_len,
                                                     normalize_node_info,
                                                     num_bin_x,
                                                     num_bin_y,
                                                     num_mov_nodes);

    torch::Tensor net_grad = density_backward(
        normalize_node_info, mov_conn_sorted_map, density_map, num_bin_x, num_bin_y, num_mov_nodes);

    return net_grad;
}