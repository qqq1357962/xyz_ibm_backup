#include "electronic_density_layer.h"

namespace GP3D {

ElectronicDensityLayer::ElectronicDensityLayer(at::Tensor _unit_len,
                                               int _num_bin_x,
                                               int _num_bin_y,
                                               int _num_bin_z,
                                               torch::Device device,
                                               tuple<int, int, function<at::Tensor(at::Tensor)>> _overflow_helper,
                                               at::Tensor _expand_ratio,
                                               tuple<at::Tensor, at::Tensor, at::Tensor> _sorted_maps,
                                               torch::Tensor _macro_mask,
                                               bool _inference_mode,
                                               bool _scale_w_k) {
    num_bin_x = _num_bin_x;
    num_bin_y = _num_bin_y;
    num_bin_z = _num_bin_z;
    inference_mode = _inference_mode;
    overflow_helper = _overflow_helper;
    expand_ratio = _expand_ratio;
    sorted_maps = _sorted_maps;
    scale_w_k = _scale_w_k;
    macro_mask = _macro_mask;

    // FIXME: is None -> !.numel()
    if (!_unit_len.numel())
        unit_len =
            at::tensor({1.0 / num_bin_x, 1.0 / num_bin_y, 1.0 / num_bin_z}, torch::dtype(torch::kFloat).device(device));
    else
        unit_len = _unit_len;
    min_node_w = unit_len[0] * sqrt(2);
    min_node_h = unit_len[1] * sqrt(2);
    min_node_d = unit_len[2] * sqrt(2);

    // Pre-compute some constants in FFT computation
    at::Tensor w_j = torch::arange(num_bin_x, torch::dtype(torch::kFloat).device(device))
                         .mul(2 * at::tensor(M_PI, torch::dtype(torch::kFloat).device(device)) / num_bin_x)
                         .reshape({num_bin_x, 1, 1});
    at::Tensor w_k = torch::arange(num_bin_y, torch::dtype(torch::kFloat).device(device))
                         .mul(2 * at::tensor(M_PI, torch::dtype(torch::kFloat).device(device)) / num_bin_y)
                         .reshape({1, num_bin_y, 1});
    at::Tensor w_l = torch::arange(num_bin_z, torch::dtype(torch::kFloat).device(device))
                         .mul(2 * at::tensor(M_PI, torch::dtype(torch::kFloat).device(device)) / num_bin_z)
                         .reshape({1, 1, num_bin_z});  // FIXME: shape
    // scale_w_k because the aspect ratio of a bin may not be 1
    // NOTE: we will not scale down w_k in NN since it may distrub the training
    if (scale_w_k) w_k.mul_(unit_len[0] / unit_len[1]);
    if (scale_w_k) w_l.mul_(unit_len[0] / unit_len[2]);

    at::Tensor sum_wj2_wk2_wl2 = w_j.pow(2) + w_k.pow(2) + w_l.pow(2);
    sum_wj2_wk2_wl2[0][0][0] = 1.0;

    at::Tensor potential_scale = 1.0 / sum_wj2_wk2_wl2;
    potential_scale[0][0][0] = 0.0;

    at::Tensor force_x_scale = w_j * potential_scale * 0.5;
    at::Tensor force_y_scale = w_k * potential_scale * 0.5;
    at::Tensor force_z_scale = w_l * potential_scale * 0.5;

    at::Tensor force_x_coeff = (torch::tensor(-1, torch::dtype(torch::kFloat).device(device))
                                    .pow(torch::arange(num_bin_x, torch::dtype(torch::kFloat).device(device))))
                                   .unsqueeze(1)
                                   .unsqueeze(2);
    at::Tensor force_y_coeff = (torch::tensor(-1, torch::dtype(torch::kFloat).device(device))
                                    .pow(torch::arange(num_bin_y, torch::dtype(torch::kFloat).device(device))))
                                   .unsqueeze(0)
                                   .unsqueeze(2);
    at::Tensor force_z_coeff = (torch::tensor(-1, torch::dtype(torch::kFloat).device(device))
                                    .pow(torch::arange(num_bin_z, torch::dtype(torch::kFloat).device(device))))
                                   .unsqueeze(0)
                                   .unsqueeze(1);

    at::Tensor potential_coeff = at::tensor(1.0, torch::dtype(torch::kFloat).device(device));
    fft_scale = make_tuple(potential_scale,
                           potential_coeff,
                           force_x_scale,
                           force_y_scale,
                           force_z_scale,
                           force_x_coeff,
                           force_y_coeff,
                           force_z_coeff);
    // TODO: cache
}  // END MODULE

//---------------------------------------------------------------------

at::Tensor ElectronicDensityLayer::get_cache_var(at::Tensor node_pos, at::Tensor node_size, at::Tensor node_weight) {
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

    auto node_pos_channel = node_pos.index({Slice(mov_lhs, mov_rhs), 2});  // fetch z-cord
    //die_info:xl,xh,yl,yh,zl,zh
    auto slicer = die_info[2 * 2 + 1] * 0.5;                               // FIXME:
    auto parter = (node_pos_channel > slicer);
    node_die = torch::_cast_Int(parter);  // TODO:
    auto node_size_morm_util = node_size.clone();
    
    auto ratio = node_pos_channel / die_info[2 * 2 + 1];
    ratio = (-(1.5 + 2 * macro_mask) + ratio * (4 + macro_mask * 4)).unsqueeze(1);;
    ratio = ratio.clamp(0,1);
    // mov_node_size = mov_node_size_top * ratio + mov_node_size_bot * (1-ratio);
    node_size_morm_util.slice(0,mov_lhs, mov_rhs).slice(1,0,2) *= (ratio_difference*(1-ratio)+ratio);
    auto node_height = node_size_morm_util.index({torch::indexing::Slice(), 1});
    int n = 50;
    auto large_macro_mask = node_height > n * node_height.index({torch::indexing::Slice(mov_lhs, mov_rhs)}).mean();
    large_macro_mask = large_macro_mask.unsqueeze(1).expand({-1, 3});
    large_macro_mask = large_macro_mask.to(torch::dtype(node_size_morm_util.dtype()).device(node_size_morm_util.device()));
    node_size_morm_util *= (1 - large_macro_mask);

    // node_size_morm_util.index({Slice(mov_lhs, mov_rhs), 1}) *= node_util_weight_y.index_select(0, node_die);//[CUDAFloatType [947970, 3]]
    // node_size_morm_util.index({Slice(mov_lhs, mov_rhs), 0}) *= node_util_weight_x.index_select(0, node_die);//@@@@@@@

    torch::Tensor normalize_node_info = node_size.new_empty({num_mov_nodes, 7});  // x_l, x_h, y_l, y_h, weight
    normalize_node_info = density_map_normalize_node(node_pos.index({Slice({mov_lhs, mov_rhs})}),
                                                     node_size_morm_util.index({Slice({mov_lhs, mov_rhs})}),
                                                     node_weight.index({Slice({mov_lhs, mov_rhs})}),
                                                     expand_ratio.index({Slice({mov_lhs, mov_rhs})}),
                                                     unit_len,
                                                     normalize_node_info,
                                                     num_bin_x,
                                                     num_bin_y,
                                                     num_bin_z,
                                                     num_mov_nodes);
    torch::Tensor aux_mat = init_density_map.clone();
    // torch::Tensor aux_mat2 = init_density_map.clone();
    // int macro_num = torch::sum(macro_mask).item<int>();
    // auto mov_conn_sorted_map_without_macro = mov_conn_sorted_map.slice(0, macro_num + mov_lhs, mov_rhs);
    // torch::Tensor mov_density_map_without_macro = density_map_forward(
    //     normalize_node_info, mov_conn_sorted_map_without_macro, aux_mat2, num_bin_x, num_bin_y, num_bin_z, num_mov_nodes - macro_num);
    torch::Tensor mov_density_map = density_map_forward(
        normalize_node_info, mov_conn_sorted_map, aux_mat, num_bin_x, num_bin_y, num_bin_z, num_mov_nodes);

    torch::Tensor overflow = overflow_fn(mov_density_map);
    return {overflow, mov_density_map};
}  // END MODULE

//---------------------------------------------------------------------

torch::Tensor ElectronicDensityLayer::density_grad(at::Tensor node_pos,
                                                at::Tensor node_size,
                                                at::Tensor density_map) {
    auto [mov_lhs, mov_rhs, overflow_fn] = overflow_helper;
    // TODO: density cache
    torch::Tensor node_weight;
    auto [tmp1, mov_conn_sorted_map, tmp2] = sorted_maps;
    int num_mov_nodes = mov_rhs - mov_lhs;
    node_weight = get_cache_var(node_pos, node_size, node_weight);

    torch::Tensor normalize_node_info = node_size.new_empty({num_mov_nodes, 7});  // x_l, x_h, y_l, y_h, weight
    normalize_node_info = density_map_normalize_node(node_pos.index({Slice({mov_lhs, mov_rhs})}),
                                                     node_size.index({Slice({mov_lhs, mov_rhs})}),
                                                     node_weight.index({Slice({mov_lhs, mov_rhs})}),
                                                     expand_ratio.index({Slice({mov_lhs, mov_rhs})}),
                                                     unit_len,
                                                     normalize_node_info,
                                                     num_bin_x,
                                                     num_bin_y,
                                                     num_bin_z,
                                                     num_mov_nodes);
    torch::Tensor node_grad = density_backward(
        normalize_node_info, mov_conn_sorted_map, density_map, num_bin_x, num_bin_y, num_bin_z, num_mov_nodes);

    return node_grad;
}

}  // namespace GP3D