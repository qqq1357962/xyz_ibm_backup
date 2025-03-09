#pragma once

#include "global.h"
#include "hpwl/hpwl.h"
#include "rudy_map/rudy_map.h"

class RudyLayer : public torch::nn::Module {
public:
    RudyLayer(torch::Tensor _unit_len, int _num_bin_x, int _num_bin_y, int _margin)
        : unit_len(_unit_len), num_bin_x(_num_bin_x), num_bin_y(_num_bin_y), margin(_margin) {
        ;
    }

public:
    // torch::Tensor forward(NodeData& data, torch::Tensor node_pos) {
    //     auto [mov_lhs, mov_rhs] = data.movable_index;
    //     auto [fix_lhs, fix_rhs] = data.fixed_connected_index;
    //     torch::Tensor conn_node_pos =
    //         torch::cat({node_pos.index({Slice({mov_lhs, mov_rhs})}), node_pos.index({Slice({fix_lhs, fix_rhs})})},
    //         0);
    //     torch::Tensor pin_pos =
    //         wa_wirelength_hpwl::nodePosToPinPos(conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos);

    //     torch::Tensor horizontal_map = torch::zeros({num_bin_x, num_bin_y});
    //     torch::Tensor vertical_map = torch::zeros({num_bin_x, num_bin_y});

    //     rudy_map_forward_naive(data, pin_pos, unit_len, horizontal_map, vertical_map, num_bin_x, num_bin_y, margin);

    //     auto rudy_map = torch::max(horizontal_map.abs_(), vertical_map.abs_());

    //     return rudy_map;
    // }

    torch::Tensor forward(NodeData& data, torch::Tensor node_pos) {
        auto [mov_lhs, mov_rhs] = data.movable_index;
        auto [fix_lhs, fix_rhs] = data.fixed_connected_index;
        torch::Tensor conn_node_pos =
            torch::cat({node_pos.index({Slice({mov_lhs, mov_rhs})}), node_pos.index({Slice({fix_lhs, fix_rhs})})}, 0);
        torch::Tensor pin_pos =
            wa_wirelength_hpwl::nodePosToPinPos(conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos);

        torch::Tensor pin_density_map = torch::zeros({num_bin_x, num_bin_y});

        pin_density_map_forward_naive(
            data, pin_pos.to(torch::kCPU), unit_len.to(torch::kCPU), pin_density_map, num_bin_x, num_bin_y, margin);

        return pin_density_map;
    }

public:
    torch::Tensor unit_len;
    int num_bin_x;
    int num_bin_y;
    int margin;
};