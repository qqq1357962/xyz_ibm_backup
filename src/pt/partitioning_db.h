#pragma once

#include "global.h"
#include "placer/database.h"


namespace pt {

int floorDiv(float a, float b);

int ceilDiv(float a, float b);

int floorDivRound(float a, float b, int prec);

int ceilDivRound(float a, float b, int prec);

int roundDiv(float a, float b);

struct Space {
    float xl;
    float xh;
};

struct RowMapIndex {
    int row_id;
    int sub_id;
};

struct BinMapIndex {
    int bin_id;
    int sub_id;
};

struct Box {
    float xl;
    float yl;
    float xh;
    float yh;
    Box() {
        xl = std::numeric_limits<float>::max();
        yl = std::numeric_limits<float>::max();
        xh = std::numeric_limits<float>::lowest();
        yh = std::numeric_limits<float>::lowest();
    }
    Box(float xxl, float yyl, float xxh, float yyh) : xl(xxl), yl(yyl), xh(xxh), yh(yyh) {}

    float center_x() const { return (xl + xh) / 2; }
    float center_y() const { return (yl + yh) / 2; }
};

class PartitionDataTensor {
public:
    PartitionDataTensor(NodeData& data, torch::Tensor node_pos, torch::Tensor node_size);

public:
    /* node info */
    torch::Tensor node_pos_init;
    torch::Tensor exact_node_size;
    torch::Tensor node_die;
    torch::Tensor node_weight;

    /* node info */
    torch::Tensor init_x;
    torch::Tensor init_y;
    torch::Tensor x;
    torch::Tensor y;
    torch::Tensor node_size_x;
    torch::Tensor node_size_y;

    /* pin info */
    torch::Tensor pin_offset_x;
    torch::Tensor pin_offset_y;
    torch::Tensor pin_rel_cpos;

    /* multi lib info */
    vector<torch::Tensor> node_sizes;
    vector<torch::Tensor> node_size_xs;
    vector<torch::Tensor> node_size_ys;
    vector<torch::Tensor> pin_rel_lposes;
    vector<torch::Tensor> pin_offset_xs;
    vector<torch::Tensor> pin_offset_ys;

    /* circuit info */
    torch::Tensor flat_node2pin_start_map;
    torch::Tensor flat_node2pin_map;
    torch::Tensor pin2node_map;
    torch::Tensor flat_net2pin_start_map;
    torch::Tensor flat_net2pin_map;
    torch::Tensor pin2net_map;
    torch::Tensor net_mask;
    int num_pins;

    // void update_node_weight(torch::Tensor weight, torch::Tensor die = torch::empty({0})) { 
    //     node_weight = weight;
    //     if (!die.numel()) {
    //         node_die = weight.clone();
    //     } else {
    //         node_die = die;
    //     }
    // };
    void update_node_pos(torch::Tensor node_pos) {
        node_pos.index({"...", 0}).data().copy_(x + node_size_x / 2);
        node_pos.index({"...", 1}).data().copy_(y + node_size_y / 2);
        init_x.data().copy_(x);
        init_y.data().copy_(y);
    }
};

class PartitionData {
public:
    PartitionData(NodeData& data, PartitionDataTensor& at_db);

public:
    /* node info */
    torch::TensorAccessor<float, 1> x;
    torch::TensorAccessor<float, 1> y;
    torch::TensorAccessor<float, 1> init_x;
    torch::TensorAccessor<float, 1> init_y;
    torch::TensorAccessor<float, 1> node_size_x;
    torch::TensorAccessor<float, 1> node_size_y;

    vector<torch::TensorAccessor<float, 1>> node_size_xs;
    vector<torch::TensorAccessor<float, 1>> node_size_ys;

    /* pin info */
    torch::TensorAccessor<float, 1> pin_offset_x;
    torch::TensorAccessor<float, 1> pin_offset_y;

    vector<torch::TensorAccessor<float, 1>> pin_offset_xs;
    vector<torch::TensorAccessor<float, 1>> pin_offset_ys;

    /* circuit info */
    torch::TensorAccessor<int64_t, 1> flat_node2pin_start_map;
    torch::TensorAccessor<int64_t, 1> flat_node2pin_map;
    torch::TensorAccessor<int64_t, 1> pin2node_map;
    torch::TensorAccessor<int64_t, 1> flat_net2pin_start_map;
    torch::TensorAccessor<int64_t, 1> flat_net2pin_map;
    torch::TensorAccessor<int64_t, 1> pin2net_map;

    /* masks */
    torch::TensorAccessor<int, 1> net_mask;
    torch::TensorAccessor<int, 1> node_weight;
    torch::TensorAccessor<int, 1> node_die;
    // torch::Tensor node_die;

    torch::Tensor partial_hpwl;

    /* chip info */
    float xl;
    float yl;
    float xh;
    float yh;

    /* node info */
    int num_movable_nodes;
    int num_nodes;
    int num_threads;
    int i_bgn;
    int i_end;

public:
    /* net info */
    int num_nets;
    int num_pins;
    torch::Tensor pin_id2node_id;
    torch::Tensor pin_rel_cpos;
    torch::Tensor pin_rel_cpos_bot;
    torch::Tensor pin_rel_cpos_top;
    torch::Tensor pin_pos;
    torch::Tensor pin_die;

};

}  // namespace pt