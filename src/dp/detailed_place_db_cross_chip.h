#pragma once

#include "global.h"
#include "placer/database.h"
#include "gr/ViaData.h"

namespace dpc {

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
    int chip_id;
};

struct BinMapIndex {
    int bin_id;
    int sub_id;
    int chip_id;
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

class DetailedPlaceDataTensor {
public:
    DetailedPlaceDataTensor(NodeData& data, ViaData& via_data, torch::Tensor node_pos, torch::Tensor node_sizes);

public:
    /* node info */
    torch::Tensor node_pos_init;
    torch::Tensor exact_node_size;
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

    /* via info */
    torch::Tensor via_map;
    torch::Tensor via_index;
    torch::Tensor via_site_width;
    torch::Tensor via_site_height;

    void update_node_weight(torch::Tensor weight) { node_weight = weight; };

    void update_node_info(NodeData& data, torch::Tensor node_pos, torch::Tensor node_size, torch::Tensor node_die) {
        // node_pos.index({"...", 0}).data().copy_(x + node_size_x / 2);
        // node_pos.index({"...", 1}).data().copy_(y + node_size_y / 2);

        node_pos.index({"...", 0}).data().copy_(init_x);
        node_pos.index({"...", 1}).data().copy_(init_y);

        // for (int i = 0; i < node_pos.size(0); i++) {
        //     if (((x[i] + node_size_x[i] / 2) != init_x[i]).item<bool>()) {
        //         cout << i << " diff " << (x[i] + node_size_x[i] / 2).item<float>() << " and " <<
        //         init_x[i].item<float>()
        //              << endl;
        //     }
        // }

        node_size.index({"...", 0}).data().copy_(node_size_x);
        node_size.index({"...", 1}).data().copy_(node_size_y);

        node_die.data().copy_(node_weight);
        data.pin_rel_cpos.data().copy_(pin_rel_cpos);

        data.mov_node_weights[1]
            .index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)})
            .data()
            .copy_(node_die.index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)}));
        data.mov_node_weights[0]
            .index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)})
            .data()
            .copy_((1 - node_die).index({Slice(data.cell_mov_lhs, data.cell_mov_rhs)}));
    }
};

class DetailedPlaceData {
public:
    DetailedPlaceData(NodeData& data, DetailedPlaceDataTensor& at_db);
    DetailedPlaceData(NodeData& data, DetailedPlaceDataTensor& at_db, ViaData& via_data);

    vector<shared_ptr<grNode>> nodes;
    vector<shared_ptr<grNet>> nets;

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
    torch::TensorAccessor<int, 2> via_map;
    torch::TensorAccessor<int, 2> via_index;

    /* tensor */
    torch::Tensor partial_hpwl;
    torch::Tensor node_swap;

public:
    /* chip info */
    float xl;
    float yl;
    float xh;
    float yh;

    /* row info */
    int num_sites_x;
    int num_sites_y;
    float row_size_y;
    float row_size_x;
    float row_width;
    float row_height;
    float site_width;
    float row_shift;

    float via_site_width;
    float via_site_height;

    int site_width_safe_divide = 1;

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
    int a;
    int b;

    /* dp bin info */
    int num_bins_x;
    int num_bins_y;
    float bin_size_x;
    float bin_size_y;

    /* node info */
    int num_movable_nodes;
    int num_nodes;
    int num_threads;

    /* rows info */
    vector<int> num_sites_ys;
    vector<float> row_heights;

public:
    inline void shift_box_to_layout(Box& box) const {
        box.xl = std::max(box.xl, xl);
        box.xl = std::min(box.xl, xh);
        box.xh = std::max(box.xh, xl);
        box.xh = std::min(box.xh, xh);
        box.yl = std::max(box.yl, yl);
        box.yl = std::min(box.yl, yh);
        box.yh = std::max(box.yh, yl);
        box.yh = std::min(box.yh, yh);
    }

    inline int pos2bin_x(float xx) const {
        int bx = floorDiv(xx - xl, bin_size_x);
        bx = std::max(bx, 0);
        bx = std::min(bx, num_bins_x - 1);
        return bx;
    }
    inline int pos2bin_y(float yy) const {
        int by = floorDiv(yy - yl, bin_size_y);
        by = std::max(by, 0);
        by = std::min(by, num_bins_y - 1);
        return by;
    }
    inline float align2site(float xx) const { return floorDiv(xx - xl, site_width) * site_width + xl; }
    inline Space align2site(Space space) const {
        space.xl = ceilDiv(space.xl - xl, site_width) * site_width + xl;
        space.xh = floorDiv(space.xh - xl, site_width) * site_width + xl;
        return space;
    }

    inline bool is_dummy_fixed(int node_id) const {
        float height = node_size_y[node_id];
        // DUMMY_FIXED_NUM_ROWS == 2
        if(st::setting.withMacro)
        {
            return (node_id < num_movable_nodes && height > (row_height * 1));
        }
        return (node_id < num_movable_nodes && height > (row_height * 2));
    }

public:
    void make_row2node_map(const torch::TensorAccessor<float, 1> vx,
                           const torch::TensorAccessor<float, 1> vy,
                           std::vector<std::vector<int>>& row2node_map);
    void make_row2node_map(const torch::TensorAccessor<float, 1> vx,
                           const torch::TensorAccessor<float, 1> vy,
                           std::vector<std::vector<std::vector<int>>>& row2node_maps);

    void make_bin2node_map(const torch::TensorAccessor<float, 1> host_x,
                           const torch::TensorAccessor<float, 1> host_y,
                           const torch::TensorAccessor<float, 1> host_node_size_x,
                           const torch::TensorAccessor<float, 1> host_node_size_y,
                           std::vector<std::vector<int>>& bin2node_map,
                           std::vector<BinMapIndex>& node2bin_map);

    void make_bin2node_map(const torch::TensorAccessor<float, 1> host_x,
                           const torch::TensorAccessor<float, 1> host_y,
                           const torch::TensorAccessor<float, 1> host_node_size_x,
                           const torch::TensorAccessor<float, 1> host_node_size_y,
                           std::vector<std::vector<std::vector<int>>>& bin2node_maps,
                           std::vector<BinMapIndex>& node2bin_map);
    float compute_total_hpwl();
    float compute_net_hpwl(int net_id) const;
    float compute_total_hpwl_concurrent(const torch::TensorAccessor<float, 1> x,
                                        const torch::TensorAccessor<float, 1> y,
                                        float* net_hpwls);
    Box compute_optimal_region(int node_id) const;

    float compute_net_hpwl_c(int net_id) const;
    float compute_total_hpwl_concurrent_c(const torch::TensorAccessor<float, 1> x,
                                          const torch::TensorAccessor<float, 1> y,
                                          float* net_hpwls);
};

}  // namespace dpc