#pragma once

#include "global.h"
#include "parser/db/Database.h"
#include "placer/database.h"
#include "core/core.h"

class grNet;
class grNode;

class ViaLegalizationData;
class ViaLegalizationDataTensor;

class grNet {
public:
    int id = -1;

    grNet(int i) : id(i) { ; }
    vector<shared_ptr<grNode>> Nodes;
};

class grNode {
public:
    int id = -1;

    vector<long> sizes;
    grNode(int i) : id(i) { sizes.reserve(2); }  // bot&top size
    vector<shared_ptr<grNet>> Nets;
};  // END MODULE

//---------------------------------------------------------------------

/// @brief database that place vias
/// follow the PlaceData
/// @param bonding_map via is visiblility
/// @param node_size via node size
/// @param node_orient via orientation
/// @param node_pos  via node pos
/// @param node_pos_prime  cell node pos
class ViaData : public PlaceData {
public:
    // ViaData(NodeData& data_): data(data_) { ; }
    ViaData() { ; }
    ViaData(NodeData& data_, shared_ptr<db::Database> _rawdb, torch::Tensor node_die_);

    /* for GPU.ver */
    torch::Device device = torch::kCPU;

    void postscale();
    void init_vars();
    tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_mov_node_info();
    void dump(
        torch::Tensor node_pos, torch::Tensor node_size, torch::Tensor node_die, 
        int cell_mov_lhs, int cell_mov_rhs, torch::Tensor cell_orient);

    void to(torch::Device device_) {
        node_size = node_size.to(device_);
        bonding_map = bonding_map.to(device_);
        core_info = core_info.to(device_);
        row_height = row_height.to(device_);
        numRows = numRows.to(device_);
    }

public:
    /* database info */
    shared_ptr<db::Database> rawdb;
    // NodeData& data;
    shared_ptr<NodeData> data_ptr;
    vector<shared_ptr<grNode>> nodes;
    vector<shared_ptr<grNet>> nets;

    /* via info */
    // int num_bonds;
    torch::Tensor bonding_map;
    torch::Tensor mov_cell_areas;
    torch::Tensor max_mov_cell_areas;  // TODO: allowed utilization
    torch::Tensor bondingInfo;

    // torch::Tensor node_die;
    torch::Tensor node_pos_legal;

    torch::Tensor node_swap;

    /* node info */
    int num_cells_prime;
    torch::Tensor cell_pos_prime;
    torch::Tensor cell_die_prime;

    void legalize_via_maching(NodeData& data, ViaLegalizationData& via_db, torch::Tensor node_pos);
};


class ViaLegalizationDataTensor {
public:
    ViaLegalizationDataTensor(ViaData& data, torch::Tensor node_pos, torch::Tensor via_bbox);

public:
    int num_nets;

    /* node info */
    torch::Tensor node_pos;
    torch::Tensor x;
    torch::Tensor y;

    torch::Tensor via_bbox;
    torch::Tensor node_weight;


    torch::Tensor via_site_x;
    torch::Tensor via_site_y;
    torch::Tensor via_map;
    torch::Tensor via_index;

    /* box info */
    torch::Tensor optim_box_x_l;
    torch::Tensor optim_box_x_h;
    torch::Tensor optim_box_y_l;
    torch::Tensor optim_box_y_h;

    torch::Tensor outer_box_x_l;
    torch::Tensor outer_box_x_h;
    torch::Tensor outer_box_y_l;
    torch::Tensor outer_box_y_h;

    int num_via_x;
    int num_via_y;
    float row_shift;
    float via_site_width;
    float via_site_height;

    void update_node_pos(torch::Tensor node_pos) {
        cout << via_site_x.max() << endl;
        cout << via_site_x.min() << endl;

        cout << via_site_y.max() << endl;
        cout << via_site_y.min() << endl;

        node_pos.index({"...", 0}).data().copy_(x);
        node_pos.index({"...", 1}).data().copy_(y);
    }
};

class ViaLegalizationData {
public:
    ViaLegalizationData(ViaData& data, ViaLegalizationDataTensor& at_db);

public:
    /* node info */
    torch::TensorAccessor<float, 1> x;
    torch::TensorAccessor<float, 1> y;
    torch::TensorAccessor<int, 1> node_weight;

    /* via info */
    torch::TensorAccessor<int, 2> via_map;
    torch::TensorAccessor<int, 2> via_index;

    torch::TensorAccessor<float, 2> via_site_x;
    torch::TensorAccessor<float, 2> via_site_y;

    torch::TensorAccessor<float, 1> optim_box_x_l;
    torch::TensorAccessor<float, 1> optim_box_y_l;
    torch::TensorAccessor<float, 1> optim_box_x_h;
    torch::TensorAccessor<float, 1> optim_box_y_h;

    torch::TensorAccessor<float, 1> outer_box_x_l;
    torch::TensorAccessor<float, 1> outer_box_y_l;
    torch::TensorAccessor<float, 1> outer_box_x_h;
    torch::TensorAccessor<float, 1> outer_box_y_h;

    int num_via_x;
    int num_via_y;
    float row_shift;
    float via_site_width;
    float via_site_height;
};
