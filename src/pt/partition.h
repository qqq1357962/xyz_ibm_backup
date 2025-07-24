#pragma once

#include "global.h"
#include "parser/db/Database.h"
#include "placer/database.h"
#include "placer/run_placement.h"
#include "partitioning_db.h"

class ptNet;
class ptNode;
class Vertex;
typedef shared_ptr<Vertex> VertexPtr;

class placeholder;
typedef shared_ptr<placeholder> placeholderPtr;

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

    Box(const Box& bx_cp) {
        xl = bx_cp.xl;
        yl = bx_cp.yl;
        xh = bx_cp.xh;
        yh = bx_cp.yh;
    }

    float center_x() const { return (xl + xh) / 2; }
    float center_y() const { return (yl + yh) / 2; }

    friend ostream& operator<<(ostream& os, const Box& b) {
        return os << "(" << b.xl << ", " << b.yl << ") (" << b.xh << ", " << b.yh << ") \n";
    }

    float operator+(const Box& b) {
        float x_max_mid = min(xh, b.xh);
        float x_min_mid = max(xl, b.xl);
        float x_ovlp = max(x_max_mid - x_min_mid, (float)0);
        float x_max = max(xh, b.xh);
        float x_min = min(xl, b.xl);
        float x_bbox = x_max - x_min;

        float y_max_mid = min(yh, b.yh);
        float y_min_mid = max(yl, b.yl);
        float y_ovlp = max(y_max_mid - y_min_mid, (float)0);
        float y_max = max(yh, b.yh);
        float y_min = min(yl, b.yl);
        float y_bbox = y_max - y_min;

        return (x_bbox + x_ovlp) + (y_bbox + y_ovlp);
    }
};

struct Macro_Box {
    float xl;
    float yl;
    float xh;
    float yh;

    int node_die;

    Macro_Box(float xxl, float yyl, float xxh, float yyh, int c_id) : xl(xxl), yl(yyl), xh(xxh), yh(yyh), node_die(c_id) {}

    friend ostream& operator<<(ostream& os, const Macro_Box& b) {
        return os << "(" << b.xl << ", " << b.yl << ") (" << b.xh << ", " << b.yh << ") \n";
    }

    bool comp(float xxl, float yyl, float xxh, float yyh, int other_c_id) {
        if (other_c_id == node_die) {
            if (((xxl > xl && xxl < xh) || (xxh < xh && xxh > xl)) && ((yyl > yl && yyl < yh) || (yyh < yh && yyh > yl)))
                return true;
            else
                return false;
        } 
        else
            return false;
    }
};

class ptNet {
public:
    int id = -1;

    ptNet(int i) : id(i) { ; }
    vector<shared_ptr<ptNode>> Nodes;
};

class ptNode {
public:
    int id = -1;
    int group = -1;
    int gain = -1;

    vector<long> sizes;
    ptNode(int i) : id(i) { sizes.reserve(2); }  // bot&top size
    vector<shared_ptr<ptNet>> Nets;
    map<int, float> gain_map;
};  // END MODULE

struct node_greater{
  bool operator()(const shared_ptr<ptNode> a, const shared_ptr<ptNode> b) const{
    return a->gain > b->gain;
  }
};

class placeholder {
public:
    int id = -1;
    int idx = -1;

    placeholder(int i_, int idx_) : id(i_), idx(idx_) { ; }  // bot&top size

    placeholderPtr pre = NULL;
    placeholderPtr next = NULL;
};

// TODO:
class netSorter {
public:   
    int id = -1;

    netSorter(int i) : id(i) { ; }  // bot&top size

    map<int, int> id2idx;

    torch::Tensor node_placer0;
    torch::Tensor node_placer1;

    vector<int> node_placer_raw;
    vector<int> node_die_raw;

    vector<placeholderPtr> node_placer_ptr;
    placeholderPtr head0;
    placeholderPtr head1;
    placeholderPtr tail0;
    placeholderPtr tail1;

    void init_list() {
        int num_nodes = node_placer_raw.size();
        node_placer0 = torch::zeros({num_nodes}, dtype(torch::kInt));
        node_placer1 = torch::zeros({num_nodes}, dtype(torch::kInt));

        head0 = make_shared<placeholder>(-1, -1);
        head1 = make_shared<placeholder>(-1, -1);
        tail0 = make_shared<placeholder>(-1, -1);
        tail1 = make_shared<placeholder>(-1, -1);
        head0->next = tail0;
        head1->next = tail1;
        tail0->pre = head0;
        tail1->pre = head1;

        placeholderPtr tmp0 = head0;
        placeholderPtr tmp1 = head1;
        for (int i = 0 ; i < node_placer_raw.size(); i++) {
            placeholderPtr place_item = make_shared<placeholder>(node_placer_raw[i], i);
            node_placer_ptr.push_back(place_item);
            if (node_die_raw[i] == 0) {
                node_placer0[i] = 1;
                place_item->next = tmp0->next;
                tmp0->next->pre = place_item;
                tmp0->next = place_item;
                place_item->pre = tmp0;
                tmp0 = place_item;
            } else {
                node_placer1[i] = 1;
                place_item->next = tmp1->next;
                tmp1->next->pre = place_item;
                tmp1->next = place_item;
                place_item->pre = tmp1;
                tmp1 = place_item;
            }
        }
    }
};


//-------------------------------------------------------------------------------

class Vertex {
public:
    string _name;
    int id = -1;

public:
    Vertex() { _name = "null"; }
    Vertex(const string& name) : _name(name) { ; }

    int gain;
    int num_list = 0;

    VertexPtr pre = NULL;
    VertexPtr next = NULL;
};

//-------------------------------------------------------------------------------

/// @brief database that place vias
/// follow the PlaceData
/// @param bonding_map via is visiblility
/// @param node_size via node size
/// @param node_pos  via node pos
/// @param node_pos_prime  cell node pos
class Partitioner {
public:
    Partitioner(NodeData& data_, states& hpwl_state_);

    /* vertex op */
    void addVertex(shared_ptr<ptNode> cell);
    void renewVertex(shared_ptr<ptNode> cell);
    void rmVertex(int& cell, int& gain_index);
    void update_vertex(int& cell, int& gain_index);
    /* global op */
    void initList(bool update = false);
    bool check_balance(int idx);
    void update_area(shared_ptr<ptNode> cell_mov);
    void update_gain(shared_ptr<ptNode> cell_mov);
    void swap(shared_ptr<ptNode> cell, int gain_offset);
    /* list op */
    int pop_max();
    void pass();
    void run();
    void rpt_cut_size();
    void greedy_macro_partition(NodeData &data);
    void greedy_partition_by_area(NodeData &data);
    void greedy_partition_by_std_area(NodeData &data);
    void greedy_partition_by_std_area2(NodeData &data);
    torch::Tensor remove_macro_margin(NodeData& data, torch::Tensor node_pos, torch::Tensor node_size, torch::Tensor node_size_backup);
    void greedy_partition_by_maximize_cuts(NodeData& data);
    void adjust_macros(NodeData& data);

public:
    states& hpwl_state;
    /* database info */
    shared_ptr<db::Database> rawdb;
    NodeData& data;
    vector<shared_ptr<ptNode>> nodes;
    vector<shared_ptr<ptNet>> nets;
    int64_t maxDegree = 0;
    int64_t GAIN;
    vector<int64_t> GAINS;

    float GAIN_FLOAT;
    vector<float> GAINS_FLOAT;

public:
    /* circuit info */
    int num_nodes;
    int num_nets;
    int num_pins;
    int num_x_bin;
    int num_y_bin;
    float unit_len_x;
    float unit_len_y;
    torch::Tensor cell_xl;
    torch::Tensor cell_xh;
    torch::Tensor cell_yl;
    torch::Tensor cell_yh;
    torch::Tensor mov_cell_areas;
    torch::Tensor max_mov_cell_areas;
    torch::Tensor node_die;
    int cutsize;
    bool running;
    vector<torch::Tensor> net_masks;

public:
    // vector<bool> freecells;
    torch::Tensor freecells;
    torch::Tensor gainlist;
    torch::Tensor density_gainlist;
    torch::Tensor swap_gainlist;
    torch::Tensor swap_node_id;
    torch::Tensor density_map;
    int swap_cell_num;
    int swap_c_id;
    torch::Tensor via_gainlist;
    vector<vector<int>> bin2node_id;
    vector<int> surround_bin_id;
    int maxGAINIndex;

    torch::Tensor mov_node_xl;
    torch::Tensor mov_node_xh;
    torch::Tensor mov_node_yl;
    torch::Tensor mov_node_yh;
    torch::Tensor mov_node_xl_b;
    torch::Tensor mov_node_yl_b;
    torch::Tensor mov_node_xh_b;
    torch::Tensor mov_node_yh_b;

    /* bucket list */
    vector<VertexPtr> vertexList;
    vector<map<int, VertexPtr>> BucketList;
    vector<int> maxGainIndex;

    /* gain helper */
    vector<int> tracker;
    vector<int> gain_history;

    /* PaToh  https://faculty.cc.gatech.edu/~umit/software.html */
    int numPart = 2;
    vector<int> partitions;

    torch::Tensor net_to_num_nodes;
    torch::Tensor net_to_num_pins;
    torch::Tensor node_naive_flag;
    torch::Tensor node_areas;

    void run_patoh(NodeData& data, bool is_move_macro = true);

    void run_patoh(NodeData &data, torch::Tensor net_wgt_grad);
    void run_patoh_area(NodeData &data);

    void set_macro2d(NodeData& data);
    void set_macro3d(NodeData& data);


    /* GP2D */
    torch::Tensor run_gp2d(NodeData& data);
    torch::Tensor run_gp2d_grid(NodeData& data);
    tuple<torch::Tensor, torch::Tensor, torch::Tensor> run_gp3d(NodeData& data, bool rotate_90);

    void run_patoh_grided(NodeData &data, torch::Tensor node_pos);
    void run_patoh_mononlithic(NodeData &data, torch::Tensor node_pos);
    void run_patoh_sub_grid(NodeData &data, torch::Tensor node_pos);

    Box check_net_weight(NodeData &data, int i, torch::Tensor node_pos);

    /* hpwl-driven fm */
    void run_fm_wl(NodeData &data, bool skip=false);
    void initWLGain(pt::PartitionData& db, bool update);
    void update_gainWL(pt::PartitionData& db, int cell_mov);
    void passWL(pt::PartitionData& db);
    int pop_maxWL();
    bool check_balance_global(int cell_mov_idx);
    void swap_node(pt::PartitionData& db, int cell_mov);
    float overlap(float x_l, float x_h, float bin_x_l) {
        // bin_x_h == bin_x_l + 1
        return std::min(x_h, bin_x_l + 1) - std::max(x_l, bin_x_l);
    }

    shared_ptr<pt::PartitionDataTensor> pt_db_at_ptr;

    float max_hpwl;
    float hpwl_track;
    // int pop_valid(VertexPtr tmp, int gain_idx, int group);
    // bool check_balance_local(int idx);
    // void update_area_local(shared_ptr<ptNode> cell_mov);

    vector<shared_ptr<ptNode>> heap_nodes;
    vector<shared_ptr<netSorter>> net_sorter;

    torch::Tensor node_to_num_neighbor;
    torch::Device device = torch::kCPU;
    torch::Tensor net_mask;
    torch::Tensor node_grid;
    torch::Tensor grid_mov_cell_areas;
    torch::Tensor max_grid_mov_cell_areas;
    torch::Tensor upper_lower_bound_ratio;
    torch::Tensor macro_mask;

    /* Hyperedge */
    torch::Tensor pin_id2node_id;
    torch::Tensor hyperedge_list;
    torch::Tensor hyperedge_list_end;
    torch::Tensor pin_id2net_id;

    torch::Tensor node_pos_2d_ground;

    int max_neighbor;
};  // END MODULE

//-------------------------------------------------------------------------------
