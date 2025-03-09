#pragma once

#include "detailed_place_db.h"
#include "global.h"
#include "pt/patoh.h"

namespace dp {

struct DiePartitioner {
    vector<int> real_id_to_die_id;
    vector<int> die_id_to_real_id;
    
    vector<int> net_list;
    vector<int> net_list_end;

    torch::Tensor net_list_real;
    torch::Tensor net_list_real_end;
    int num_nodes;
    int num_nets;
    int num_pins;
    vector<int> partVec;
    // vector<int> devide_to_pieces(NodeData &data, DetailedPlaceData dp_db, int numPart);
    vector<int> devide_to_pieces(NodeData &data, torch::Tensor node_weight, int numPart);

    vector<int> merge(vector<int> partition_result_1, int valid_part_num_1, vector<int> partition_result_2);
};

}  // namespace dp