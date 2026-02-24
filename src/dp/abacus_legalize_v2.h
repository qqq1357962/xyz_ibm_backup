#pragma once

#include "detailed_place_db.h"
#include "global.h"
#include "core/core.h"

namespace dp {

struct AbacusCluster {
    int prev_cluster_id;  ///< previous cluster, set to INT_MIN if the cluster is
                          ///< invalid
    int next_cluster_id;  ///< next cluster, set to INT_MIN if the cluster is
                          ///< invalid
    int bgn_row_node_id;  ///< id of first node in the row
    int end_row_node_id;  ///< id of last node in the row
    float e;              ///< weight of displacement in the objective
    float q;              ///< x = q/e
    float w;              ///< width
    float x;              ///< optimal location

    /// @return whether this is a valid cluster
    bool valid() const { return prev_cluster_id != INT_MIN && next_cluster_id != INT_MIN; }
};

void abacusLegalizationV2(NodeData& data, DetailedPlaceData& db, int num_bins_x, int num_bins_y, int layer, float step);
}  // namespace dp