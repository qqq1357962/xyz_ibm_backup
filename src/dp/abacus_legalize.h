#pragma once

#include "global.h"

/// A cluster recording abutting cells
/// behave liked a linked list but allocated on a continuous memory
struct AbacusCluster {
    int prev_cluster_id;  ///< previous cluster, set to INT_MIN if the cluster is
                          ///< invalid
    int next_cluster_id;  ///< next cluster, set to INT_MIN if the cluster is
                          ///< invalid
    int bgn_row_node_id;  ///< id of first node in the row
    int end_row_node_id;  ///< id of last node in the row
    double e;             ///< weight of displacement in the objective
    double q;             ///< x = q/e
    double w;             ///< width
    double x;             ///< optimal location

    /// @return whether this is a valid cluster
    bool valid() const { return prev_cluster_id != INT_MIN && next_cluster_id != INT_MIN; }
};

void assignCell2Bin(const torch::Tensor node_pos,
                    const torch::Tensor node_size,
                    torch::Tensor die_info,
                    torch::Tensor node_weight,
                    torch::Tensor bin_size_x,
                    torch::Tensor bin_size_y,
                    torch::Tensor row_shift,
                    int num_bin_x,
                    int num_bin_y,
                    int num_nodes,
                    std::vector<std::vector<int>>& row_cells);

bool abacusPlaceRow(const torch::Tensor node_pos,
                    const torch::Tensor node_size,
                    torch::Tensor node_pos_legal,
                    torch::Tensor node_weight,
                    torch::Tensor row_xl,
                    torch::Tensor row_xh,
                    torch::Tensor rowHeights,
                    const int num_nodes,
                    int* row_nodes,
                    AbacusCluster* clusters,
                    const int num_row_nodes);

void abacusLegalizeRow(const torch::Tensor node_pos,
                       const torch::Tensor node_size,
                       torch::Tensor node_pos_legal,
                       torch::Tensor node_weight,
                       torch::Tensor die_info,
                       torch::Tensor row_size_x,
                       torch::Tensor row_size_y,
                       const int num_row_x,
                       const int num_row_y,
                       const int num_nodes,
                       std::vector<std::vector<int>>& row_cells,
                       std::vector<std::vector<AbacusCluster>>& row_clusters);

void abacusLegalization(const torch::Tensor node_pos,
                        const torch::Tensor node_size,
                        torch::Tensor node_pos_legal,
                        torch::Tensor node_weight,
                        torch::Tensor die_info,
                        torch::Tensor numRows,
                        torch::Tensor rowHeights,
                        int num_bin_x,
                        int num_bin_y,
                        const int num_nodes,
                        torch::Tensor bonding_info = torch::empty({0}));

void abacusLGCrossChip(const torch::Tensor node_pos,
                       torch::Tensor node_pos_legal,
                       const vector<torch::Tensor>& node_pos_mT,
                       const vector<torch::Tensor>& node_size_mT,
                       vector<torch::Tensor>& node_pos_legal_mT,
                       torch::Tensor node_die,
                       vector<vector<int>>& node_maps,
                       torch::Tensor die_info,
                       torch::Tensor numRows,
                       torch::Tensor rowHeights,
                       int num_bin_x,
                       int num_bin_y,
                       const int* num_nodes);

/* unused: ultra slow */
double PlaceRowTrial(const torch::Tensor node_pos,
                     const torch::Tensor node_size,
                     const torch::Tensor row_xl,
                     const torch::Tensor row_xh,
                     const torch::Tensor site_height,
                     const int node_id,
                     const int row_id,
                     const vector<double>& row_cell_area,
                     const vector<vector<int>>& row_cells);