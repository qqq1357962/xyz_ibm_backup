#pragma once

#include "global.h"

struct Interval {
    double x_l;
    double x_h;
    double y_l;
    double y_h;

    Interval(double xl, double xh, double yl, double yh) : x_l(xl), x_h(xh), y_l(yl), y_h(yh) {}
    void intersect(double rhs_xl, double rhs_xh) {
        x_l = std::max(x_l, rhs_xl);
        x_h = std::min(x_h, rhs_xh);
    }
};

void greedyLegalizeRow(const torch::Tensor node_pos,
                       const torch::Tensor node_size,
                       torch::Tensor node_pos_legal,
                       torch::Tensor node_weight,
                       torch::Tensor row_size_x,
                       torch::Tensor row_size_y,
                       torch::Tensor row_shift,
                       int num_row_x,
                       int num_row_y,
                       vector<int>& node_map,
                       vector<double>& row_cell_area,
                       vector<vector<Interval>>& row_intervals,
                       vector<int>& legalized_node_map,  // FIXME:
                       const int search_range = 3);

void greedyLegalization(const torch::Tensor node_pos,
                        const torch::Tensor node_size,
                        torch::Tensor node_pos_legal,
                        torch::Tensor node_weight,
                        torch::Tensor die_info,
                        torch::Tensor numRows,
                        torch::Tensor rowHeights,
                        int num_bin_x,
                        int num_bin_y,
                        int num_nodes);

void greedyLegalizeRowCrossChip(const torch::Tensor node_pos,
                               const torch::Tensor node_sizes,
                               torch::Tensor node_pos_legal,
                               torch::Tensor node_die,
                               torch::Tensor row_size_x,
                               torch::Tensor die_row_size_y,
                               int num_row_x,
                               int* die_num_row_y,
                               vector<int>& node_map,
                               torch::Tensor max_mov_cell_areas,
                               vector<vector<double>>& die_row_cell_area,
                               vector<vector<vector<Interval>>>& die_row_intervals,
                               vector<int>& legalized_node_map,
                               const int search_range);

void greedyLGCrossChip(const torch::Tensor node_pos,
                       const torch::Tensor node_sizes,
                       torch::Tensor node_pos_legal,
                       torch::Tensor node_die,
                       torch::Tensor die_info,
                       torch::Tensor numRows,
                       torch::Tensor rowHeights,
                       torch::Tensor die_mov_cell_areas,
                       int num_bin_x,
                       int num_bin_y,
                       int num_nodes);