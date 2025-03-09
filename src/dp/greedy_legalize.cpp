#include "greedy_legalize.h"

void greedyLegalization(const torch::Tensor node_pos,
                        const torch::Tensor node_size,
                        torch::Tensor node_pos_legal,
                        torch::Tensor node_weight,
                        torch::Tensor die_info,
                        torch::Tensor numRows,
                        torch::Tensor rowHeights,
                        int num_bin_x,
                        int num_bin_y,
                        int num_nodes) {
    /* row info */
    torch::Tensor row_size_x = (die_info[1] - die_info[0]) / 1;  // FIXME: hardcode num_bin_x
    torch::Tensor row_size_y = torch::_cast_Double(rowHeights.clone());
    int num_row_x = 1;
    int num_row_y = numRows.item<int>();
    int num_visible_node = node_weight.sum().item<int>();
    int num_invisible_node = num_nodes - num_visible_node;
    printlog(LOG_INFO,
             "Legalizetion row_size (%.5f, %.5f) with %d x %d rows ",
             row_size_x.item().toFloat(),
             row_size_y.item().toFloat(),
             num_row_x,
             num_row_y);
    double row_xl = die_info[0].item<double>();
    double row_xh = die_info[1].item<double>();
    torch::Tensor row_shift = die_info[2];

    vector<int> legalized_node_map(0);  // FIXME:
    for (int dir = 0; dir < 2; dir++) {
        /* recorders: interval/row_area/sorted_node_map */
        vector<vector<Interval>> row_intervals(num_row_x * num_row_y);
        for (int i = 0; i < num_row_x * num_row_y; i++) {
            row_intervals.at(i).emplace_back(Interval(row_xl,
                                                      row_xh,
                                                      (i * row_size_y + row_shift).item<double>(),
                                                      ((i + 1) * row_size_y + row_shift).item<double>()));
        }  // TODO: boundary
        vector<double> row_cell_area(num_row_y, 0);
        /* sort cell */
        auto [tmp1, node_pos_sorted_map_prime] =
            torch::sort(node_pos.index({"...", 0}).flatten(), 0, (dir + 0) % 2);  // TODO: beckward iteration
        auto node_pos_sorted_map = torch::_cast_Int(node_pos_sorted_map_prime).contiguous();
        vector<int> node_map(node_pos_sorted_map.data_ptr<int>(),
                             node_pos_sorted_map.data_ptr<int>() + node_pos_sorted_map.numel());

        legalized_node_map.clear();
        for (int iter = 0; iter < 3; iter++) {  // TODO: #search iteration
            greedyLegalizeRow(node_pos,
                              node_size,
                              node_pos_legal,
                              node_weight,
                              row_size_x,
                              row_size_y,
                              row_shift,
                              num_row_x,
                              num_row_y,
                              node_map,
                              row_cell_area,
                              row_intervals,
                              legalized_node_map,
                              2 + iter);  // TODO: #search range
            printlog(LOG_INFO,
                     "Greedy search iteration %d: %d cells unplaced",
                     iter,
                     (node_map.size() - num_invisible_node));
            if (node_map.size() == num_invisible_node) {
                break;
            }
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

void greedyLegalizeRow(const torch::Tensor node_pos,
                       const torch::Tensor node_size,
                       torch::Tensor node_pos_legal,
                       torch::Tensor node_weight,
                       torch::Tensor row_size_x,
                       torch::Tensor row_size_y,
                       torch::Tensor row_shift,  // TODO: used to solve spacing rule with die boundary
                       int num_row_x,
                       int num_row_y,
                       vector<int>& node_map,
                       vector<double>& row_cell_area,
                       vector<vector<Interval>>& row_intervals,
                       vector<int>& legalized_node_map,
                       const int search_range) {
    double alpha = 0.5;
    double beta = 2;  // TODO: tolerance
    double row_height = row_size_y.item<double>();
    // int search_range = 3;  // FIXME:

    // Iterate over all cells
    // for (int i = 0; i < node_map.size(); i++) {       // FIXME:
    for (int i = node_map.size() - 1; i >= 0; i--) {
        int node_id = node_map[i];
        if (node_weight[node_id].item<int>() != 0) {
            // best record
            int row_best = -1;
            int interval_best = -1;
            double x_l_best = -1;
            double y_l_best = -1;
            double cost_best = INT_MAX;
            double cost;
            // align cell to closest row
            // bottom left cord
            // double init_xl = (node_pos[node_id][0] - node_size[node_id][0] / 2).item<double>();
            // double init_yl = (row_size_y * prime_row_id + node_size[node_id][1] / 2).item<double>();
            double init_xl =
                (alpha * node_pos[node_id][0] + (1 - alpha) * node_pos_legal[node_id][0] - node_size[node_id][0] / 2)
                    .item<double>();
            double init_yl =
                (alpha * node_pos[node_id][1] + (1 - alpha) * node_pos_legal[node_id][1] - node_size[node_id][1] / 2)
                    .item<double>();
            double init_yc = (alpha * node_pos[node_id][1] + (1 - alpha) * node_pos_legal[node_id][1]).item<double>();
            double width = node_size[node_id][0].item<double>();
            int prime_row_id = max(0, torch::floor((init_yc - row_shift) / row_size_y).item().toInt());

            // Iterate over neighbour rows
            for (int offset = 0; abs(offset) < search_range; offset = (offset > 0) ? -offset : -(offset - 1)) {
                int row_id = prime_row_id + offset;
                if (row_id < 0 || row_id >= num_row_y) continue;
                if ((row_cell_area[row_id] + node_size[node_id][0]).item<double>() > row_size_x.item<double>())
                    continue;
                // best record within this row
                int interval_best_row = -1;
                double x_l_best_row = -1;
                double y_l_best_row = -1;
                double cost_best_row = INT_MAX;

                const std::vector<Interval>& intervals = row_intervals.at(row_id);
                // Iterate over intervals in this row
                bool search_flag = true;
                for (unsigned int j = 0; search_flag && j < intervals.size(); ++j) {
                    auto interval = intervals.at(j);

                    double intersect_width = interval.x_h - interval.x_l;
                    if (intersect_width >= width) {
                        // compute displacement
                        double target_xl = init_xl;
                        double target_yl = interval.y_l;
                        // alow tolerance to avoid more dead space
                        double tolerance = std::min(beta * width, intersect_width / beta);
                        if (target_xl <= interval.x_l + tolerance) {
                            target_xl = interval.x_l;
                        } else if (target_xl + width >= interval.x_h - tolerance) {
                            target_xl = (interval.x_h - width);
                        }
                        cost = fabs(target_xl - init_xl) + fabs(target_yl - init_yl);
                        if (cost < cost_best_row) {
                            cost_best_row = cost;
                            x_l_best_row = target_xl;
                            y_l_best_row = target_yl;
                            interval_best_row = j;
                        } else  // early exit since we iterate within rows from left to right
                        {
                            search_flag = false;
                        }
                    }
                }
                if (cost_best_row < cost_best) {
                    cost_best = cost_best_row;
                    x_l_best = x_l_best_row;
                    y_l_best = y_l_best_row;
                    interval_best = interval_best_row;
                    row_best = row_id;
                } else if (cost_best + row_height <
                           offset * row_height)  // early exit since we iterate from close row to far-away row
                {
                    break;
                }
            }

            // free sites found
            if (row_best != -1) {
                // update cell position
                node_pos_legal[node_id][0].data().copy_(x_l_best + node_size[node_id][0] / 2);
                node_pos_legal[node_id][1].data().copy_(y_l_best + node_size[node_id][1] / 2);
                row_cell_area[row_best] += node_size[node_id][0].item<double>();

                // update interval
                std::vector<Interval>& intervals = row_intervals.at(row_best);
                Interval& interval = intervals.at(interval_best);
                assert(x_l_best >= interval.x_l && x_l_best + width <= interval.x_h);
                if (x_l_best == interval.x_l) {
                    // Case1: stick to left
                    interval.x_l += width;
                    if (interval.x_l >= interval.x_h) {
                        row_intervals.at(row_best).erase(row_intervals.at(row_best).begin() + interval_best);
                    }
                } else if (x_l_best + width == interval.x_h) {
                    // Case2: stick to right
                    interval.x_h -= width;
                    if (interval.x_l >= interval.x_h) {
                        row_intervals.at(row_best).erase(row_intervals.at(row_best).begin() + interval_best);
                    }
                } else {
                    // Case2: in the middle; insert to the right
                    Interval new_interval(x_l_best + width, interval.x_h, interval.y_l, interval.y_h);
                    interval.x_h = x_l_best;
                    row_intervals.at(row_best).insert(row_intervals.at(row_best).begin() + interval_best + 1,
                                                      new_interval);
                }
                // remove from cells
                node_map.erase(node_map.begin() + i);
                legalized_node_map.push_back(node_id);  // FIXME:
            }
        }
    }
}