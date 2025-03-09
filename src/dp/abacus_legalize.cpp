#include "abacus_legalize.h"

void assignCell2Bin(const torch::Tensor node_pos, const torch::Tensor node_size, torch::Tensor die_info,
                    torch::Tensor node_weight, torch::Tensor bin_size_x, torch::Tensor bin_size_y,
                    torch::Tensor row_shift, int num_bin_x, int num_bin_y, int num_nodes,
                    std::vector<std::vector<int>>& row_cells) {
    vector<double> bin_cell_area;
    bin_cell_area.resize(row_cells.size(), 0);
    int serach_dir[2] = {-1, 1};
    for (int i = 0; i < num_nodes; i++) {
        if (node_weight[i].item<int>() == 0) continue;
        int bin_id_x = torch::floor(node_pos[i][0] / bin_size_x).item<int>();                // FIXME: center
        int bin_id_y = torch::floor((node_pos[i][1] - row_shift) / bin_size_y).item<int>();  // FIXME: floor

        bin_id_x = std::min(std::max(bin_id_x, 0), num_bin_x - 1);
        bin_id_y = std::min(std::max(bin_id_y, 0), num_bin_y - 1);
        int bin_id = bin_id_y * num_bin_x + bin_id_x;

        // move cell to a row with enough capacity
        vector<bool> bin_y_visited;
        bin_y_visited.resize(row_cells.size(), 0);
        queue<int> rows_to_search;
        rows_to_search.push(bin_id_y);

        while (!rows_to_search.empty()) {
            int cur_row = rows_to_search.front();
            bin_y_visited[cur_row] = true;
            bin_id = cur_row * num_bin_x + bin_id_x;
            if ((bin_cell_area[bin_id] + node_size[i][0]).item<double>() <= bin_size_x.item<double>()) {
                bin_cell_area[bin_id] += node_size[i][0].item<double>();
                bin_id_y = cur_row;
                break;
            }
            rows_to_search.pop();
            for (int j : serach_dir) {
                int next_row = cur_row + j;
                if (next_row >= 0 && next_row < num_bin_y && !bin_y_visited[next_row]) {
                    rows_to_search.push(next_row);
                }
            }
        }
        // FIXME: center -> lower
        // node_pos[i][1] = bin_id_y * bin_size_y + node_size[i][1] / 2;  // FIXME: validate row capacity
        node_pos[i][1].data().copy_(
            (bin_id_y * bin_size_y + node_size[i][1] / 2 + row_shift).data());  // FIXME: validate row capacity
        row_cells[bin_id].push_back(i);
    }
}  // END MODULE

//---------------------------------------------------------------------

bool abacusPlaceRow(const torch::Tensor node_pos, const torch::Tensor node_size, torch::Tensor node_pos_legal,
                    torch::Tensor node_weight, torch::Tensor row_xl, torch::Tensor row_xh, torch::Tensor rowHeights,
                    const int num_nodes, int* row_nodes, AbacusCluster* clusters, const int num_row_nodes) {
    double xl = row_xl.item<double>();
    double xh = row_xh.item<double>();
    double row_height = rowHeights.item<double>();
    // a very large number
    double M = pow(10, ceil(log((xh - xl) * num_row_nodes) / log(10)));
    bool ret_flag = true;

    // merge two clusters
    // the second cluster will be invalid
    auto merge_cluster = [&](int dst_cluster_id, int src_cluster_id) {
        assert(dst_cluster_id < num_row_nodes);
        AbacusCluster& dst_cluster = clusters[dst_cluster_id];
        assert(src_cluster_id < num_row_nodes);
        AbacusCluster& src_cluster = clusters[src_cluster_id];

        assert(dst_cluster.valid() && src_cluster.valid());
        for (int i = dst_cluster_id + 1; i < src_cluster_id; ++i) {
            assert(!clusters[i].valid());
        }
        dst_cluster.end_row_node_id = src_cluster.end_row_node_id;
        assert(dst_cluster.e < M && src_cluster.e < M);
        dst_cluster.e += src_cluster.e;
        dst_cluster.q += src_cluster.q - src_cluster.e * dst_cluster.w;
        dst_cluster.w += src_cluster.w;
        // dst_cluster.x += src_cluster.x - dst_cluster.w;
        // update linked list
        if (src_cluster.next_cluster_id < num_row_nodes) {
            clusters[src_cluster.next_cluster_id].prev_cluster_id = dst_cluster_id;
        }
        dst_cluster.next_cluster_id = src_cluster.next_cluster_id;
        src_cluster.prev_cluster_id = INT_MIN;
        src_cluster.next_cluster_id = INT_MIN;
    };

    // collapse clusters between [0, cluster_id]
    // compute the locations and merge clusters
    auto collapse = [&](int cluster_id, double range_xl, double range_xh) {
        int cur_cluster_id = cluster_id;
        assert(cur_cluster_id < num_row_nodes);
        int prev_cluster_id = clusters[cur_cluster_id].prev_cluster_id;
        AbacusCluster* cluster = nullptr;
        AbacusCluster* prev_cluster = nullptr;

        while (true) {
            assert(cur_cluster_id < num_row_nodes);
            cluster = &clusters[cur_cluster_id];
            // cluster->x = cluster->q / cluster->e;
            cluster->x = round(cluster->q / cluster->e);  // FIXME: very important
            // make sure cluster >= range_xl, so fixed nodes will not be moved
            // in illegal case, cluster+w > range_xh may occur, but it is OK.
            // We can collect failed clusters later
            cluster->x = std::max(std::min(cluster->x, range_xh - cluster->w), range_xl);
            if (!(cluster->x >= range_xl && cluster->x + cluster->w <= range_xh)) {
                cout << "Out of range " << cluster->x << " " << range_xl << " " << cluster->w << " " << range_xh
                     << endl;
            }
            // assert(cluster->x >= range_xl && cluster->x + cluster->w <= range_xh);

            prev_cluster_id = cluster->prev_cluster_id;
            if (prev_cluster_id >= 0) {
                prev_cluster = &clusters[prev_cluster_id];
                if (prev_cluster->x + prev_cluster->w > cluster->x) {
                    merge_cluster(prev_cluster_id, cur_cluster_id);
                    cur_cluster_id = prev_cluster_id;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    };

    // initial cluster has only one cell
    for (int i = 0; i < num_row_nodes; ++i) {
        int node_id = row_nodes[i];
        AbacusCluster& cluster = clusters[i];
        cluster.prev_cluster_id = i - 1;
        cluster.next_cluster_id = i + 1;
        cluster.bgn_row_node_id = i;
        cluster.end_row_node_id = i;
        cluster.e = 1.0;                                                                            // TODO: Macro
        cluster.q = cluster.e * (node_pos[node_id][0] - node_size[node_id][0] / 2).item<double>();  // FIXME: left x
        cluster.w = node_size[node_id][0].item<double>();
        // this is required since we also include fixed nodes
        cluster.x = (node_pos[node_id][0] - node_size[node_id][0] / 2).item<double>();
    }

    // kernel algorithm for placeRow
    double range_xl = xl;
    double range_xh = xh;
    for (int i = 0; i < num_row_nodes; ++i) {
        const AbacusCluster& cluster = clusters[i];
        if (!(std::abs(node_size[row_nodes[i]][1].item<double>() - row_height) < 1e-6)) {
            cout << node_size[row_nodes[i]][1].item<double>() << " | " << row_height << endl;
        }
        assert(std::abs(node_size[row_nodes[i]][1].item<double>() - row_height) < 1e-6);
        collapse(i, range_xl, range_xh);
    }

    // apply solution
    for (int i = 0; i < num_row_nodes; ++i) {
        if (clusters[i].valid()) {
            const AbacusCluster& cluster = clusters[i];
            double xc = cluster.x;  // FIXME: left x
            for (int j = cluster.bgn_row_node_id; j <= cluster.end_row_node_id; ++j) {
                int node_id = row_nodes[j];
                if (node_id < num_nodes && std::abs(node_size[node_id][1].item<double>() - row_height) < 1e-6) {
                    // node_pos_legal[node_id][0] = xc + node_size[node_id][0] / 2;  // FIXME: center x
                    node_pos_legal[node_id][0].data().copy_(xc + node_size[node_id][0] / 2);  // FIXME: center x
                } else if (xc != node_pos_legal[node_id][0].item<double>()) {
                    if (node_id < num_nodes)
                        printlog(LOG_WARN,
                                 "multi-row node %d tends to move from %.12f to "
                                 "%.12f, ignored\n",
                                 node_id, node_pos_legal[node_id][0].item<double>(), xc);
                    else
                        printlog(LOG_WARN, "fixed node %d tends to move from %.12f to %.12f, ignored\n", node_id,
                                 node_pos_legal[node_id][0].item<double>(), xc);
                    ret_flag = false;
                }
                xc += node_size[node_id][0].item<double>();
            }
        }
    }

    return ret_flag;
}  // END MODULE

//---------------------------------------------------------------------

void abacusLegalizeRow(const torch::Tensor node_pos, const torch::Tensor node_size, torch::Tensor node_pos_legal,
                       torch::Tensor node_weight, torch::Tensor die_info, torch::Tensor row_size_x,
                       torch::Tensor row_size_y, const int num_row_x, const int num_row_y, const int num_nodes,
                       std::vector<std::vector<int>>& row_cells,
                       std::vector<std::vector<AbacusCluster>>& row_clusters) {
    for (unsigned int i = 0; i < row_cells.size(); i++) {
        auto& row2nodes = row_cells.at(i);
        // sort by center pos
        std::sort(row2nodes.begin(), row2nodes.end(), [&](int node_id1, int node_id2) {
            auto x1 = node_pos[node_id1][0].item<double>();
            auto x2 = node_pos[node_id2][0].item<double>();
            return x1 < x2 || (x1 == x2 && node_id1 < node_id2);
        });

        auto& clusters = row_clusters.at(i);
        int num_row_nodes = row2nodes.size();
        int row_id_y = i / num_row_x;
        int row_id_x = i - row_id_y * num_row_x;

        // torch::Tensor row_xl = (die_info[0] + row_size_x * row_id_x);
        // torch::Tensor row_xh = torch::min(row_xl + row_size_x, die_info[1]);
        torch::Tensor row_xl = die_info[0];
        torch::Tensor row_xh = die_info[1];

        abacusPlaceRow(node_pos, node_size, node_pos_legal, node_weight, row_xl, row_xh, row_size_y, num_nodes,
                       row2nodes.data(), clusters.data(), num_row_nodes);
    }
}  // END MODULE

//---------------------------------------------------------------------

void abacusLegalization(const torch::Tensor node_pos, const torch::Tensor node_size, torch::Tensor node_pos_legal,
                        torch::Tensor node_weight, torch::Tensor die_info, torch::Tensor numRows,
                        torch::Tensor rowHeights, int num_bin_x, int num_bin_y, const int num_nodes,
                        torch::Tensor bonding_info) {
    /* row info */
    torch::Tensor row_size_x = (die_info[1] - die_info[0]) / 1;  // FIXME: hardcode num_bin_x
    if (bonding_info.numel()) {
        auto site_width = (bonding_info[0] + bonding_info[2]);
        row_size_x = torch::floor((die_info[1] - die_info[0]) / site_width) * site_width;
    }
    torch::Tensor row_size_y = torch::_cast_Double(rowHeights.clone());
    int num_row_x = 1;
    int num_row_y = numRows.item<int>();
    int num_visible_node = node_weight.sum().item<int>();
    int num_invisible_node = num_nodes - num_visible_node;
    printlog(LOG_INFO, "Legalizetion bin_size (%.5f, %.5f) with %d x %d rows ", row_size_x.item().toFloat(),
             row_size_y.item().toFloat(), num_row_x, num_row_y);
    // double row_xl = die_info[0].item<double>();
    // double row_xh = die_info[1].item<double>();
    torch::Tensor row_shift = die_info[2];

    // torch::Tensor bin_size_x = die_info[1] / 1;  // FIXME: hardcode num_bin_x
    // torch::Tensor bin_size_y = (die_info[3] - die_info[2]) / num_bin_y;
    // distribute cells to bins
    std::vector<std::vector<int>> row_cells(num_row_x * num_row_y);
    assignCell2Bin(node_pos_legal, node_size, die_info, node_weight, row_size_x, row_size_y, row_shift, num_row_x,
                   num_row_y, num_nodes, row_cells);

    // FIXME: float type round issue
    // for (int i = 0; i < num_nodes; i++) {
    //     if (node_weight[i].item<int>() == 0) continue;
    //     int node_x_safe = (node_pos[i][0] - node_size[i][0] / 2).item<int>();
    //     node_pos[i][0].data().copy_(node_x_safe + node_size[i][0] / 2);
    // }

    /* create clusters and place each row*/
    std::vector<std::vector<AbacusCluster>> row_clusters(num_row_x * num_row_y);
    for (unsigned int i = 0; i < row_cells.size(); ++i) {
        row_clusters[i].resize(row_cells[i].size());
    }
    abacusLegalizeRow(node_pos, node_size, node_pos_legal, node_weight, die_info, row_size_x, row_size_y, num_row_x,
                      num_row_y, num_nodes, row_cells, row_clusters);

    if (bonding_info.numel()) {
        auto site_width = (bonding_info[0] + bonding_info[2]);
        auto site_height = (bonding_info[1] + bonding_info[2]);
        logger.info("Align to dummy site(%.2f) for vias", site_width.item<float>());
        auto row_height = rowHeights;
        assert(abs((row_height - site_height).item<float>()) < 1e-4);
        // this also considers cell width which is not integral times of site_width
        for (auto const& cells : row_cells) {
            for (auto node_id : cells) {
                if (node_weight[node_id].item<int>() == 0) continue;

                node_pos_legal[node_id][0] =
                    torch::round((node_pos_legal[node_id][0] - node_size[node_id][0] / 2 - row_shift) / site_width) * site_width +
                    node_size[node_id][0] / 2 + row_shift;
            }
        }
    }
    auto displace = torch::mean(torch::abs(node_pos - node_pos_legal), 0);
    printlog(LOG_INFO, "Legalizatoin displacement: (%.5f, %.5f)", displace[0].item().toFloat(),
             displace[1].item().toFloat());
}  // END MODULE
