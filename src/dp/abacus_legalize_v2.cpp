#include "abacus_legalize_v2.h"

namespace dp {

void distributeMovableAndFixedCells2BinsCPU(DetailedPlaceData& db,
                                            std::vector<std::vector<int>>& bin_cells,
                                            int num_bins_x,
                                            int num_bins_y,
                                            float bin_size_x,
                                            float bin_size_y) {
    for (int i = 0; i < db.num_nodes; i += 1) {
        if (i < db.num_movable_nodes &&
            db.node_size_y[i] <= bin_size_y)  // single-row movable nodes only distribute to one bin
        {
            if (db.node_weight[i] == 0) continue;
            int bin_id_x = (db.x[i] + db.node_size_x[i] / 2 - db.xl) / bin_size_x;
            int bin_id_y = (db.y[i] + db.node_size_y[i] / 2 - db.yl) / bin_size_y;

            bin_id_x = std::min(std::max(bin_id_x, 0), num_bins_x - 1);
            bin_id_y = std::min(std::max(bin_id_y, 0), num_bins_y - 1);

            int bin_id = bin_id_x * num_bins_y + bin_id_y;

            bin_cells[bin_id].push_back(i);
        } else  // fixed nodes may distribute to multiple bins
        {
            if (db.node_weight[i] == 0 || i >= db.num_movable_nodes) continue;  // FIXME
            int node_id = i;
            int bin_id_xl = std::max((db.x[node_id] - db.xl) / bin_size_x, (float)0);
            int bin_id_xh =
                std::min((int)ceil((db.x[node_id] + db.node_size_x[node_id] - db.xl) / bin_size_x), num_bins_x);
            int bin_id_yl = std::max((db.y[node_id] - db.yl) / bin_size_y, (float)0);
            int bin_id_yh =
                std::min((int)ceil((db.y[node_id] + db.node_size_y[node_id] - db.yl) / bin_size_y), num_bins_y);

            for (int bin_id_x = bin_id_xl; bin_id_x < bin_id_xh; ++bin_id_x) {
                for (int bin_id_y = bin_id_yl; bin_id_y < bin_id_yh; ++bin_id_y) {
                    int bin_id = bin_id_x * num_bins_y + bin_id_y;
                    bin_cells[bin_id].push_back(node_id);
                }
            }
        }
    }
}

bool abacusPlaceRowCPU(DetailedPlaceData& db, int* row_nodes, AbacusCluster* clusters, const int num_row_nodes) {
    // a very large number
    float M = pow(10, ceil(log((db.xh - db.xl) * num_row_nodes) / log(10)));
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
    auto collapse = [&](int cluster_id, float range_xl, float range_xh) {
        int cur_cluster_id = cluster_id;
        assert(cur_cluster_id < num_row_nodes);
        int prev_cluster_id = clusters[cur_cluster_id].prev_cluster_id;
        AbacusCluster* cluster = nullptr;
        AbacusCluster* prev_cluster = nullptr;

        while (true) {
            assert(cur_cluster_id < num_row_nodes);
            cluster = &clusters[cur_cluster_id];
            cluster->x = cluster->q / cluster->e;
            // make sure cluster >= range_xl, so fixed nodes will not be moved
            // in illegal case, cluster+w > range_xh may occur, but it is OK.
            // We can collect failed clusters later
            cluster->x = std::max(std::min(cluster->x, range_xh - cluster->w), range_xl);
            assert(cluster->x >= range_xl && cluster->x + cluster->w <= range_xh);

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
        cluster.e = (node_id < db.num_movable_nodes && db.node_size_y[node_id] <= db.row_height)
                        ? 1.0 /*node_weights[node_id]*/
                        : M;
        cluster.q = cluster.e * db.init_x[node_id];
        cluster.w = db.node_size_x[node_id];
        // this is required since we also include fixed nodes
        cluster.x = (node_id < db.num_movable_nodes && db.node_size_y[node_id] > db.row_height) ? db.x[node_id]
                                                                                                : db.init_x[node_id];
    }

    // kernel algorithm for placeRow //@@
    float range_xl = db.xl;
    float range_xh = db.xh;
    for (int j = 0; j < num_row_nodes; ++j) {
        const AbacusCluster& next_cluster = clusters[j];
        if (next_cluster.e >= M)  // fixed node
        {
            range_xh = std::min(next_cluster.x, range_xh);
            break;
        } else {
            if (std::abs(db.node_size_y[row_nodes[j]] - db.row_height) > 1e-6)
            {
                cout << row_nodes[j] << endl;
                cout << std::abs(db.node_size_y[row_nodes[j]] - db.row_height) << endl;
                cout << db.node_size_y[row_nodes[j]] << endl;
            }
            assert(db.node_size_y[row_nodes[j]] - db.row_height < 1e-6);
        }
    }
    for (int i = 0; i < num_row_nodes; ++i) {
        const AbacusCluster& cluster = clusters[i];
        if (cluster.e < M) {
            if (std::abs(db.node_size_y[row_nodes[i]] - db.row_height) > 1e-6)
            {
                cout << row_nodes[i] << endl;
                cout << std::abs(db.node_size_y[row_nodes[i]] - db.row_height) << endl;
                cout << db.node_size_y[row_nodes[i]] << endl;
            }
            assert(db.node_size_y[row_nodes[i]] - db.row_height < 1e-6);
            collapse(i, range_xl, range_xh);
        } else  // set range xl/xh according to fixed nodes
        {
            range_xl = cluster.x + cluster.w;
            range_xh = db.xh;
            for (int j = i + 1; j < num_row_nodes; ++j) {
                const AbacusCluster& next_cluster = clusters[j];
                if (next_cluster.e >= M)  // fixed node
                {
                    range_xh = std::min(next_cluster.x, range_xh);
                    break;
                }
            }
        }
    }

    // apply solution
    for (int i = 0; i < num_row_nodes; ++i) {
        if (clusters[i].valid()) {
            const AbacusCluster& cluster = clusters[i];
            float xc = cluster.x;
            for (int j = cluster.bgn_row_node_id; j <= cluster.end_row_node_id; ++j) {
                int node_id = row_nodes[j];
                if (node_id < db.num_movable_nodes && std::abs(db.node_size_y[node_id] - db.row_height) < 1e-6) {
                    db.x[node_id] = xc;
                } else if (xc != db.x[node_id]) {
                    if (node_id < db.num_movable_nodes)
                        logger.warning(
                            "multi-row node %d tends to move from %.12f to "
                            "%.12f, ignored",
                            node_id,
                            db.x[node_id],
                            xc);
                    else
                        logger.warning(
                            "fixed node %d tends to move from %.12f to %.12f, ignored", node_id, db.x[node_id], xc);
                    ret_flag = false;
                }
                xc += db.node_size_x[node_id];
            }
        }
    }

    return ret_flag;
}

void abacusLegalizeRowCPU(DetailedPlaceData& db,
                          std::vector<std::vector<int>>& bin_cells,
                          std::vector<std::vector<AbacusCluster>>& bin_clusters,
                          int num_bins_x,
                          int num_bins_y,
                          float bin_size_x,
                          float bin_size_y) {
    for (unsigned int i = 0; i < bin_cells.size(); i += 1) {
        auto& row2nodes = bin_cells.at(i);

        // sort bin cells from left to right
        // we need to remove fixed cells if it is inside another fixed cell
        // first sort by left edge
        std::sort(row2nodes.begin(), row2nodes.end(), [&](int node_id1, int node_id2) {
            float x1 = db.x[node_id1];
            float x2 = db.x[node_id2];
            return x1 < x2 || (x1 == x2 && node_id1 < node_id2);//@@存疑
        });
        // After sorting by left edge,
        // there is a special case for fixed cells where
        // one fixed cell is completely within another in a row.
        // This will cause failure to detect some overlaps.
        // We need to remove the "small" fixed cell that is inside another.
        if (!row2nodes.empty()) {
            std::vector<int> tmp_nodes;
            tmp_nodes.reserve(row2nodes.size());
            tmp_nodes.push_back(row2nodes.front());
            for (int j = 1, je = row2nodes.size(); j < je; ++j) {
                int node_id1 = row2nodes.at(j - 1);
                int node_id2 = row2nodes.at(j);
                // two fixed cells
                if (node_id1 >= db.num_movable_nodes && node_id2 >= db.num_movable_nodes) {
                    float xl1 = db.x[node_id1];
                    float xl2 = db.x[node_id2];
                    float width1 = db.node_size_x[node_id1];
                    float width2 = db.node_size_x[node_id2];
                    float xh1 = xl1 + width1;
                    float xh2 = xl2 + width2;
                    // only collect node_id2 if its right edge is righter than node_id1
                    if (xh1 < xh2) {
                        tmp_nodes.push_back(node_id2);
                    }
                } else {
                    tmp_nodes.push_back(node_id2);
                }
            }
            row2nodes.swap(tmp_nodes);

            // sort according to center
            std::sort(row2nodes.begin(), row2nodes.end(), [&](int node_id1, int node_id2) {
                float x1 = db.x[node_id1] + db.node_size_x[node_id1] / 2;
                float x2 = db.x[node_id2] + db.node_size_x[node_id2] / 2;
                return x1 < x2 || (x1 == x2 && node_id1 < node_id2);
            });
        }

        auto& clusters = bin_clusters.at(i);
        int num_row_nodes = row2nodes.size();

        int bin_id_x = i / num_bins_y;
        // int bin_id_y = i-bin_id_x*num_bins_y;

        float bin_xl = db.xl + bin_size_x * bin_id_x;
        float bin_xh = std::min(bin_xl + bin_size_x, db.xh);

        abacusPlaceRowCPU(db, row2nodes.data(), clusters.data(), num_row_nodes);
    }
    float displace = 0;
    for (int i = 0; i < db.num_movable_nodes; ++i) {
        if (db.node_weight[i] == 0) continue;
        displace += fabs(db.x[i] - db.init_x[i]);
    }
    logger.debug("average displace = %g", displace / db.num_movable_nodes);
}

void abacusLegalizationV2(DetailedPlaceData& db, int num_bins_x, int num_bins_y) {
    // adjust bin sizes
    float bin_size_x = (db.xh - db.xl) / num_bins_x;
    float bin_size_y = db.row_height;
    // num_bins_x = ceil((xh-xl)/bin_size_x);
    num_bins_y = ceil((db.yh - db.yl) / bin_size_y);

    // include both movable and fixed nodes
    std::vector<std::vector<int>> bin_cells(num_bins_x * num_bins_y);
    // distribute cells to bins
    distributeMovableAndFixedCells2BinsCPU(db, bin_cells, num_bins_x, num_bins_y, bin_size_x, bin_size_y);//@@

    std::vector<std::vector<AbacusCluster>> bin_clusters(num_bins_x * num_bins_y);
    for (unsigned int i = 0; i < bin_cells.size(); ++i) {
        bin_clusters[i].resize(bin_cells[i].size());
    }

    abacusLegalizeRowCPU(db, bin_cells, bin_clusters, num_bins_x, num_bins_y, bin_size_x, bin_size_y);
    // need to align nodes to sites
    // this also considers cell width which is not integral times of site_width
    for (auto const& cells : bin_cells) {
        float xxl = db.xl;
        for (auto node_id : cells) {
            if (node_id < db.num_movable_nodes) {
                db.x[node_id] = std::max(std::min(db.x[node_id], db.xh - db.node_size_x[node_id]), xxl);
                db.x[node_id] = floor((db.x[node_id] - xxl) / db.site_width) * db.site_width + xxl;
                xxl += ceil(db.node_size_x[node_id] / db.site_width) * db.site_width;
            } else if (node_id < db.num_nodes) {
                if (db.node_weight[node_id] == 0 || node_id >= db.num_movable_nodes) continue;  // FIXME
                xxl = ceil((db.x[node_id] + db.node_size_x[node_id] - db.xl) / db.site_width) * db.site_width + db.xl;
            }
        }
    }
}
}  // namespace dp