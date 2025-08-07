#include "legality_check.h"

namespace dp {

bool boundaryCheck(const DetailedPlaceData& db, float scale_factor, int num_movable_nodes) {
    // use scale factor to control the precision
    float precision = (scale_factor == 1.0) ? 1e-6 : scale_factor * 0.1;
    bool legal_flag = true;
    // check node within boundary
    for (int i = 0; i < num_movable_nodes; ++i) {
        if (db.node_weight[i] == 0) continue;
        float node_xl = db.x[i];
        float node_yl = db.y[i];
        float node_xh = node_xl + db.node_size_x[i];
        float node_yh = node_yl + db.node_size_y[i];
        if (node_xl + precision < db.xl || node_xh > db.xh + precision || node_yl + precision < db.yl ||
            node_yh > db.yh + precision) {
            logger.warning("node %d (%g, %g, %g, %g) out of boundary", i, node_xl, node_yl, node_xh, node_yh);
            legal_flag = false;
        }
    }
    return legal_flag;
}

bool siteAlignmentCheck(const DetailedPlaceData& db, float scale_factor, int num_movable_nodes) {
    // use scale factor to control the precision
    float precision = (scale_factor == 1.0) ? 1e-4 : scale_factor * 0.1;
    bool legal_flag = true;
    // check row and site alignment
    for (int i = 0; i < num_movable_nodes; ++i) {
        if (db.node_weight[i] == 0) continue;
        float node_xl = db.x[i];
        float node_yl = db.y[i];

        // FIXME: float divide
        float row_id_f =
            round((node_yl - db.yl) * db.site_width_safe_divide) / round(db.site_width_safe_divide * db.row_height);
        // int row_id = floorDiv(round((node_yl - db.yl) * db.site_width_safe_divide),
        //                       round(db.site_width_safe_divide * db.row_height));  // FIXME: float divide
        int row_id = floorDivRound(node_yl - db.yl, db.row_height, db.site_width_safe_divide);
        float row_yl = db.yl + db.row_height * row_id;
        float row_yh = row_yl + db.row_height;

        if (std::abs(row_id_f - row_id) > precision) {
            logger.error(

                "node %d (%g, %g) failed to align to row %d (%g, %g), gap %g",
                i,
                node_xl,
                node_yl,
                row_id,
                row_yl,
                row_yh,
                std::abs(node_yl - row_yl));
            legal_flag = false;
        }

        float site_id_f = (node_xl - db.xl) / db.site_width;
        // int site_id = floorDiv(node_xl - db.xl, db.site_width);
        int site_id = floorDivRound(node_xl - db.xl, db.site_width, db.site_width_safe_divide);
        float site_xl = db.xl + db.site_width * site_id;
        float site_xh = site_xl + db.site_width;
        // if (std::abs(site_id_f - site_id) > precision) {
        if (std::abs(double(node_xl - site_xl)) > double(db.site_width) * double(precision)) {
            cout << node_xl << endl;
            cout << site_xl << endl;
            cout << db.site_width << endl;
            cout << precision << endl;
            cout << std::abs(double(node_xl - site_xl)) << endl;
            cout << double(db.site_width) * double(precision) << endl;
            logger.error("node %d (%g, %g) failed to align to row %d (%g, %g) and site %d (%g, %g)(%f, %f)",
                         i,
                         node_xl,
                         node_yl,
                         row_id,
                         row_yl,
                         row_yh,
                         site_id,
                         site_xl,
                         site_xh,
                         site_id_f,
                         db.xl);
            legal_flag = false;
        }
    }

    return legal_flag;
}

bool overlapCheck(const DetailedPlaceData& db, float scale_factor, int num_nodes, int num_movable_nodes) {
    bool legal_flag = true;
    int num_rows = ceilDiv(db.yh - db.yl, db.row_height);
    assert(num_rows > 0);
    std::vector<std::vector<int>> row_nodes(num_rows);

    // general to node and fixed boxes
    auto getXL = [&](int id) { return db.x[id]; };
    auto getYL = [&](int id) { return db.y[id]; };
    auto getXH = [&](int id) { return db.x[id] + db.node_size_x[id]; };
    auto getYH = [&](int id) { return db.y[id] + db.node_size_y[id]; };
    // add a box to row
    auto addBox2Row = [&](int id, float bxl, float byl, float bxh, float byh) {
        // FIXME: float divide
        // int row_idxl = floorDiv(byl - db.yl, db.row_height);
        int row_idxl = floorDivRound(byl - db.yl, db.row_height, db.site_width_safe_divide);
        // int row_idxh = ceilDiv(byh - db.yl, db.row_height);
        int row_idxh = ceilDivRound(byh - db.yl, db.row_height, db.site_width_safe_divide);
        row_idxl = std::max(row_idxl, 0);
        row_idxh = std::min(row_idxh, num_rows);

        for (int row_id = row_idxl; row_id < row_idxh; ++row_id) {
            float row_yl = db.yl + row_id * db.row_height;
            float row_yh = row_yl + db.row_height;

            if (byl < row_yh && byh > row_yl)  // overlap with row
            {
                row_nodes[row_id].push_back(id);
            }
        }
    };
    // distribute movable cells to rows
    for (int i = 0; i < num_nodes; ++i) {
        if (db.node_weight[i] == 0 || i >= num_movable_nodes) continue;  // FIXME
        float node_xl = db.x[i];
        float node_yl = db.y[i];
        float node_xh = node_xl + db.node_size_x[i];
        float node_yh = node_yl + db.node_size_y[i];

        addBox2Row(i, node_xl, node_yl, node_xh, node_yh);
    }

    // sort cells within rows
    for (int i = 0; i < num_rows; ++i) {
        auto& nodes_in_row = row_nodes.at(i);
        // using left edge
        std::sort(nodes_in_row.begin(), nodes_in_row.end(), [&](int node_id1, int node_id2) {
            float x1 = getXL(node_id1);
            float x2 = getXL(node_id2);
            return x1 < x2 || (x1 == x2 && (node_id1 < node_id2));
        });
        // After sorting by left edge,
        // there is a special case for fixed cells where
        // one fixed cell is completely within another in a row.
        // This will cause failure to detect some overlaps.
        // We need to remove the "small" fixed cell that is inside another.
        if (!nodes_in_row.empty()) {
            std::vector<int> tmp_nodes;
            tmp_nodes.reserve(nodes_in_row.size());
            tmp_nodes.push_back(nodes_in_row.front());
            for (int j = 1, je = nodes_in_row.size(); j < je; ++j) {
                int node_id1 = nodes_in_row.at(j - 1);
                int node_id2 = nodes_in_row.at(j);
                // two fixed cells
                if (node_id1 >= num_movable_nodes && node_id2 >= num_movable_nodes) {
                    float xh1 = getXH(node_id1);
                    float xh2 = getXH(node_id2);
                    if (xh1 < xh2) {
                        tmp_nodes.push_back(node_id2);
                    }
                } else {
                    tmp_nodes.push_back(node_id2);
                }
            }
            nodes_in_row.swap(tmp_nodes);
        }
    }

    // check overlap
    // use scale factor to control the precision
    auto scaleBack2Integer = [&](float value) {
        return (scale_factor == 1.0) ? value : std::round(value / scale_factor);
    };
    for (int i = 0; i < num_rows; ++i) {
        for (unsigned int j = 0; j < row_nodes.at(i).size(); ++j) {
            if (j > 0) {
                int node_id = row_nodes[i][j];
                int prev_node_id = row_nodes[i][j - 1];

                if (node_id < num_movable_nodes || prev_node_id < num_movable_nodes) {
                    // ignore two fixed nodes
                    float prev_xl = getXL(prev_node_id);
                    float prev_yl = getYL(prev_node_id);
                    float prev_xh = getXH(prev_node_id);
                    float prev_yh = getYH(prev_node_id);
                    float cur_xl = getXL(node_id);
                    float cur_yl = getYL(node_id);
                    float cur_xh = getXH(node_id);
                    float cur_yh = getYH(node_id);
                    // detect overlap
                    if (scaleBack2Integer(prev_xh) > scaleBack2Integer(cur_xl)) {
                        logger.error(
                            "row %d (%g, %g), overlap node %d (%g, %g, %g, %g) with "
                            "node %d (%g, %g, %g, %g), gap %g",
                            i,
                            db.yl + i * db.row_height,
                            db.yl + (i + 1) * db.row_height,
                            prev_node_id,
                            prev_xl,
                            prev_yl,
                            prev_xh,
                            prev_yh,
                            node_id,
                            cur_xl,
                            cur_yl,
                            cur_xh,
                            cur_yh,
                            prev_xh - cur_xl);
                        legal_flag = false;
                    }
                }
            }
        }
    }

    return legal_flag;
}

bool legalityCheckKernelCPU(const DetailedPlaceData& db, float scale_factor, int num_nodes, int num_movable_nodes) {
    bool legal_flag = true;
    int num_rows = ceil((db.yh - db.yl) / db.row_height);
    assert(num_rows > 0);
    fflush(stdout);
    std::vector<std::vector<int>> row_nodes(num_rows);

    // check node within boundary
    if (!boundaryCheck(db, scale_factor, num_movable_nodes)) {
        legal_flag = false;
    }

    // check row and site alignment
    if (!siteAlignmentCheck(db, scale_factor, num_movable_nodes)) {
        legal_flag = false;
    }

    if (!overlapCheck(db, scale_factor, num_nodes, num_movable_nodes)) {
        legal_flag = false;
    }

    return legal_flag;
}

bool legalityCheck_main(const DetailedPlaceData& db) {
    return legalityCheckKernelCPU(db, 1.0, db.num_nodes, db.num_movable_nodes);
}

}  // namespace dp