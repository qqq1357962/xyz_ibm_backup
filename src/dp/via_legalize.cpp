#include "via_legalize.h"

struct coordPoints {
    int id;
    int x;
};
struct pos {
    int x;
    int y;
};
bool via_sort_asc(coordPoints a, coordPoints b) {
    if (a.x < b.x) {
        return true;
    }
    return false;
}
void insert(vector<coordPoints> &array, coordPoints pts) {
    std::vector<coordPoints>::iterator it = std::lower_bound(array.begin(), array.end(), pts, via_sort_asc);
    array.insert(it, pts);
}
void viaLegalization(const torch::Tensor node_pos,
                     const torch::Tensor node_size,
                     torch::Tensor node_pos_legal,
                     torch::Tensor node_weight,
                     torch::Tensor die_info,
                     torch::Tensor numRows,
                     torch::Tensor rowHeights,
                     int num_nodes) {
    int num_visible_node = node_weight.sum().item<int>();
    vector<coordPoints> via_array;
    int xl = die_info[0].item<int>();
    int xh = die_info[1].item<int>();
    int yl = die_info[2].item<int>();
    int yh = die_info[3].item<int>();
    int count = 0;
    for (int no = 0; no < num_nodes; no++) {
        if (node_weight[no].item<int>()) {
            coordPoints via;
            via.id = no;
            via.x = node_pos_legal[no][0].item<int>();
            // count++;
            via_array.push_back(via);
        }
    }

    std::sort(via_array.begin(), via_array.end(), via_sort_asc);
    vector<coordPoints> placed_via_array;
    // logger.info("Hello  %d", via_array.size());
    for (int no = 0; no < via_array.size(); no++) {
        int via_x = node_pos_legal[via_array[no].id][0].item<int>();
        int via_y = node_pos_legal[via_array[no].id][1].item<int>();
        int size_x = node_size[via_array[no].id][0].item<int>();
        int size_y = node_size[via_array[no].id][1].item<int>();
        if (via_x + size_x / 2 >= xh) {
            via_x = xh - size_x / 2 - 1;
        } else if (via_x - size_x / 2 < xl) {
            via_x = xl + size_x / 2;
        }
        if (via_y + size_y / 2 >= yh) {
            via_y = yh - size_y / 2 - 1;
        } else if (via_y - size_y / 2 < yl) {
            via_y = yl + size_y / 2;
        }
        int up = 0;
        int last = 0;
        int overlap = 1;
        // logger.info("placing %d", via_array[no].id);
        std::map<std::string, int> visited;

        std::stack<pos> visited_stack;
        pos original_pos;
        original_pos.x = via_x;
        original_pos.y = via_y;
        visited_stack.push(original_pos);
        std::string origin_str = std::to_string(via_x) + " " + std::to_string(via_y);
        visited[origin_str] = 1;

        while (overlap == 1) {
            coordPoints tmp;
            tmp.id = -1;
            tmp.x = via_x - size_x;
            int start = std::lower_bound(placed_via_array.begin(), placed_via_array.end(), tmp, via_sort_asc) -
                        placed_via_array.begin();
            tmp.x = via_x + size_x;
            int end = std::upper_bound(placed_via_array.begin(), placed_via_array.end(), tmp, via_sort_asc) -
                      placed_via_array.begin() - 1;

            int found = 0;
            // logger.info("placed %d %d %d", placed_via_array.size(), start, end);

            for (int no = start; no <= end; no++) {
                int via2_x = node_pos_legal[placed_via_array[no].id][0].item<int>();
                int via2_y = node_pos_legal[placed_via_array[no].id][1].item<int>();

                if (abs(via2_y - via_y) >= size_y || abs(via2_x - via_x) >= size_x) {
                    continue;
                }
                found = 1;

                int move_index = -1;
                int move_dis = -1;

                int direction[4][2];
                for (int i = 0; i < 4; i++) {
                    direction[i][0] = via_x;
                    direction[i][1] = via_y;
                }
                direction[0][1] = via2_y - size_y;
                direction[1][0] = via2_x - size_x;
                direction[2][1] = via2_y + size_y;
                direction[3][0] = via2_x + size_x;
                // logger.info("orginal     %d,%d", via_x, via_y);

                for (int i = 0; i < 4; i++) {
                    if (direction[i][0] >= xl + size_x / 2 && direction[i][0] < xh - size_x / 2 &&
                        direction[i][1] >= yl + size_y / 2 && direction[i][1] < yh - size_y / 2) {
                        int displacment = abs(direction[i][0] - via_x) + abs(direction[i][1] - via_y);
                        std::string s = std::to_string(direction[i][0]) + " " + std::to_string(direction[i][1]);
                        // logger.info("%s     %d,%d   %d,%d    %d", s.c_str(), visited.size(), via2_x, size_x, via2_y,
                        // size_y);
                        if (visited.count(s) == 0) {
                            if (move_dis == -1 || displacment < move_dis) {
                                move_dis = displacment;
                                move_index = i;
                            }
                        }
                    }
                }
                // logger.info()
                if (move_dis >= 0) {
                    via_x = direction[move_index][0];
                    via_y = direction[move_index][1];
                    std::string s =
                        std::to_string(direction[move_index][0]) + " " + std::to_string(direction[move_index][1]);
                    pos tmp_pos;
                    tmp_pos.x = direction[move_index][0];
                    tmp_pos.y = direction[move_index][1];
                    visited_stack.push(tmp_pos);
                    visited[s] = 1;
                } else {
                    pos tmp_pos = visited_stack.top();
                    visited_stack.pop();
                    via_x = tmp_pos.x;
                    via_y = tmp_pos.y;
                }
            }

            if (found == 0) {
                overlap = 0;
            }
        }
        node_pos_legal[via_array[no].id][0] = via_x;
        node_pos_legal[via_array[no].id][1] = via_y;
        coordPoints insertPoint;
        insertPoint.id = via_array[no].id;
        insertPoint.x = via_x;
        insert(placed_via_array, insertPoint);
        count++;
        // logger.info("place %d",count);
    }
}