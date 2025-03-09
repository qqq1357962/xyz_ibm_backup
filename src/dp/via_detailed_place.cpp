#include "via_detailed_place.h"

struct boarder {
    // x/y min/max
    int64_t top[2][2];
    int64_t bot[2][2];
    int64_t opt[2][2];
    bool in_opt = false;
    int64_t pin_id;
};
map<int, struct boarder> net2bd;
// col            row        net_id
map<int64_t, map<int64_t, int64_t>> table;

void NodePosToPinPos(int num_pins,
                     torch::Tensor pin_id2node_id,
                     torch::Tensor pin_pos,
                     torch::Tensor pin_die,
                     torch::Tensor node_pos_all,
                     torch::Tensor node_die) {
    for (int i = 0; i < num_pins; ++i) {
        //        logger.info("i is %d", i);
        int64_t node_id = pin_id2node_id[i].item<long>();
        //    logger.info("node_id is %d", node_id);
        //    logger.info("before trans pos %d: (%.1f, %.1f), node_die: %d",
        //                node_id, node_pos_all[node_id][0].item().toFloat(), node_pos_all[node_id][1].item().toFloat(),
        //                node_die[node_id].item<int>());
        for (int c = 0; c < 2; ++c) {
            pin_pos[i][c] += node_pos_all[node_id][c];
            pin_die[i].data() = node_die[node_id];
        }
        //    logger.info("after trans pos %d: (%.1f, %.1f), pin_die: %d",
        //                i, pin_pos[i][0].item().toFloat(), pin_pos[i][1].item().toFloat(), pin_die[i].item<int>());
    }
}

void setNetBoarder(torch::Tensor pin_pos,
                   torch::Tensor pin_die,
                   torch::Tensor node_die,
                   torch::Tensor hyperedge_list,
                   torch::Tensor hyperedge_list_end) {
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_cells = node_die.size(0) - num_nets;

    for (int i = 0; i < num_nets; i++) {
        if (node_die[i + num_cells].item<int>() == -1) continue;
        for (int c = 0; c < 2; ++c) {
            priority_queue<int64_t> get_mean;
            int64_t start_idx = 0;
            if (i != 0) {
                start_idx = hyperedge_list_end[i - 1].item().toInt();
            }
            int64_t end_idx = hyperedge_list_end[i].item().toInt();

            int64_t bot_count = 0;
            int64_t top_count = 0;
            int64_t x_min_bot = 0;
            int64_t x_max_bot = 0;
            int64_t x_min_top = 0;
            int64_t x_max_top = 0;

            // calculate the boarder without bonding
            for (int64_t idx = start_idx; idx < end_idx - 1; idx++) {
                int64_t pin_id = hyperedge_list[idx].item<int>();

                if (pin_die[pin_id].item<int>() == 0) {
                    if (bot_count == 0) {
                        x_min_bot = pin_pos[pin_id][c].item<int>();
                        x_max_bot = pin_pos[pin_id][c].item<int>();
                    } else {
                        int64_t xx = pin_pos[pin_id][c].item<int>();
                        x_min_bot = min(xx, x_min_bot);
                        x_max_bot = max(xx, x_max_bot);
                    }
                    bot_count++;
                }
                if (pin_die[pin_id].item<int>() == 1) {
                    if (top_count == 0) {
                        x_min_top = pin_pos[pin_id][c].item<int>();
                        x_max_top = pin_pos[pin_id][c].item<int>();
                    } else {
                        int64_t xx = pin_pos[pin_id][c].item<int>();
                        x_min_top = min(xx, x_min_top);
                        x_max_top = max(xx, x_max_top);
                    }
                    top_count++;
                }
            }
            net2bd[i].top[c][0] = x_min_top;
            net2bd[i].top[c][1] = x_max_top;
            net2bd[i].bot[c][0] = x_min_bot;
            net2bd[i].bot[c][1] = x_max_bot;

            get_mean.push(x_min_top);
            get_mean.push(x_max_top);
            get_mean.push(x_min_bot);
            get_mean.push(x_max_bot);
            get_mean.pop();
            net2bd[i].opt[c][1] = get_mean.top();
            get_mean.pop();
            net2bd[i].opt[c][0] = get_mean.top();
        }
        int64_t end_idx = hyperedge_list_end[i].item().toInt();
        int64_t pin_id = hyperedge_list[end_idx - 1].item<int>();
        if (net2bd[i].opt[0][0] <= pin_pos[pin_id][0].item<int>() &&
            net2bd[i].opt[0][1] >= pin_pos[pin_id][0].item<int>() &&
            net2bd[i].opt[1][0] <= pin_pos[pin_id][1].item<int>() &&
            net2bd[i].opt[1][1] >= pin_pos[pin_id][1].item<int>())
            net2bd[i].in_opt = true;
        net2bd[i].pin_id = pin_id;
        table[pin_pos[pin_id][1].item().toInt()][pin_pos[pin_id][0].item().toInt()] = i;
    }
}

int64_t getMinHpwl(torch::Tensor pin_pos,
                   torch::Tensor pin_die,
                   torch::Tensor node_die,
                   torch::Tensor hyperedge_list,
                   torch::Tensor hyperedge_list_end) {
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_cells = node_die.size(0) - num_nets;
    int64_t min_hpwl_ext = 0, min_hpwl_bot = 0, min_hpwl_top = 0;

    for (int i = 0; i < num_nets; i++) {
        int64_t temp_hpwl = 0;
        for (int c = 0; c < 2; ++c) {
            int64_t start_idx = 0;
            if (i != 0) {
                start_idx = hyperedge_list_end[i - 1].item().toInt();
            }
            int64_t end_idx = hyperedge_list_end[i].item().toInt();

            int64_t bot_count = 0;
            int64_t top_count = 0;
            int64_t x_min_bot = 0;
            int64_t x_max_bot = 0;
            int64_t x_min_top = 0;
            int64_t x_max_top = 0;

            // calculate the boarder without bonding
            for (int64_t idx = start_idx; idx < end_idx - 1; idx++) {
                int64_t pin_id = hyperedge_list[idx].item<int>();

                if (pin_die[pin_id].item<int>() == 0) {
                    if (bot_count == 0) {
                        x_min_bot = pin_pos[pin_id][c].item<int>();
                        x_max_bot = pin_pos[pin_id][c].item<int>();
                    } else {
                        int64_t xx = pin_pos[pin_id][c].item<int>();
                        x_min_bot = min(xx, x_min_bot);
                        x_max_bot = max(xx, x_max_bot);
                    }
                    bot_count++;
                }
                if (pin_die[pin_id].item<int>() == 1) {
                    if (top_count == 0) {
                        x_min_top = pin_pos[pin_id][c].item<int>();
                        x_max_top = pin_pos[pin_id][c].item<int>();
                    } else {
                        int64_t xx = pin_pos[pin_id][c].item<int>();
                        x_min_top = min(xx, x_min_top);
                        x_max_top = max(xx, x_max_top);
                    }
                    top_count++;
                }
            }
            min_hpwl_bot += abs(x_min_bot - x_max_bot);
            min_hpwl_top += abs(x_min_top - x_max_top);
            if (node_die[i + num_cells].item<int>() == -1) continue;
            if (x_max_bot < x_min_top)
                min_hpwl_ext += abs(x_min_top - x_max_bot);
            else if (x_max_top < x_min_bot)
                min_hpwl_ext += abs(x_min_bot - x_max_top);

        }  // for one direction
    }      // for one net
    logger.info("Minimal HPWL (bot, top, ext, total): (%d, %d, %d, %d)",
                min_hpwl_bot,
                min_hpwl_top,
                min_hpwl_ext,
                min_hpwl_bot + min_hpwl_top + min_hpwl_ext);
    return min_hpwl_bot + min_hpwl_top + min_hpwl_ext;
}

int64_t getHpwl(torch::Tensor node_pos_legal, int src_net, int tar_net) {
    // calculate the hpwl of cells of src_net and bonding of tar_net
    int64_t hpwl = 0;
    int64_t x_min_bot = 0;
    int64_t x_max_bot = 0;
    int64_t x_min_top = 0;
    int64_t x_max_top = 0;

    for (int c = 0; c < 2; c++) {
        int64_t xx = node_pos_legal[tar_net][c].item<int>();
        x_min_bot = min(net2bd[src_net].bot[c][0], xx);
        x_max_bot = max(net2bd[src_net].bot[c][1], xx);
        x_min_top = min(net2bd[src_net].top[c][0], xx);
        x_max_top = max(net2bd[src_net].top[c][1], xx);
        hpwl += abs(x_max_bot - x_min_bot);
        hpwl += abs(x_max_top - x_min_top);
    }
    return hpwl;
}

int64_t SwapWithSpace(int64_t net_id,
                      int64_t opt_boarder[][2],
                      int64_t bonding_length[4],
                      torch::Tensor node_pos_legal,
                      vector<int64_t>& target_bonding) {
    for (int x = 0; x < 2; x++) {      // x
        for (int y = 0; y < 2; y++) {  // y
            // FIXME: set the boarder limit
            int64_t x_pos = net2bd[net_id].opt[0][x];
            int64_t y_pos = net2bd[net_id].opt[1][y];
            if (x_pos >= bonding_length[0] && x_pos + bonding_length[1] <= bonding_length[2] &&
                y_pos >= bonding_length[1] && y_pos + bonding_length[1] <= bonding_length[3]) {
                bool found_space = true;
                for (auto iter = target_bonding.begin(); iter != target_bonding.end(); iter++) {
                    auto tar_id = *iter;
                    if (abs(node_pos_legal[tar_id][0].item().toInt() - x_pos) >= bonding_length[0] &&
                        abs(node_pos_legal[tar_id][1].item().toInt() - y_pos) >= bonding_length[1]) {
                        continue;
                    } else {
                        found_space = false;
                        break;
                    }
                }
                if (!found_space)
                    continue;
                else {
                    int64_t before_hpwl = getHpwl(node_pos_legal, net_id, net_id);
                    // swap node_pos_legal
                    //                    logger.info("before swap node_pos_legal (%.1f, %.1f)",
                    //                                node_pos_legal[net_id][0].item().toFloat(),
                    //                                node_pos_legal[net_id][1].item().toFloat());
                    table[node_pos_legal[net_id][1].item().toInt()].erase(node_pos_legal[net_id][0].item().toInt());
                    node_pos_legal[net_id][0].data() = x_pos;
                    node_pos_legal[net_id][1].data() = y_pos;
                    //                    logger.info("after swap node_pos_legal (%.1f, %.1f)",
                    //                                node_pos_legal[net_id][0].item().toFloat(),
                    //                                node_pos_legal[net_id][1].item().toFloat());
                    table[y_pos][x_pos] = net_id;
                    net2bd[net_id].in_opt = true;
                    int64_t after_hpwl = getHpwl(node_pos_legal, net_id, net_id);
                    int64_t benefit = before_hpwl - after_hpwl;
                    //                    logger.info("move net_id: %d to (%d, %d) with benefit: %d (opt)", net_id,
                    //                    x_pos, y_pos, benefit);
                    return benefit;
                }
            }
        }
    }
    return 0;
}

void swapCells(int64_t i, int64_t j, torch::Tensor node_pos_legal) {
    // swap node_pos_legal and table
    // i and j are net_id

    //    logger.info("before swap node_pos_legal (%.1f, %.1f), (%.1f, %.1f) of via: %d and via: %d",
    //                node_pos_legal[i][0].item().toFloat(), node_pos_legal[i][1].item().toFloat(),
    //                node_pos_legal[j][0].item().toFloat(), node_pos_legal[j][1].item().toFloat(), i, j);
    auto node_i_x = node_pos_legal[i][0].item<int>();
    auto node_i_y = node_pos_legal[i][1].item<int>();
    auto node_j_x = node_pos_legal[j][0].item<int>();
    auto node_j_y = node_pos_legal[j][1].item<int>();
    node_pos_legal[i][0].data() = node_j_x;
    node_pos_legal[i][1].data() = node_j_y;
    node_pos_legal[j][0].data() = node_i_x;
    node_pos_legal[j][1].data() = node_i_y;
    //    logger.info("after swap node_pos_legal (%.1f, %.1f), (%.1f, %.1f)",
    //                node_pos_legal[i][0].item().toFloat(), node_pos_legal[i][1].item().toFloat(),
    //                node_pos_legal[j][0].item().toFloat(), node_pos_legal[j][1].item().toFloat());
    table[node_i_y][node_i_x] = j;
    table[node_j_y][node_j_x] = i;
}

bool isSpace(int64_t x_pos, int64_t y_pos, int64_t bonding_length[4]) {
    // if in die boundary
    if (x_pos < bonding_length[0] || x_pos + bonding_length[1] > bonding_length[2] || y_pos < bonding_length[1] ||
        y_pos + bonding_length[1] > bonding_length[3])
        return false;

    for (auto iter_y = table.begin(); iter_y != table.end(); iter_y++) {
        // FIXME: whether it is < or <=
        if (y_pos - bonding_length[1] < iter_y->first && iter_y->first < y_pos + bonding_length[1])
            for (auto iter_x = iter_y->second.begin(); iter_x != iter_y->second.end(); iter_x++)
                if (x_pos - bonding_length[0] < iter_x->first && iter_x->first < x_pos + bonding_length[0])
                    return false;
    }
    return true;
}

int64_t hpwlDrivenSwap(torch::Tensor node_pos_legal, int64_t bonding_length[4]) {
    int64_t total_benefit = 0;
    int64_t thres = 200;
    int64_t found_num = 0;
    int64_t failed_num = 0;
    for (auto iter = net2bd.begin(); iter != net2bd.end(); iter++) {
        if (iter->second.in_opt) continue;
        auto net_id = iter->first;
        int64_t x_pos = node_pos_legal[net_id][0].item().toInt();
        int64_t y_pos = node_pos_legal[net_id][1].item().toInt();
        // calculate the min manhattan distance from the bonding to its opt region
        int64_t mid_opt_x = (net2bd[net_id].opt[0][0] + net2bd[net_id].opt[0][1]) / 2;
        int64_t mid_opt_y = (net2bd[net_id].opt[1][0] + net2bd[net_id].opt[1][1]) / 2;
        int64_t min_dist_x = mid_opt_x - x_pos;
        int64_t min_dist_y = mid_opt_y - y_pos;
        //        int64_t min_dist_x = (abs(net2bd[net_id].opt[0][0]-x_pos)>abs(net2bd[net_id].opt[0][1]-x_pos)) ?
        //                (net2bd[net_id].opt[0][0]-x_pos) : (net2bd[net_id].opt[0][1]-x_pos);
        //        int64_t min_dist_y = (abs(net2bd[net_id].opt[1][0]-y_pos)>abs(net2bd[net_id].opt[1][1]-y_pos)) ?
        //                             (net2bd[net_id].opt[1][0]-y_pos) : (net2bd[net_id].opt[1][1]-y_pos);
        int64_t min_dist = abs(min_dist_x) + abs(min_dist_y);
        if (min_dist <= (bonding_length[2] + bonding_length[3]) / thres) continue;
        found_num++;
        int64_t per = 0;
        for (per = 0; per <= 40; per++) {
            int64_t x_temp = x_pos + min_dist_x * (100 - per) / 100;
            int64_t y_temp = y_pos + min_dist_y * (100 - per) / 100;
            if (isSpace(x_temp, y_temp, bonding_length)) {
                // move bonding to valid space
                // from (x_pos, y_pos) to (x_temp, y_temp)
                int64_t before_hpwl = getHpwl(node_pos_legal, net_id, net_id);
                table[y_pos].erase(x_pos);
                node_pos_legal[net_id][0].data() = x_temp;
                node_pos_legal[net_id][1].data() = y_temp;
                table[y_temp][x_temp] = net_id;
                if (net2bd[net_id].opt[0][0] <= x_temp && net2bd[net_id].opt[0][1] >= x_temp &&
                    net2bd[net_id].opt[1][0] <= y_temp && net2bd[net_id].opt[1][1] >= y_temp)
                    net2bd[net_id].in_opt = true;
                int64_t after_hpwl = getHpwl(node_pos_legal, net_id, net_id);
                int64_t benefit = before_hpwl - after_hpwl;
                total_benefit += benefit;
                //                logger.info("move- net_id: %d to (%d, %d) with benefit: %d", net_id, x_temp, y_temp,
                //                benefit);
                break;
            }
            x_temp = x_pos + min_dist_x * (100 + per) / 100;
            y_temp = y_pos + min_dist_y * (100 + per) / 100;
            if (isSpace(x_temp, y_temp, bonding_length)) {
                // move bonding to valid space
                // from (x_pos, y_pos) to (x_temp, y_temp)
                int64_t before_hpwl = getHpwl(node_pos_legal, net_id, net_id);
                table[y_pos].erase(x_pos);
                node_pos_legal[net_id][0].data() = x_temp;
                node_pos_legal[net_id][1].data() = y_temp;
                table[y_temp][x_temp] = net_id;
                if (net2bd[net_id].opt[0][0] <= x_temp && net2bd[net_id].opt[0][1] >= x_temp &&
                    net2bd[net_id].opt[1][0] <= y_temp && net2bd[net_id].opt[1][1] >= y_temp)
                    net2bd[net_id].in_opt = true;
                int64_t after_hpwl = getHpwl(node_pos_legal, net_id, net_id);
                int64_t benefit = before_hpwl - after_hpwl;
                total_benefit += benefit;
                //                logger.info("move+ net_id: %d to (%d, %d) with benefit: %d", net_id, x_temp, y_temp,
                //                benefit);
                break;
            }
        }
        if (per > 40) failed_num++;
    }
    logger.info("found_num: %d, failed_num: %d", found_num, failed_num);
    return total_benefit;
}

int64_t windowSwap(torch::Tensor node_pos_legal) {
    int64_t total_benefit = 0;
    for (auto iter_y = table.begin(); iter_y != table.end(); iter_y++) {
        int64_t col = iter_y->first;
        int64_t size = table[col].size();
        if (size < 3) continue;
        auto iter_x = table[col].begin();
        for (int64_t cnt = 0; cnt < size - 2; cnt++) {
            int64_t i = iter_x->second;
            iter_x++;
            int64_t j = iter_x->second;
            iter_x++;
            int64_t k = iter_x->second;
            iter_x--;
            int64_t temp_hpwl[6];
            temp_hpwl[0] =
                getHpwl(node_pos_legal, i, i) + getHpwl(node_pos_legal, j, j) + getHpwl(node_pos_legal, k, k);
            temp_hpwl[1] =
                getHpwl(node_pos_legal, i, i) + getHpwl(node_pos_legal, j, k) + getHpwl(node_pos_legal, k, j);
            temp_hpwl[2] =
                getHpwl(node_pos_legal, i, j) + getHpwl(node_pos_legal, j, i) + getHpwl(node_pos_legal, k, k);
            temp_hpwl[3] =
                getHpwl(node_pos_legal, i, k) + getHpwl(node_pos_legal, j, i) + getHpwl(node_pos_legal, k, j);
            temp_hpwl[4] =
                getHpwl(node_pos_legal, i, j) + getHpwl(node_pos_legal, j, k) + getHpwl(node_pos_legal, k, i);
            temp_hpwl[5] =
                getHpwl(node_pos_legal, i, k) + getHpwl(node_pos_legal, j, j) + getHpwl(node_pos_legal, k, i);
            int64_t min_hpwl = temp_hpwl[0], index = 0;
            for (int r = 1; r < 6; r++)
                if (temp_hpwl[r] < min_hpwl) {
                    min_hpwl = temp_hpwl[r];
                    index = r;
                }
            if (index == 1)
                swapCells(j, k, node_pos_legal);
            else if (index == 2)
                swapCells(i, j, node_pos_legal);
            else if (index == 3) {
                swapCells(i, k, node_pos_legal);
                swapCells(j, k, node_pos_legal);
            } else if (index == 4) {
                swapCells(i, j, node_pos_legal);
                swapCells(j, k, node_pos_legal);
            } else if (index == 5)
                swapCells(i, k, node_pos_legal);
            if (index != 0) {
                total_benefit += temp_hpwl[0] - min_hpwl;
                // logger.info("windowSwap with benefit: %d", temp_hpwl[0] - min_hpwl);
            }
        }
    }
    return total_benefit;
}

void viaDetailedPlace(const torch::Tensor node_pos,
                      const torch::Tensor node_size,
                      torch::Tensor node_pos_legal,
                      torch::Tensor node_weight,
                      torch::Tensor die_info,
                      torch::Tensor numRows,
                      torch::Tensor rowHeights,
                      int num_nodes,
                      torch::Tensor node_pos_all,
                      torch::Tensor node_die,
                      NodeData& data) {
    torch::NoGradGuard no_grad;
    /* row info */
    int num_visible_node = node_weight.sum().item<int>();
    int num_invisible_node = num_nodes - num_visible_node;

    // node_pos = via_node_pos + node(cell)_pos
    torch::Tensor pin_id2node_id = data.pin_id2node_id;
    torch::Tensor hyperedge_list = data.hyperedge_list;
    torch::Tensor hyperedge_list_end = data.hyperedge_list_end;
    torch::Tensor pin_rel_cpos = data.pin_rel_cpos;

    //    const auto num_nodes = node_pos.size(0);
    const auto num_pins = pin_id2node_id.size(0);
    const auto num_nets = hyperedge_list_end.size(0);
    const auto num_cells = node_die.size(0) - num_nets;
    const auto num_channels = 2;  // x, y

    auto pin_pos = pin_rel_cpos.clone();
    auto pin_die = torch::zeros(num_pins, torch::dtype(torch::kInt));
    // bonding width, height  die width, height
    int64_t bonding_length[4] = {data.bondingInfo[0].item().toInt() + data.bondingInfo[2].item().toInt(),
                                 data.bondingInfo[1].item().toInt() + data.bondingInfo[2].item().toInt(),
                                 data.die_info[1].item().toInt(),
                                 data.die_info[3].item().toInt()};

    //    print info
    /*    logger.info("here is via_node_pos_legal");
        for(int i=0;i<node_pos_legal.size(0);i++)
            logger.info("%d: (%.1f: %.1f)", i, node_pos_legal[i][0].item().toFloat(),
       node_pos_legal[i][1].item().toFloat()); logger.info("here is pin_id2node_id"); for(int i=0;i<num_pins;i++)
            logger.info("%d: %d", i, pin_id2node_id[i].item().toInt());
        logger.info("here is hyperedge_list");
        for(int i=0;i<hyperedge_list.size(0);i++)
            logger.info("%d: %d", i, hyperedge_list[i].item().toInt());
        logger.info("here is hyperedge_list_end");
        for(int i=0;i<num_nets;i++)
            logger.info("%d: %d", i, hyperedge_list_end[i].item().toInt());
        logger.info("here is node_die");
        for(int i=0;i<node_die.size(0);i++)
            logger.info("%d: %d", i, node_die[i].item().toInt()); */

    NodePosToPinPos(num_pins, pin_id2node_id, pin_pos, pin_die, node_pos_all, node_die);
    setNetBoarder(pin_pos, pin_die, node_die, hyperedge_list, hyperedge_list_end);
    // int64_t min_hpwl = getMinHpwl(pin_pos, pin_die, node_die, hyperedge_list, hyperedge_list_end);

    bool if_improve = false;
    int round = 0;
    int max_round = 10;
    int64_t improvement = 0;
    while (round < max_round) {
        if_improve = false;
        improvement = 0;
        round++;

        for (auto iter = net2bd.begin(); iter != net2bd.end(); iter++) {
            if (iter->second.in_opt) continue;
            auto net_id = iter->first;

            int64_t opt_boarder[2][2];
            opt_boarder[0][0] = iter->second.opt[0][0] - bonding_length[0];
            opt_boarder[0][1] = iter->second.opt[0][1] + bonding_length[0];
            opt_boarder[1][0] = iter->second.opt[1][0] - bonding_length[1];
            opt_boarder[1][1] = iter->second.opt[1][1] + bonding_length[1];
            vector<int64_t> target_bonding;  // net_id of bonding
            // find the target bonding to swap
            for (auto iter_y = table.begin(); iter_y != table.end(); iter_y++) {
                // FIXME: whether it is < or <=
                if (opt_boarder[1][0] < iter_y->first && iter_y->first < opt_boarder[1][1])
                    for (auto iter_x = iter_y->second.begin(); iter_x != iter_y->second.end(); iter_x++)
                        if (opt_boarder[0][0] < iter_x->first && iter_x->first < opt_boarder[0][1] &&
                            iter_x->second != net_id)
                            target_bonding.push_back(iter_x->second);
            }
            // swap bonding with space
            auto benefit = SwapWithSpace(iter->first, opt_boarder, bonding_length, node_pos_legal, target_bonding);
            if (benefit > 0) {
                if_improve = true;
                improvement += benefit;
                continue;
            }
            // calculate the benefit by swap
            int64_t max_benefit = 0;
            int64_t max_tar = 0;
            for (auto iter_target = target_bonding.begin(); iter_target != target_bonding.end(); iter_target++) {
                auto src = iter->first;
                auto tar = *iter_target;
                int64_t before_hpwl = getHpwl(node_pos_legal, src, src) + getHpwl(node_pos_legal, tar, tar);
                int64_t after_hpwl = getHpwl(node_pos_legal, src, tar) + getHpwl(node_pos_legal, tar, src);
                benefit = before_hpwl - after_hpwl;
                if (benefit > max_benefit) {
                    max_benefit = benefit;
                    max_tar = tar;
                }
            }
            // swap
            auto i = iter->first;
            auto j = max_tar;
            if (max_benefit > 0) {
                swapCells(i, j, node_pos_legal);
                if_improve = true;
                improvement += max_benefit;
                //                logger.info("swapping bonding %d and %d with benefit: %d", i, j, max_benefit);
            }
        }

        auto benefit = hpwlDrivenSwap(node_pos_legal, bonding_length);
        if (benefit > 0) {
            if_improve = true;
            improvement += benefit;
        }

        benefit = windowSwap(node_pos_legal);
        if (benefit > 0) {
            if_improve = true;
            improvement += benefit;
        }

        logger.info("-----round: %d, improvement: %d -----", round, improvement);
        int64_t opt_num = 0;
        for (auto iter = net2bd.begin(); iter != net2bd.end(); iter++) {
            if (iter->second.in_opt == true) opt_num++;
        }
        logger.info("total net num: %d, opt_num: %d", net2bd.size(), opt_num);

        if (!if_improve) break;
    }

}  // END MODULE