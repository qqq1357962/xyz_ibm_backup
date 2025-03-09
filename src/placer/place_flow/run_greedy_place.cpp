#include "../run_placement.h"

void run_greedy_place(NodeData& data) {
    auto node_pos = data.node_pos.detach().clone();
    // die info
    float bound_x = data.__ori_die_hx__;
    float bound_y = data.__ori_die_hy__;
    // top_height.resize(data.__ori_die_hx__);
    torch::Tensor node_area_top = data.node_area_top;
    float yl = data.die_info[2].item<float>();

    // status variants
    vector<int> top_height;
    vector<int> is_placed;
    is_placed.resize(data.cell_mov_rhs);

    // auto [sorted_tensor, indices] = data.node_area_top.slice(0, 0, data.cell_mov_rhs).sort();
    auto [sorted_tensor, indices] = data.node_size_top.index({torch::indexing::Slice(), 0}).sort();
    torch::Tensor node_sizes = torch::cat({data.node_size_bot.unsqueeze(0), data.node_size_top.unsqueeze(0)}, 0);

    auto node_pos_at = node_pos.accessor<float, 2>();
    for (int layer = 1; layer >= 0; layer--) {
        float startx = 0;
        float starty = 0;
        float targetx = 0;
        float targety = 0;
        float current_right = 0;
        float row_height = data.rowHeights[layer].item<float>();
        float row_num = data.numRows[layer].item<float>();
        float layer_area = 0;
        float area_bound = data.max_mov_cell_areas[layer].item<float>();
        for (int i = 0; i < data.cell_mov_rhs; i++) {
            int node_id = 0;
            // if(i%2==0)
            // {
            //     node_id = indices[data.cell_mov_rhs-1-i/2].item<int>();
            // }else{
            //     node_id = indices[i/2].item<int>();
            // }
            node_id = indices[data.cell_mov_rhs - 1 - i].item<int>();
            // int node_id = i;
            if (is_placed[node_id] == 1) {
                continue;
            }
            float w = node_sizes[layer][node_id][0].item<float>();
            float h = node_sizes[layer][node_id][1].item<float>();
            if (layer_area + w * h > area_bound) {
                continue;
            }
            // judge macro/std cell and decide its position
            if (data.macro_mask[node_id].item<int>() == 1) {
                targety = starty;
                bound_y = data.__ori_die_hy__;
            } else {
                // targety = start
                int row_id = ceil((starty - yl) / row_height);
                float row_yl = yl + row_height * row_id;
                float row_yh = row_yl + row_height;
                targety = row_yl;
                bound_y = row_num * row_height;
            }
            // place this cell
            if (targety + h <= bound_y && targetx + w <= bound_x) {
                node_pos_at[node_id][0] = targetx + w / 2;
                node_pos_at[node_id][1] = targety + h / 2;
                starty = targety + h;
                current_right = max(targetx + w, current_right);
                is_placed[node_id] = 1;
                data.node_die[node_id] = layer;
                layer_area += w * h;
            } else if (targety + h > bound_y && current_right + w <= bound_x) {
                targetx = current_right;
                if (data.macro_mask[node_id].item<int>() == 1) {
                    targety = 0;
                } else {
                    targety = yl;
                }
                node_pos_at[node_id][0] = targetx + w / 2;
                node_pos_at[node_id][1] = targety + h / 2;
                starty = targety + h;
                is_placed[node_id] = 1;
                data.node_die[node_id] = layer;
                current_right = max(targetx + w, current_right);
                layer_area += w * h;
            } else {
                is_placed[node_id] = 0;
            }
        }
    }
    data.node_pos = node_pos.detach().clone();
}

void run_greedy_place_for_fp(NodeData& data) {
    auto node_pos = data.node_pos.detach().clone();
    // die info
    float bound_x = data.__ori_die_hx__;
    float bound_y = data.__ori_die_hy__;
    // top_height.resize(data.__ori_die_hx__);
    torch::Tensor node_area_top = data.node_area_top;
    float yl = data.die_info[2].item<float>();

    // status variants
    vector<int> top_height;
    vector<int> is_placed;
    is_placed.resize(data.cell_mov_rhs);

    // auto [sorted_tensor, indices] = data.node_area_top.slice(0, 0, data.cell_mov_rhs).sort();
    auto [sorted_tensor, indices] = data.node_size_top.index({torch::indexing::Slice(), 0}).sort();
    torch::Tensor node_sizes = torch::cat({data.node_size_bot.unsqueeze(0), data.node_size_top.unsqueeze(0)}, 0);

    auto node_pos_at = node_pos.accessor<float, 2>();
    for (int layer = 1; layer >= 0; layer--) {
        float startx = 0;
        float starty = 0;
        float targetx = 0;
        float targety = 0;
        float current_right = 0;
        float row_height = data.rowHeights[layer].item<float>();
        float row_num = data.numRows[layer].item<float>();
        float layer_area = 0;
        float area_bound = data.max_mov_cell_areas[layer].item<float>();
        for (int i = 0; i < data.cell_mov_rhs; i++) {
            int node_id = 0;
            // if(i%2==0)
            // {
            //     node_id = indices[data.cell_mov_rhs-1-i/2].item<int>();
            // }else{
            //     node_id = indices[i/2].item<int>();
            // }
            node_id = indices[data.cell_mov_rhs - 1 - i].item<int>();
            if(data.node_die[node_id].item<int>()!=layer)
            {
                continue;
            }
            if(data.macro_mask[node_id].item<int>()==0)
            {
                continue;
            }
            // int node_id = i;
            if (is_placed[node_id] == 1) {
                continue;
            }
            float w = node_sizes[layer][node_id][0].item<float>();
            float h = node_sizes[layer][node_id][1].item<float>();
            if (layer_area + w * h > area_bound) {
                continue;
            }
            // judge macro/std cell and decide its position
            if (data.macro_mask[node_id].item<int>() == 1) {
                targety = starty;
                bound_y = data.__ori_die_hy__;
            } else {
                // targety = start
                int row_id = ceil((starty - yl) / row_height);
                float row_yl = yl + row_height * row_id;
                float row_yh = row_yl + row_height;
                targety = row_yl;
                bound_y = row_num * row_height;
            }
            // place this cell
            if (targety + h <= bound_y && targetx + w <= bound_x) {
                node_pos_at[node_id][0] = targetx + w / 2;
                node_pos_at[node_id][1] = targety + h / 2;
                starty = targety + h;
                current_right = max(targetx + w, current_right);
                is_placed[node_id] = 1;
                data.node_die[node_id] = layer;
                layer_area += w * h;
            } else if (targety + h > bound_y && current_right + w <= bound_x) {
                targetx = current_right;
                if (data.macro_mask[node_id].item<int>() == 1) {
                    targety = 0;
                } else {
                    targety = yl;
                }
                node_pos_at[node_id][0] = targetx + w / 2;
                node_pos_at[node_id][1] = targety + h / 2;
                starty = targety + h;
                is_placed[node_id] = 1;
                data.node_die[node_id] = layer;
                current_right = max(targetx + w, current_right);
                layer_area += w * h;
            } else {
                is_placed[node_id] = 0;
            }
        }
    }
    data.node_pos = node_pos.detach().clone();
}