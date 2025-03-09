#include "ViaData.h"
#include <lemon/list_graph.h>
#include <lemon/matching.h>

using namespace lemon;

typedef ListGraph UGraph;
typedef UGraph::EdgeMap<double> DistMap;
typedef MaxWeightedPerfectMatching<UGraph,DistMap> MWPM;
typedef MaxWeightedMatching<UGraph,DistMap> MWM;

ViaLegalizationDataTensor::ViaLegalizationDataTensor(ViaData& data, torch::Tensor node_pos_, torch::Tensor via_bbox_) {
    num_nets = data.num_nets;
    node_pos = node_pos_.clone();
    via_bbox = via_bbox_.clone();
    x = node_pos.index({Slice(data.cell_mov_rhs, None), 0});
    y = node_pos.index({Slice(data.cell_mov_rhs, None), 1});
    node_weight = data.bonding_map.clone();

    row_shift = data.core_info[0].item<float>();
    via_site_width = (data.bondingInfo[0] + data.bondingInfo[2]).item<float>();
    via_site_height = (data.bondingInfo[1] + data.bondingInfo[2]).item<float>();
    num_via_x = torch::floor((data.core_info[1] - data.core_info[0]) / via_site_width).item<int>();
    num_via_y = data.numRows.item<int>();
    logger.info("Via sites: %d x %d = %d", num_via_x, num_via_y, num_via_x * num_via_y);

    via_site_x = torch::zeros({num_via_x, num_via_y}, torch::kFloat);
    via_site_y = torch::zeros({num_via_x, num_via_y}, torch::kFloat);
    via_index = -torch::ones({num_nets, 2}, torch::kInt); // TODO:
    via_map = -torch::ones({num_via_x, num_via_y}, torch::kInt);

    for (int i = 0; i < num_via_x; i++) {
        for (int j = 0; j < num_via_y; j++) {
            via_site_x[i][j] = via_site_width * ((float)i + 0.5) + row_shift;
            via_site_y[i][j] = via_site_height * ((float)j + 0.5) + row_shift;
        }
    }

    optim_box_x_l = via_bbox.index({"...", 0});
    optim_box_y_l = via_bbox.index({"...", 1});
    optim_box_x_h = via_bbox.index({"...", 2});
    optim_box_y_h = via_bbox.index({"...", 3});
    outer_box_x_l = via_bbox.index({"...", 4});
    outer_box_y_l = via_bbox.index({"...", 5});
    outer_box_x_h = via_bbox.index({"...", 6});
    outer_box_y_h = via_bbox.index({"...", 7});


    cout << via_site_x.max() << endl;
    cout << via_site_x.min() << endl;

    cout << via_site_y.max() << endl;
    cout << via_site_y.min() << endl;


} 

ViaLegalizationData::ViaLegalizationData(ViaData& data, ViaLegalizationDataTensor& at_db)
    : x(at_db.x.accessor<float, 1>()),
      y(at_db.y.accessor<float, 1>()),
      node_weight(at_db.node_weight.accessor<int, 1>()),
      via_site_x(at_db.via_site_x.accessor<float, 2>()),
      via_site_y(at_db.via_site_y.accessor<float, 2>()),
      via_map(at_db.via_map.accessor<int, 2>()),
      via_index(at_db.via_index.accessor<int, 2>()),
      optim_box_x_l(at_db.optim_box_x_l.accessor<float, 1>()),
      optim_box_y_l(at_db.optim_box_y_l.accessor<float, 1>()),
      optim_box_x_h(at_db.optim_box_x_h.accessor<float, 1>()),
      optim_box_y_h(at_db.optim_box_y_h.accessor<float, 1>()),
      outer_box_x_l(at_db.outer_box_x_l.accessor<float, 1>()),
      outer_box_y_l(at_db.outer_box_y_l.accessor<float, 1>()),
      outer_box_x_h(at_db.outer_box_x_h.accessor<float, 1>()),
      outer_box_y_h(at_db.outer_box_y_h.accessor<float, 1>())     
{
    num_via_x = at_db.num_via_x;
    num_via_y = at_db.num_via_y;
    row_shift = at_db.row_shift;
    via_site_width = at_db.via_site_width;
    via_site_height = at_db.via_site_height;
}

struct SiteCost {
    int index;
    float cost;
    SiteCost(int id, float c) : index(id), cost(c) {}
};

void ViaData::legalize_via_maching(NodeData& data, ViaLegalizationData& via_db, torch::Tensor node_pos) {
    /* via index */
    vector<int> via_map_back;
    for (int i = 0; i < num_nets; i++) {
        if (via_db.node_weight[i]) 
            via_map_back.push_back(i);
    }

    cout << via_map_back.size() << endl;
    cout << num_bonds << endl;


    cout << "------------------------1\n" << endl;
    cout << via_db.row_shift << endl;
    cout << via_db.via_site_width << endl;
    cout << via_db.via_site_height << endl;
    
    /* cost map */
    unordered_map<int, vector<SiteCost>> cost_edges;
    for (int i = 0; i < num_bonds; i++) {
        int via_id = via_map_back[i];
        // float optim_x_l = via_db.optim_box_x_l[via_id];
        // float optim_y_l = via_db.optim_box_y_l[via_id];
        // float optim_x_h = via_db.optim_box_x_h[via_id];
        // float optim_y_h = via_db.optim_box_y_h[via_id];
        // int optim_idx_x_l = std::floor((optim_x_l - via_db.row_shift) / via_db.via_site_width);
        // int optim_idx_y_l = std::floor((optim_y_l - via_db.row_shift) / via_db.via_site_height);
        // int optim_idx_x_h = std::ceil((optim_x_h - via_db.row_shift) / via_db.via_site_width);
        // int optim_idx_y_h = std::ceil((optim_y_h - via_db.row_shift) / via_db.via_site_height);

        // float outer_x_l = via_db.outer_box_x_l[via_id];
        // float outer_y_l = via_db.outer_box_y_l[via_id];
        // float outer_x_h = via_db.outer_box_x_h[via_id];
        // float outer_y_h = via_db.outer_box_y_h[via_id];
        // int outer_idx_x_l = std::floor((outer_x_l - via_db.row_shift) / via_db.via_site_width);
        // int outer_idx_y_l = std::floor((outer_y_l - via_db.row_shift) / via_db.via_site_height);
        // int outer_idx_x_h = std::ceil((outer_x_h - via_db.row_shift) / via_db.via_site_width);
        // int outer_idx_y_h = std::ceil((outer_y_h - via_db.row_shift) / via_db.via_site_height);

        // // via_db.x[via_id] = via_db.via_site_x[idx_x_l][idx_y_l];
        // // via_db.y[via_id] = via_db.via_site_y[idx_x_l][idx_y_l];
        // // if ((idx_x_h == idx_x_l) && (idx_y_l == idx_y_h)) cout << "warning\n";

        // for (int m = max(0, idx_x_l - st::setting.omni_int); m < min(via_db.num_via_x, idx_x_h + st::setting.omni_int); m++) {
        //     for (int n = max(0, idx_y_l - st::setting.omni_int); n < min(via_db.num_via_y, idx_y_h + st::setting.omni_int); n++) {
        //         int site_idx = n * via_db.num_via_x + m;
        //         if (site_idx >= 2000) cout <<  m << " " << n << " " << m + n * via_db.num_via_x << endl;
        //         float cost = 0;
        //         cost_edges[i].emplace_back(site_idx, cost);
        //     }
        // }
        
    }

    /* Creating Graph */
    logger.info("Creating Graph");
    UGraph g;
    DistMap distmap(g);

    vector<UGraph::Node> via_node;
    for (int i = 0; i < num_bonds; i++) {
        UGraph::Node u = g.addNode();
        via_node.push_back(u);
    }

    vector<UGraph::Node> via_sites;
    for (int i = 0; i < via_db.num_via_x * via_db.num_via_y; i++) {
        UGraph::Node v = g.addNode();
        via_sites.push_back(v);
    }

    /* Assign weight to edges; weight is Max_Distance - Distance */
    for (int i = 0; i < num_bonds; i++) {
        auto u = via_node[i];
        for (auto site_cost : cost_edges[i]) {
            auto v = via_sites[site_cost.index];
            if (site_cost.index >= 2000) cout << " ---- " << site_cost.index << endl;
            UGraph::Edge e = g.addEdge(u, v);
            distmap.set(e, site_cost.cost);
        }

    }

    /* Run Matching */
    MWM Matching(g, distmap);
    Matching.run();

    cout << "------------------------4\n" << endl;

    /* Matching to cluster */
    for (auto u : via_node) {
        if (Matching.mate(u) == INVALID) std::cout << "Error\n";
        else {
            int via_id = via_map_back[g.id(u)];
            int site_idx = g.id(Matching.mate(u)) - num_bonds; // FIXME: graph id??

            // if (site_idx >= 2000) {
            //     cout << " =====  " << site_idx << endl;
            // }
            // cout << site_idx << endl;

            int site_idx_y = site_idx / via_db.num_via_x;
            int site_idx_x = site_idx - via_db.num_via_x * site_idx_y;

            via_db.via_index[via_id][0] = site_idx_x;
            via_db.via_index[via_id][1] = site_idx_y;

            via_db.x[via_id] = via_db.via_site_x[site_idx_x][site_idx_y];
            via_db.y[via_id] = via_db.via_site_y[site_idx_x][site_idx_y];

            // via_db.x[via_id] = via_db.via_site_x[idx_x_l][idx_y_l];
            // via_db.y[via_id] = via_db.via_site_y[idx_x_l][idx_y_l];
            // if((via_db.x[via_id] < -1e5) || (via_db.y[via_id] < -1e5)) {
            //     cout << via_id << endl;
            //     cout << site_idx << endl;
            //     cout << " / " << via_db.num_via_x << " = " << site_idx / via_db.num_via_x << endl;
            //     cout << " / " << via_db.num_via_y << " = " << site_idx - via_db.num_via_x * site_idx_y << endl;
                
            //     cout << site_idx_x << " " << site_idx_y << endl;
            //     cout << via_db.via_site_x[site_idx_x][site_idx_y] << " " << via_db.via_site_y[site_idx_x][site_idx_y] << endl;
            // }
        }
    }

    cout << "------------------------5\n" << endl;

    // /* update via position */
    // for (int i = 0; i < num_nets; i++) {
    //     if (via_db.node_weight[i]) {
    //         int site_idx_x = via_db.via_index[i][0];
    //         int site_idx_y = via_db.via_index[i][1];
    //         via_db.x[i] = via_db.via_site_x[site_idx_x][site_idx_y];
    //         via_db.y[i] = via_db.via_site_y[site_idx_x][site_idx_y];
    //     }
    // }
    
    // int num_via_x = torch::floor((core_info[1] - core_info[0]) / via_site_width).item<int>();
    // int num_via_y = numRows.item<int>();
    // via_map = -torch::ones({num_via_x, num_via_y}, torch::kInt);
}