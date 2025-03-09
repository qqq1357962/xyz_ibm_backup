#include "../run_placement.h"
#include "fp/fp/floorplan.h"
#include "utils/myUtils.h"

void run_patoh() {
    /* General settings */
    logger.info("#threads %d", st::setting.num_threads);
    torch::set_num_threads(st::setting.num_threads);
    setenv("OMP_NUM_THREADS", to_string(st::setting.num_threads).c_str(), true);

    /* set device */
    torch::Device device = torch::kCPU;
    int num_device = torch::cuda::device_count();
    logger.info("CUDA DEVICE COUNT: %d", num_device);
    if (torch::cuda::is_available() && (st::setting.gpu < num_device) && (st::setting.gpu >= 0)) {
        logger.info("CUDA is available! Training on GPU %d.", st::setting.gpu);
        // device = torch::kCUDA;

        device = torch::Device(torch::kCUDA, st::setting.gpu);
    }

    logger.info("Use Nesterov optimizer!");
    if (st::setting.scale_design) {
        logger.warning("Eplace's nesterov optimizer cannot support normalized die. Disable scale_design.");
        st::setting.scale_design = false;
    }

    /* tmp output for verification */
    std::string output_path_tmp = st::setting.output_path + ".tmp";

    /* visualization via color */
    vector<double> viaColor = {0.5, 0.3, 0.6, 0.5};  // TODO: via color
    // ======================================================================================================
    //
    //                                             PARTITION
    //
    // ======================================================================================================
    /* database */
    auto [design_info, rawdb, gpdb] = load_dataset();
    NodeData data(design_info, device);
    auto macro_mask_2d = data.macro_mask.clone().unsqueeze(1);
    // data.setMacroOrient();
    // grad.slice(0, 0, macro_mask_2d.size(0)) *= (1-0.99*macro_mask_2d);
    // for(int i=0;i<data.pin_rel_cpos.size(0);i++)
    // {
    //     int node_id = data.pin_id2node_id[i].item<int>();
    //     if(data.macro_mask[node_id].item<int>()==1)
    //     {
    //         data.pin_rel_cpos_bot[i]*=(data.node_size_bot[0]/data.node_size_bot[node_id]);
    //         data.pin_rel_cpos_top[i]*=(data.node_size_top[0]/data.node_size_top[node_id]);
    //         data.pin_rel_cpos[i]*=(data.node_size[0]/data.node_size[node_id]);
    //     }
    // }

    // for(int i=0;i<data.node_size_bot.size(0);i++)
    // {
    //     if(data.macro_mask[i].item<int>()==1)
    //     {
    //         data.node_size_bot[i]=data.node_size_bot[0];
    //         data.node_size_top[i]=data.node_size_top[0];
    //         data.node_size[i]=data.node_size[0];
    //         data.node_area_bot[i]=data.node_size[i][0]*data.node_size[i][1];

    //     }
    // }

    data.preprocess();
    torch::Tensor node_die;
    torch::Tensor node_pos;
    torch::Tensor node_size;

    states hpwl_state;

    /* node information */
    int cell_mov_lhs;
    int cell_mov_rhs;
    int via_mov_lhs;
    int via_mov_rhs;
    int mov_lhs;
    int mov_rhs;
    torch::Tensor node_size_bot;
    torch::Tensor node_size_top;
    torch::Tensor node_orient_bot;
    torch::Tensor node_orient_top;
    /* cell visualization information */
    node_size_bot = torch::zeros({data.num_nodes, 2}, torch::dtype(torch::kFloat));
    node_size_top = torch::zeros({data.num_nodes, 2}, torch::dtype(torch::kFloat));
    std::tie(cell_mov_lhs, cell_mov_rhs) = data.movable_index;
    logger.info("#mode: %d", st::setting.mode);

    /* pt/via database */
    ViaData via_data;

    torch::Tensor partial_hpwl3d;
    Partitioner pt(data, hpwl_state);
    pt.run_patoh(data);
}