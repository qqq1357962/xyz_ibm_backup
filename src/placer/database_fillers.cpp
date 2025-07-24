#include "database.h"

void NodeData::compute_filler() {
    use_filler = st::setting.use_filler;
    if (st::setting.use_filler) {
        auto [mov_lhs, mov_rhs] = movable_index;
        at::Tensor mov_node_size = node_size.index({Slice(mov_lhs, mov_rhs)});
        at::Tensor die_area = at::prod(die_ur - die_ll);
        // init_density_map already multiplies with args.target_density,
        // we need to divide it back
        at::Tensor ori_dmap = (init_density_map / target_density).sum();
        // init_density_map are all normalized to (0.0, 1.0)
        at::Tensor fixed_node_area = ori_dmap * bin_area;
        at::Tensor placeable_area = die_area - fixed_node_area;
        if (true) {
            // Following DREAMPlace
            at::Tensor mov_cell_area = torch::prod(mov_node_size, 1);
            int num_movable_nodes = mov_rhs - mov_lhs;
            at::Tensor mov_node_xsize_order = torch::argsort(mov_node_size.index({"...", 0}));

            at::Tensor filler_size_x =
                torch::mean(mov_node_size.index({"...", 0})
                                .index({mov_node_xsize_order.index(
                                    {Slice(int(num_movable_nodes * 0.05), int(num_movable_nodes * 0.95))})}));
            at::Tensor filler_size_y = site_height / __die_scale__[1];
            //std::cout<<"mov_node_size"<<mov_node_size<<std::endl;
            //std::cout<<"filler_size_x: "<<filler_size_x<<" filler_size_y: "<<filler_size_y<<std::endl;

            at::Tensor total_filler_area =
                max(target_density * placeable_area - torch::sum(mov_cell_area), at::tensor(0.0));
            single_filler_size = at::tensor({filler_size_x.item<float>(), filler_size_y.item<float>()},
                                            torch::dtype(mov_node_size.dtype()));
            
            //std::cout<<target_density<<" placeable_area:"<<placeable_area.item<float>()<<" mov_cell_area:"
            //       <<torch::sum(mov_cell_area).item<float>()<<"???"<<__die_scale__[1].item<float>()<<std::endl;
            // float xx1 = total_filler_area.item<float>();
            // float xx2 = (filler_size_x * filler_size_y).item<float>();
            // std::cout<<xx1<<" "<<xx2<<std::endl;

            __num_fillers__ = torch::round(total_filler_area / (filler_size_x * filler_size_y)).item<int>();
        }
        if (__num_fillers__ > 0) {
            filler_size = single_filler_size.repeat({__num_fillers__, 1});
            logger.info("#Fillers: %d [%.2f], Filler size: (%.4e, %.4e)",
                        __num_fillers__,
                        target_density,
                        single_filler_size[0].item<float>(),
                        single_filler_size[1].item<float>());
        } else {
            logger.info(
                "num_fillers[%d] is smaller or equal to 0. Please make sure target_density[%.2f]"
                " is larger than movable cell utilization[%.2f]. use_filler is disable.",
                __num_fillers__,
                target_density,
                torch::sum(torch::prod(mov_node_size, 1)).item<float>());
            use_filler = false;
        }
    }
}  // END MODULE

//---------------------------------------------------------------------

void NodeData::compute_precond_var() {
    auto [mov_lhs, mov_rhs] = movable_index;
    mov_node_area = node_area.index({Slice(mov_lhs, mov_rhs)});

    mov_node_to_num_pins = node_to_num_pins.index({Slice(mov_lhs, mov_rhs)});
    if (use_filler) {
        at::Tensor filler_area = torch::prod(filler_size, 1).unsqueeze(1);
        mov_node_area = torch::cat({mov_node_area, filler_area}, 0);
        at::Tensor filler_to_num_pins = mov_node_to_num_pins.new_zeros({__num_fillers__, 1});
        assert(filler_to_num_pins.sizes() == filler_area.sizes());
        mov_node_to_num_pins = torch::cat({mov_node_to_num_pins, filler_to_num_pins}, 0);
    }
}  // END MODULE

//---------------------------------------------------------------------

void NodeData::compute_sorted_node_map() {
    auto [tmp, mov_sorted_map] = torch::sort(mov_node_area.flatten(), 0, true);

    mov_sorted_map = mov_sorted_map.contiguous();
    at::Tensor mov_conn_sorted_map = mov_sorted_map;
    if (use_filler) {
        auto [mov_lhs, mov_rhs] = movable_index;
        auto [tmp1, mov_conn_sorted_map] =
            torch::sort(mov_node_area.index({Slice(mov_lhs, mov_rhs)}).flatten(), 0, true);
        auto [tmp2, filler_sorted_map] = torch::sort(mov_node_area.index({Slice(mov_rhs)}).flatten(), 0, true);
        mov_conn_sorted_map = mov_conn_sorted_map.contiguous();
        filler_sorted_map = filler_sorted_map.contiguous();
        sorted_maps =
            make_tuple(mov_sorted_map.to(device), mov_conn_sorted_map.to(device), filler_sorted_map.to(device));
    } else {
        auto [mov_lhs, mov_rhs] = movable_index;
        auto [tmp1, mov_conn_sorted_map] =
            torch::sort(mov_node_area.index({Slice(mov_lhs, mov_rhs)}).flatten(), 0, true);
        at::Tensor filler_sorted_map;
        mov_conn_sorted_map = mov_conn_sorted_map.contiguous();
        sorted_maps = make_tuple(mov_sorted_map.to(device), mov_conn_sorted_map.to(device), filler_sorted_map);
    }
}  // END MODULE

//---------------------------------------------------------------------

void NodeData::init_filler() {
    logger.info("Processing fillers");
    compute_filler();
    compute_precond_var();
    compute_sorted_node_map();
}  // END MODULE

//---------------------------------------------------------------------