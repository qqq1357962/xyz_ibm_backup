#include "Drawer.h"

bool DrawGlobalPlacement(
    const std::vector<double>& node_pos_x,
    const std::vector<double>& node_pos_y,
    const std::vector<double>& node_size_x,
    const std::vector<double>& node_size_y,
    const std::vector<std::string>& node_name,
    const std::tuple<double, double, double, double>& die_info,
    const std::tuple<double, double>& site_info,
    const std::tuple<double, double>& bin_size_info,
    const std::vector<std::tuple<index_type, index_type, std::string>>& node_types_indices,
    const std::vector<std::tuple<std::string, double, double, double, double>>& ele_type_to_rgba_vec,
    const std::string& filename,
    double width,
    double height,
    const std::vector<std::string>& draw_contents,
    int iopin_mov_lhs,
    int iopin_mov_rhs,
    bool debug_mode=false,
    std::vector<std::pair<int,int> > poses_vis={});
