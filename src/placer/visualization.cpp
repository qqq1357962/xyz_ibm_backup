#include "visualization.h"

bool draw_fig_with_cairo_cpp(torch::Tensor node_pos,
                             torch::Tensor node_size,
                             PlaceData &data,
                             tuple<int, int, string> info,
                             vector<double> rgbv,
                             int base_size) {
    if(st::setting.skip_draw)
    {
        return 0;
    }
    auto _die_info = data.die_info_back_up;  // FIXME:

    double lx = _die_info[0].item().toDouble();
    double hx = _die_info[1].item().toDouble() * 1.01;
    double ly = _die_info[2].item().toDouble();
    double hy = _die_info[3].item().toDouble() * 1.01;
    auto die_info = make_tuple(lx, hx, ly, hy);

    // TODO: disable prescale
    auto x_pos = torch::_cast_Double(node_pos.to(torch::kCPU)).index({"...", 0}).contiguous();
    auto y_pos = torch::_cast_Double(node_pos.to(torch::kCPU)).index({"...", 1}).contiguous();
    auto x_size = torch::_cast_Double(node_size.to(torch::kCPU)).index({"...", 0}).contiguous();
    auto y_size = torch::_cast_Double(node_size.to(torch::kCPU)).index({"...", 1}).contiguous();

    vector<double> node_pos_x(x_pos.data_ptr<double>(), x_pos.data_ptr<double>() + x_pos.numel());
    vector<double> node_pos_y(y_pos.data_ptr<double>(), y_pos.data_ptr<double>() + y_pos.numel());
    vector<double> node_size_x(x_size.data_ptr<double>(), x_size.data_ptr<double>() + x_size.numel());
    vector<double> node_size_y(y_size.data_ptr<double>(), y_size.data_ptr<double>() + y_size.numel());

    auto name = torch::arange(node_pos.size(0), torch::dtype(torch::kInt)).contiguous();
    vector<int> name_i(name.data_ptr<int>(), name.data_ptr<int>() + name.numel());
    vector<string> node_name;
    for (int i : name_i) node_name.push_back(to_string(i));

    auto [recursion, iteration, design_name] = info;
    std::filesystem::path filename("R" + to_string(recursion) + "_iter[" + to_string(iteration) + "]" + design_name + ".png");
    std::filesystem::path current_dir(std::filesystem::current_path());
    std::filesystem::path result_dir(st::setting.result_dir);
    std::filesystem::path exp_id(st::setting.exp_id);
    std::filesystem::path png_path = (current_dir / result_dir / exp_id / filename);

    tuple<double, double> site_info = make_tuple(data.site_width, data.site_height);
    tuple<double, double> bin_size_info =
        //make_tuple(round(1 / double(data.num_bin_x) * (hx - lx)), round(1 / double(data.num_bin_y) * (hy - ly)));
        make_tuple(1 / double(data.num_bin_x) * (hx - lx), 1 / double(data.num_bin_y) * (hy - ly));

    auto node_type_indices = data.node_type_indices;

    const std::vector<std::tuple<std::string, double, double, double, double>> ele_type_to_rgba_vec = {
        make_tuple("Bin", 0.1, 0.1, 0.1, 1.0),
        make_tuple("Mov", rgbv[0], rgbv[1], rgbv[2], rgbv[3]),
        make_tuple("Via", 0.5, 0.3, 0.6, 0.5),
        make_tuple("Filler", 0.8, 0.8, 0.8, 0.8)};

    int width = base_size * st::setting.draw_mat_size;
    int height = round(width * (hy - ly) / (hx - lx));

    auto node_num = node_pos.size(0);
    int iopin_mov_lhs = (node_num == data.num_nets) ? 0 : data.iopin_mov_lhs;
    int iopin_mov_rhs = (node_num == data.num_nets) ? 0 : data.iopin_mov_rhs;

    std::vector<std::string> draw_contents = {"Nodes", "NodesText"};
    // std::rotate(node_type_indices.begin(), node_type_indices.begin() + 1, node_type_indices.end());
    return DrawGlobalPlacement(node_pos_x,
                               node_pos_y,
                               node_size_x,
                               node_size_y,
                               node_name,
                               die_info,
                               site_info,
                               bin_size_info,
                               node_type_indices,
                               ele_type_to_rgba_vec,
                               png_path,
                               width,
                               height,
                               draw_contents,
                               iopin_mov_lhs,
                               iopin_mov_rhs);
}

bool draw_fig_with_cairo_cpp2(torch::Tensor node_pos,
                             torch::Tensor node_size,
                             PlaceData &data,
                             tuple<int, int, string> info,
                             int check_node_id,
                             int show_depth) {
    vector<double> rgbv = {0.475, 0.706, 0.718, 0.6};
    cout<<"call me???"<<endl;
    int base_size = 2048;
    vector<int> visited;
    visited.resize(node_pos.size(0));
    auto _die_info = data.die_info;  // FIXME:

    double lx = _die_info[0].item().toDouble();
    double hx = _die_info[1].item().toDouble();
    double ly = _die_info[2].item().toDouble();
    double hy = _die_info[3].item().toDouble();
    auto die_info = make_tuple(lx, hx, ly, hy);

    // TODO: disable prescale
    auto x_pos = torch::_cast_Double(node_pos).index({"...", 0}).contiguous();
    auto y_pos = torch::_cast_Double(node_pos).index({"...", 1}).contiguous();
    auto x_size = torch::_cast_Double(node_size).index({"...", 0}).contiguous();
    auto y_size = torch::_cast_Double(node_size).index({"...", 1}).contiguous();

    vector<double> node_pos_x(x_pos.data_ptr<double>(), x_pos.data_ptr<double>() + x_pos.numel());
    vector<double> node_pos_y(y_pos.data_ptr<double>(), y_pos.data_ptr<double>() + y_pos.numel());
    vector<double> node_size_x(x_size.data_ptr<double>(), x_size.data_ptr<double>() + x_size.numel());
    vector<double> node_size_y(y_size.data_ptr<double>(), y_size.data_ptr<double>() + y_size.numel());

    auto name = torch::arange(node_pos.size(0), torch::dtype(torch::kInt)).contiguous();
    vector<int> name_i(name.data_ptr<int>(), name.data_ptr<int>() + name.numel());
    vector<string> node_name;
    for (int i : name_i) node_name.push_back(to_string(i));

    auto [recursion, iteration, design_name] = info;
    std::filesystem::path filename("R" + to_string(recursion) + "_iter[" + to_string(iteration) + "]" + design_name + ".png");
    std::filesystem::path current_dir(std::filesystem::current_path());
    std::filesystem::path result_dir(st::setting.result_dir);
    std::filesystem::path exp_id(st::setting.exp_id);
    std::filesystem::path png_path = (current_dir / result_dir / exp_id / filename);

    tuple<double, double> site_info = make_tuple(data.site_width, data.site_height);
    tuple<double, double> bin_size_info =
        //make_tuple(round(1 / double(data.num_bin_x) * (hx - lx)), round(1 / double(data.num_bin_y) * (hy - ly)));
        make_tuple(1 / double(data.num_bin_x) * (hx - lx), 1 / double(data.num_bin_y) * (hy - ly));

    auto node_type_indices = data.node_type_indices;

    const std::vector<std::tuple<std::string, double, double, double, double>> ele_type_to_rgba_vec = {
        make_tuple("Bin", 0.1, 0.1, 0.1, 1.0),
        make_tuple("Mov", rgbv[0], rgbv[1], rgbv[2], rgbv[3]),
        make_tuple("Via", 0.5, 0.3, 0.6, 0.5),
        make_tuple("Filler", 0.8, 0.8, 0.8, 0.8),
        make_tuple("Checking", 1.0, 0, 0, 1.0),
        make_tuple("Neighbors", 0.0, 0.0, 1.0, 1.0)};

    int width = base_size * st::setting.draw_mat_size;
    int height = round(width * (hy - ly) / (hx - lx));

    std::vector<std::string> draw_contents = {"Nodes", "NodesText"};
    std::rotate(node_type_indices.begin(), node_type_indices.begin() + 1, node_type_indices.end());
    node_type_indices.emplace_back(make_tuple(check_node_id,check_node_id+1,"Checking"));
    queue<pair<int,int> > q;
    q.push(make_pair(check_node_id,0));
    while(!q.empty())
    {
        int node_id = q.front().first;
        int layer = q.front().second;
        // cout<<"visiting "<<node_id <<endl;
        q.pop();
        if(visited[node_id])
        {
            continue;
        }
        visited[node_id]=1;
        if(node_id!=check_node_id)
        {
            node_type_indices.emplace_back(make_tuple(node_id,node_id+1,"Neighbors"));
        }
        if(layer>=show_depth)
        {
            continue;
        }
        int start_idx=0;
        if(node_id>0)
        {
            start_idx = data.node2pin_list_end[node_id - 1].item<int>();
        }
        int end_idx = data.node2pin_list_end[node_id].item<int>();
        for(int i=start_idx;i<end_idx;i++)
        {
            int pin_id = data.node2pin_list[i].item<int>();
            int net_id = data.pin_id2net_id[pin_id].item<int>();
            int start_idx_net = 0;
            if(net_id>0)
            {
                start_idx_net = data.hyperedge_list_end[net_id-1].item<int>();
            }
            int end_idx_net = data.hyperedge_list_end[net_id].item<int>();
            for(int j=start_idx_net;j<end_idx_net;j++)
            {
                int pin_id_next = data.hyperedge_list[j].item<int>();
                if(pin_id==pin_id_next)
                {
                    continue;
                }
                int next_id = data.pin_id2node_id[pin_id_next].item<int>();//这个不够
                q.push(make_pair(next_id,layer+1));
            }            
        }
    }
    vector<pair<int,int> > poses_vis;
    int start_idx=0;
    if(check_node_id>0)
    {
        start_idx = data.node2pin_list_end[check_node_id - 1].item<int>();
    }
    int end_idx = data.node2pin_list_end[check_node_id].item<int>();
    float pin_rel_x_sum = 0;
    float pin_rel_y_sum = 0;
    for(int i=start_idx;i<end_idx;i++)
    {
        int pin_id = data.node2pin_list[i].item<int>();
        int rel_pos_x = data.pin_rel_cpos[pin_id][0].item<int>();
        int rel_pos_y = data.pin_rel_cpos[pin_id][1].item<int>();
        pin_rel_x_sum+=rel_pos_x;
        pin_rel_y_sum+=rel_pos_y;
        int center_x = node_pos[check_node_id][0].item<int>();
        int center_y = node_pos[check_node_id][1].item<int>();
        poses_vis.push_back(make_pair(rel_pos_x+center_x, rel_pos_y+center_y));
    }
    int num = end_idx - start_idx;
    cout<<"sum: "<< pin_rel_x_sum<<" "<<pin_rel_y_sum<<endl;
    pin_rel_x_sum/=num;
    pin_rel_y_sum/=num;
    cout<<"mean: "<< pin_rel_x_sum<<" "<<pin_rel_y_sum<<endl;

    return DrawGlobalPlacement(node_pos_x,
                               node_pos_y,
                               node_size_x,
                               node_size_y,
                               node_name,
                               die_info,
                               site_info,
                               bin_size_info,
                               node_type_indices,
                               ele_type_to_rgba_vec,
                               png_path,
                               width,
                               height,
                               draw_contents,
                               data.iopin_mov_lhs,
                               data.iopin_mov_rhs,
                               true,
                               poses_vis);
}

void saveChannels(const torch::Tensor& tensor, const std::string& filenamePrefix)
{
    if (tensor.dim() != 3 || tensor.size(0) != 3)
    {
        std::cerr << "Invalid tensor shape. Expected shape: (3, h, w)" << std::endl;
        return;
    }
    int width = tensor.size(2);
    int height = tensor.size(1);
    cairo_surface_t* surface;
    cairo_t* cr;
    std::string filename_string;
    std::filesystem::path current_dir(std::filesystem::current_path());
    std::filesystem::path result_dir(st::setting.result_dir);
    std::filesystem::path exp_id(st::setting.exp_id);
    for (int channel = 0; channel < 3; ++channel)
    {
        filename_string = filenamePrefix + "_" + std::to_string(channel) + ".png";
        std::filesystem::path filename(filename_string);
        std::filesystem::path png_path = (current_dir / result_dir / exp_id / filename);
        surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, width, height);
        cr = cairo_create(surface);
        // Create a new image surface from Tensor data
        torch::Tensor channelData = tensor.select(0, channel).contiguous();
        torch::Tensor imageData = channelData.permute({1, 0}).contiguous().to(torch::kUInt8);
        cairo_surface_t* imageSurface = cairo_image_surface_create_for_data(
            static_cast<unsigned char*>(imageData.data_ptr()),
            CAIRO_FORMAT_RGB24, width, height, width * 3);
        // Draw the image surface onto the Cairo context
        cairo_set_source_surface(cr, imageSurface, 0, 0);
        cairo_paint(cr);
        // Clean up
        cairo_surface_destroy(imageSurface);
        cairo_destroy(cr);
        cairo_surface_write_to_png(surface, filename.c_str());
        cairo_surface_destroy(surface);
    }
}

void logWireLength(NodeData& data, torch::Tensor node_pos, torch::Tensor node_die)
{
    std::ofstream fs("checkNets.txt");
    for(int i=0;i<data.num_nets;i++)
    {
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = data.hyperedge_list_end[i - 1].item<int64_t>();
        }
        int64_t end_idx = data.hyperedge_list_end[i].item<int64_t>();
        fs<<"Net N"+to_string(i+1)+" "+to_string(end_idx-start_idx-1)<<endl;
        /* add cell pin */
        for (int64_t idx = start_idx; idx < end_idx-1; idx++) {
            int pin = data.hyperedge_list[idx].item<int>();
            int node_id = data.pin_id2node_id[pin].item<int>();
            fs<<"Pin C"+to_string(node_id+1)<<endl;
        }
    }
    fs<<"loggggggggg"<<endl;
    int hpwl_all=0;
    int hpwl_all2=0;
    int hpwl_all_0=0;
    int hpwl_all_1=0;
    for(int i=0;i<data.num_nets;i++)
    {
        fs<<"net "<<i<<endl;
        double max_x_0=-2000000;
        double max_y_0=-2000000;
        double min_x_0=20000000;
        double min_y_0=20000000;

        double max_x_1=-2000000;
        double max_y_1=-2000000;
        double min_x_1=20000000;
        double min_y_1=20000000;

        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = data.hyperedge_list_end[i - 1].item<int64_t>();
        }
        int64_t end_idx = data.hyperedge_list_end[i].item<int64_t>();
        /* add cell pin */
        for (int64_t idx = start_idx; idx < end_idx; idx++) {
            int pin = data.hyperedge_list[idx].item<int>();
            int node_id = data.pin_id2node_id[pin].item<int>();
            double node_x = node_pos[node_id][0].item<double>();
            double node_y = node_pos[node_id][1].item<double>();
            double pin_rel_x = data.pin_rel_cpos[pin][0].item<double>();
            double pin_rel_y = data.pin_rel_cpos[pin][1].item<double>();
            double pin_x = node_x+pin_rel_x;
            double pin_y = node_y+pin_rel_y;
            if(node_die[node_id].item<int>()==0||node_die[node_id].item<int>()==2)
            {
                if(pin_x>max_x_0) max_x_0=pin_x;
                if(pin_y>max_y_0) max_y_0=pin_y;
                if(pin_x<min_x_0) min_x_0=pin_x;
                if(pin_y<min_y_0) min_y_0=pin_y;
            }
            if(node_die[node_id].item<int>()==1||node_die[node_id].item<int>()==2)
            {
                if(pin_x>max_x_1) max_x_1=pin_x;
                if(pin_y>max_y_1) max_y_1=pin_y;
                if(pin_x<min_x_1) min_x_1=pin_x;
                if(pin_y<min_y_1) min_y_1=pin_y;
            }
            double node_size_x = data.node_size[node_id][0].item<double>();
            double node_size_y = data.node_size[node_id][1].item<double>();
            fs<<"node_id: C"<<node_id+1<<" die_id: "<<node_die[node_id].item<int>()<<endl;
            fs<<"node size:"<<node_size_x<<" "<<node_size_y<<endl;
            fs<<" node_pos: "<<node_x<<" "<<node_y<<" rel: "<<pin_rel_x<<" "<<pin_rel_y<<" pin pos:"<<pin_x<<" "<<pin_y<<endl;
            fs<<" node_pos: "<<node_x-node_size_x/2<<" "<<node_y-node_size_y/2
            <<" rel: "<<pin_rel_x+node_size_x/2<<" "<<pin_rel_y+node_size_y/2<<" pin pos:"<<pin_x<<" "<<pin_y<<endl;
        }
        double hpwl0_x=0;
        double hpwl1_x=0;
        double hpwl0_y=0;
        double hpwl1_y=0;
        if(max_x_0>0)
        {
            hpwl0_x += max_x_0-min_x_0;
        }
        if(max_y_0>0)
        {
            hpwl0_y += max_y_0-min_y_0;
        }
        if(max_x_1>0)
        {
            hpwl1_x += max_x_1-min_x_1;
        }
        if(max_y_1>0)
        {
            hpwl1_y += max_y_1-min_y_1;
        }

        double hpwl0=hpwl0_x+hpwl0_y;
        double hpwl1=hpwl1_x+hpwl1_y;

        hpwl_all+=hpwl0+hpwl1;
        hpwl_all2+=hpwl0;
        hpwl_all2+=hpwl1;
        hpwl_all_0+=hpwl0;
        hpwl_all_1+=hpwl1;
        fs<<"hpwl0_x: "<<hpwl0_x<<" hpwl0_y: "<<hpwl0_y<<endl;
        fs<<"hpwl1_x: "<<hpwl0_x<<" hpwl1_y: "<<hpwl1_y<<endl;
        fs<<"accumulate: "<<hpwl_all_0<<" "<<hpwl_all_1<<" "<<hpwl_all<<" "<<hpwl_all2<<endl;
    }
    fs<<"together: "<<hpwl_all_0<<" "<<hpwl_all_1<<" "<<hpwl_all<<" "<<hpwl_all2<<endl;
    fs<<"together: "<<(hpwl_all_0+hpwl_all_1)<<endl;
    fs<<"loggggggggg"<<endl;
    fs.close();
    cout<<"together: "<<hpwl_all_0<<" "<<hpwl_all_1<<" "<<hpwl_all<<" "<<hpwl_all2<<endl;
    cout<<"together: "<<(hpwl_all_0+hpwl_all_1)<<endl;
    cout<<"loggggggggg"<<endl;
    logger.info("hpwl: %d", hpwl_all2);
}

bool draw_fig_with_cairo_cpp_cross_chip(
    torch::Tensor node_pos, torch::Tensor node_size, PlaceData &data, tuple<int, int, string> info, int base_size) {
    auto _die_info = data.die_info_back_up; // FIXME:

    double lx = _die_info[0].item().toDouble();
    double hx = _die_info[1].item().toDouble();
    double ly = _die_info[2].item().toDouble();
    double hy = _die_info[3].item().toDouble();
    auto die_info = make_tuple(lx, hx, ly, hy);

    // TODO: disable prescale
    auto x_pos = torch::_cast_Double(node_pos).index({"...", 0}).contiguous();
    auto y_pos = torch::_cast_Double(node_pos).index({"...", 1}).contiguous();
    auto x_size = torch::_cast_Double(node_size).index({"...", 0}).contiguous();
    auto y_size = torch::_cast_Double(node_size).index({"...", 1}).contiguous();

    vector<double> node_pos_x(x_pos.data_ptr<double>(), x_pos.data_ptr<double>() + x_pos.numel());
    vector<double> node_pos_y(y_pos.data_ptr<double>(), y_pos.data_ptr<double>() + y_pos.numel());
    vector<double> node_size_x(x_size.data_ptr<double>(), x_size.data_ptr<double>() + x_size.numel());
    vector<double> node_size_y(y_size.data_ptr<double>(), y_size.data_ptr<double>() + y_size.numel());

    auto name = torch::arange(node_pos.size(0), torch::dtype(torch::kInt)).contiguous();
    vector<int> name_i(name.data_ptr<int>(), name.data_ptr<int>() + name.numel());
    vector<string> node_name;
    for (int i : name_i) node_name.push_back(to_string(i));

    // auto [iteration, hpwl, design_name] = info;
    auto [recursion, iteration, design_name] = info;
    // std::filesystem::path filename("iter" + to_string(iteration) + "-" + design_name + ".png");
    std::filesystem::path filename("R" + to_string(recursion) + "_iter[" + to_string(iteration) + "]" + design_name + ".png");
    std::filesystem::path current_dir(std::filesystem::current_path());
    std::filesystem::path result_dir(st::setting.result_dir);
    std::filesystem::path exp_id(st::setting.exp_id);
    std::filesystem::path png_path = (current_dir / result_dir / exp_id / filename);

    tuple<double, double> site_info = make_tuple(data.site_width, data.site_height);
    tuple<double, double> bin_size_info =
        //make_tuple(round(1 / double(data.num_bin_x) * (hx - lx)), round(1 / double(data.num_bin_y) * (hy - ly)));
        make_tuple(1 / double(data.num_bin_x) * (hx - lx), 1 / double(data.num_bin_y) * (hy - ly));

    vector<tuple<gp::index_type, gp::index_type, string>> node_type_indices;
    node_type_indices.emplace_back(std::make_tuple(0, int(node_pos.size(0) / 2), "Bot"));
    node_type_indices.emplace_back(std::make_tuple(int(node_pos.size(0) / 2), int(node_pos.size(0)), "Top"));

    const std::vector<std::tuple<std::string, double, double, double, double>> ele_type_to_rgba_vec = {
        make_tuple("Bin", 0.1, 0.1, 0.1, 1.0),
        // make_tuple("Bot", 0.5, 0.7, 0.7, 0.7),
        // make_tuple("Top", 0.7, 0.3, 0.5, 0.5),
        make_tuple("Bot", 0.75, 0.3, 0.25, 0.5),
        make_tuple("Top", 0.5, 0.6, 0.77, 0.7),
    };

    int width = base_size * st::setting.draw_mat_size;
    int height = round(width * (hy - ly) / (hx - lx));

    // std::vector<std::string> draw_contents = {"Nodes", "NodesText"};
    std::vector<std::string> draw_contents = {"Nodes"};

    return DrawGlobalPlacement(node_pos_x,
                               node_pos_y,
                               node_size_x,
                               node_size_y,
                               node_name,
                               die_info,
                               site_info,
                               bin_size_info,
                               node_type_indices,
                               ele_type_to_rgba_vec,
                               png_path,
                               width,
                               height,
                               draw_contents,
                               data.iopin_mov_lhs,
                               data.iopin_mov_rhs);
}

void plot_pt(torch::Tensor data, char *cmd, char *path, char *fig) {
    // std::string data_dir = std::string(path)+ "/tmp.zip";
    std::string data_dir = "tmp.zip";
    std::string fig_dir = std::string(path) + "/" + std::string(fig);

    auto bytes = torch::pickle_save(data);
    std::ofstream fout(data_dir, std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();

    string py_cmd = "python src/plot.py --cmd " + std::string(cmd) + \
                    " --path " + std::string(data_dir) +
                    " --fig " + std::string(fig_dir);

    cout << py_cmd << endl;
    system(py_cmd.c_str());
}

void plot_pt(vector<float> data_vec, char *cmd, char *path, char *fig, char *key) {
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    auto data = torch::from_blob(data_vec.data(), {static_cast<long>(data_vec.size())}, options);

    // std::string data_dir = std::string(path)+ "/tmp.zip";
    std::string data_dir = "tmp.zip";
    std::string fig_dir = std::string(path) + "/" + std::string(fig);

    auto bytes = torch::pickle_save(data);
    std::ofstream fout(data_dir, std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();

    string py_cmd = "python src/plot.py --cmd " + std::string(cmd) + \
                    " --path " + std::string(data_dir) +
                    " --fig " + std::string(fig_dir) +
                    " --key " + std::string(key);

    // cout << py_cmd << endl;
    system(py_cmd.c_str());
}