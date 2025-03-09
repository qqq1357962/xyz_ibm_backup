#pragma once

#include "database.h"
#include "global.h"

bool draw_fig_with_cairo_cpp(torch::Tensor node_pos,
                             torch::Tensor node_size,
                             PlaceData &data,
                             tuple<int, int, string> info,
                             vector<double> rgbv = {0.475, 0.706, 0.718, 0.6},
                             int base_size = 2048);
bool draw_fig_with_cairo_cpp2(torch::Tensor node_pos,
                             torch::Tensor node_size,
                             PlaceData &data,
                             tuple<int, int, string> info,
                             int check_node_id,
                             int show_depth);

void saveChannels(const torch::Tensor& tensor, const std::string& filenamePrefix);

void logWireLength(NodeData& data, torch::Tensor node_pos, torch::Tensor node_die);

bool draw_fig_with_cairo_cpp_cross_chip(torch::Tensor node_pos,
                                        torch::Tensor node_size,
                                        PlaceData &data,
                                        tuple<int, int, string> info,
                                        int base_size = 2048);

void plot_pt(torch::Tensor data, char *cmd = (char *)"plot", char *path = (char *)"./", char *fig = (char *)"fig.png");

void plot_pt(vector<float> data_vec,
             char *cmd = (char *)"plot",
             char *path = (char *)"./",
             char *fig = (char *)"fig.png",
             char *key = (char *)"default");
