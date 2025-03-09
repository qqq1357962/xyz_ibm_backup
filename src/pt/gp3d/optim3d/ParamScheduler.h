#pragma once

#include "../placer3d/database3d.h"
#include "global.h"
#include "torch/script.h"
#include "placer/visualization.h"

namespace GP3D {

class MetricRecorder {
public:
    unordered_map<string, vector<double>> metrics;

    MetricRecorder(const vector<string>& names);

    void push(const unordered_map<string, double>& name2value);
    double get(const string& name, int idx);
    vector<double>& get_metric(const string& name);

    void visualize();  // TODO: write to file and call py script
};

class ParamScheduler {
private:
    vector<string> _metrics = {
        "hpwl",
        "hpwl_xy",
        "overflow",
        "mu",
        "wa_coeff",
        "density_weight",
        "precond_coef",
        "weighted_weight",
        "force_ratio",
        "net_weight_coef",
        "iteration",
    };
    double double_inf = std::numeric_limits<double>::max();
    // PlaceData& data;

public:
    ParamScheduler(NodeData3D& data);
    // ParamScheduler(VIAPlacer& bond_placer);

    MetricRecorder recorder;

    int iter;

    void reset(){
        density_weight = st::setting.density_weight * __init_density_weight__;
        precond_coef = 1.0;
        wa_coeff = 10 * base_gamma;
        iter = 0;
    }

    torch::Tensor best_sol;
    torch::Tensor best_sol_aux;
    torch::Tensor best_sol_rollback;
    unordered_map<string, double> best_metric;
    unordered_map<string, double> best_metric_aux;
    unordered_map<string, double> best_metric_rollback;

    double precond_coef;
    torch::Tensor precond_weight;
    double __init_density_weight__;
    double density_weight;
    double density_weight_xy;
    double density_weight_coef;
    double density_weight_coef_xy;
    double base_gamma;
    double base_gamma_xy;
    double wa_coeff;
    double wa_coeff_xy;
    double mu;
    int max_life;
    int life;
    double stop_overflow;
    double base_overflow = 1;
    double net_weight_coef;
    double force_coeff;

    bool use_precond;
    bool skip_update;
    bool enable_skip_update;
    bool enable_fence;
    bool early_stop_check_plateau = true;
    int min_stop_iter = 100;

    double force_ratio;
    double weighted_weight;

    // quad penalty
    double density_quad_coeff;
    torch::Tensor quad_penalty_coeff;
    torch::Tensor density_weight_grad_precond;
    torch::Tensor init_density;

    void set_init_param(double init_density_weight);
    // void push_metric(double hpwl, double overflow, int iteration);
    void push_metric(double hpwl, double hpwl_xy, double overflow, int iteration);
    void update_precond_weight();

    // void step(double hpwl, double overflow, torch::Tensor node_pos);
    void step(double hpwl, double hpwl_xy, double overflow, torch::Tensor node_pos);
    void step_density_weight();
    void step_density_weight_xy();
    void step_local_density_weight(torch::Tensor density_maps);
    bool local_density_lock = false;
    torch::Tensor density_weight_local;

    void step_wa_coeff();
    void step_wa_coeff_xy();
    void step_precond_coef();
    bool update_best_sol(torch::Tensor sol);
    bool need_to_early_stop();
    bool check_plateau(std::vector<double>& x, int window = 10, double threshold = 0.001);
    bool check_divergence(int window = 50, double threshold = 0.05);

    tuple<torch::Tensor, double, double, int> get_best_solution();

    void visualize();  // TODO: write to file and call py script

    /* node info for ovld constructor */
    torch::Tensor mov_node_to_num_pins;
    torch::Tensor mov_node_area;
    torch::Tensor node_size;
};

}  // namespace GP3D