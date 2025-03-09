#pragma once

#include "global.h"
#include "placer/database.h"
#include "torch/script.h"
#include "placer/visualization.h"

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
        "overflow",
        "mu",
        "wa_coeff",
        "density_weight",
        "precond_coef",
        "weighted_weight",
        "force_ratio",
        "iteration",
    };
    double double_inf = std::numeric_limits<double>::max();
    // PlaceData& data;

public:
    ParamScheduler(NodeData& data);
    // ParamScheduler(VIAPlacer& bond_placer);

    MetricRecorder recorder;

    int iter;

    torch::Tensor best_sol;
    torch::Tensor best_sol_aux;
    torch::Tensor best_sol_rollback;
    unordered_map<string, double> best_metric;
    unordered_map<string, double> best_metric_aux;
    unordered_map<string, double> best_metric_rollback;

    double precond_coef;
    torch::Tensor precond_weight;
    double density_weight;
    double density_weight_coef;
    double base_gamma;
    double wa_coeff;
    double mu;
    int max_life;
    int life;
    double stop_overflow;

    bool use_precond;
    bool skip_update;
    bool enable_skip_update;
    bool enable_fence;
    bool early_stop_check_plateau = true;

    double force_ratio;
    double weighted_weight;

    // quad penalty
    double density_quad_coeff;
    torch::Tensor quad_penalty_coeff;
    torch::Tensor density_weight_grad_precond;
    torch::Tensor init_density;

    void set_init_param(double init_density_weight);
    void push_metric(double hpwl, double overflow, int iteration);
    void update_precond_weight();

    void step(double hpwl, double overflow, torch::Tensor node_pos);
    void step_density_weight();
    void step_wa_coeff();
    void step_precond_coef();
    bool update_best_sol(torch::Tensor sol);
    bool need_to_early_stop();
    bool check_plateau(std::vector<double>& x, int window = 10, double threshold = 0.001);
    bool check_divergence(int window = 50, double threshold = 0.05);
    bool check_stop_hpwl(int window, double threshold);

    tuple<torch::Tensor, double, double, int> get_best_solution();

    vector<double> density_weights;
    vector<double> wa_coeffs;
    torch::Tensor prev_hpwls;
    torch::Tensor prev_overflows;
    torch::Tensor cur_hpwls;
    torch::Tensor cur_overflows;

    torch::Tensor mov_node_weights;
    void steps(torch::Tensor hpwls, torch::Tensor overflows, torch::Tensor node_pos);
    void update_precond_weights();
    void step_density_weights();
    void step_wa_coeffs();

    void visualize();  // TODO: write to file and call py script

    /* node info for ovld constructor */
    torch::Tensor mov_node_to_num_pins;
    torch::Tensor mov_node_area;
    torch::Tensor node_size;
};