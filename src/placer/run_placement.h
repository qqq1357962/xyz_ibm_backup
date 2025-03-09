#pragma once

struct states;

#include "calculator.h"
#include "core/core.h"
#include "dp/dp.h"
#include "evaluator.h"
#include "gr/ViaData.h"
#include "initializer.h"
#include "optim/Nesterov.h"
#include "optim/ParamScheduler.h"
#include "post_process.h"
#include "pt/partition.h"
#include "visualization.h"

struct states {
    long hpwls_gp = 0, hpwls_lg = 0, hpwls_dp = 0, hpwls_pp = 0, hpwls_raw = 0;
    vector<long> hpwls{vector<long>(10)};

    int hpwl_idx = 0;

    void log() {
        logger.info("\033[1;34mhpwl %ld => %ld (imp. %g%%)\033[0m", hpwls[hpwl_idx - 2], hpwls[hpwl_idx - 1],
                    (1.0 - hpwls[hpwl_idx - 1] / (double)hpwls[hpwl_idx - 2]) * 100);
    }
};

void run_placement_main_nesterov();
void run_placement_main_multi_circuit();
void run_patoh();

torch::Tensor run_gp(NodeData& data, ViaData& via_data, torch::Tensor node_pos, states& hpwl_state, 
                     int& cell_mov_lhs,int& cell_mov_rhs, int& via_mov_lhs, int& via_mov_rhs, 
                     int& mov_lhs, int& mov_rhs, bool move_macro=true, bool is_init_macro=true,
                     bool is_init_stdcell=true,string message="");

void run_greedy_place(NodeData& data);
void run_greedy_place_for_fp(NodeData& data);

torch::Tensor run_gp_refine(NodeData& data, ViaData& via_data, torch::Tensor node_pos, states& hpwl_state,
                            int& cell_mov_lhs, int& cell_mov_rhs, int& via_mov_lhs, int& via_mov_rhs, int& mov_lhs,
                            int& mov_rhs);

torch::Tensor run_lg(NodeData& data, ViaData& via_data, torch::Tensor node_pos, states& hpwl_state, int& cell_mov_lhs,
                     int& cell_mov_rhs, int& via_mov_lhs, int& via_mov_rhs, int& mov_lhs, int& mov_rhs, bool only_macro);

torch::Tensor run_floorplan(NodeData& data, torch::Tensor node_pos, states& hpwl_state, int& cell_mov_lhs,
                     int& cell_mov_rhs);

torch::Tensor run_gp_swap(NodeData& data, ViaData& via_data, torch::Tensor node_pos, states& hpwl_state,
                          int& cell_mov_lhs, int& cell_mov_rhs, int& via_mov_lhs, int& via_mov_rhs, int& mov_lhs,
                          int& mov_rhs);