#pragma once
#include "global.h"
#include "fp/db/Database.h"

#include "fp_pack.h"

namespace fp {

class Floorplan {
public:
    Floorplan(fp::Database* database_, double alpha,
                const std::string& sa_mode, bool is_verbose,
                const string fp_path = " ");

    fp::Database* database;

    string fp_path_;

    //const Packer& best_floorplan() const;
    Packer best_floorplan_;

    void write(const string& output_path);
    void init();
    void Run();

private:
    void SA();
    void FastSA();

    double Calc_Cost_Naive(const Packer& floorplan) const;
    double Calc_Cost(const Packer& floorplan, double alpha,
                        double beta) const;

    double alpha_;
    double beta_;
    double lambda_;
    double balance_ = 0.5;

    bool flag_ = true;

    std::string sa_mode_;
    bool is_verbose_;
    double min_area_;
    double max_area_;
    double min_wirelength_;
    double max_wirelength_;
    double average_area_;
    double average_wirelength_;
    double average_uphill_cost_;
    Packer best_floorplan_invalid_;
};

}