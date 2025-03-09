#include "floorplan.h"
using namespace fp;


//---------------------------------------------------------------------

void Floorplan::SA() {
    const double P = 0.95;
    const double r = 0.9;
    const double init_T =
            average_uphill_cost_ * database->nMacros / (-1 * log(P));
    const int N = database->nMacros * 20;
    logger.log() << "Init temperature:  " << init_T << endl;
    // exit(1);

    const double stop_T = init_T / 1.0e15;
    const double stop_accept_rate = 0.01;

    const double alpha = alpha_;
    const double beta = (1 - alpha);
    const double adaptive_alpha_base = alpha;
    const double adaptive_beta_base = beta;


    Packer floorplan(best_floorplan_);
    floorplan.PackInt();

    double adaptive_alpha = adaptive_alpha_base;
    double adaptive_beta = adaptive_beta_base;
    double best_cost = average_uphill_cost_;
    // double best_cost_invalid = best_cost;
    double last_cost = best_cost;
    double T = init_T;
    int num_iteration = 0;
    utils::timer runtime;
    double run_time = 0;
    // 1. Outter loop: T
    // while (run_time < 300) {
    while (T > stop_T) {
        int num_up_hill = 0;
        int num_M = 0;
        num_iteration++;

        int num_accepted_floorplans = 0;
        int num_feasible_floorplans = 0;
        
        // for (int i = 0; i < num_perturbations; ++i) {
        while (num_up_hill <= N && num_M <= 2 * N) {
            // 1. rotate
            // 2. swap
            // 3. delete and insert
            num_M++;
            Packer new_floorplan(floorplan);
            new_floorplan.Perturb();
            new_floorplan.PackInt();
      

            const double cost =
                Calc_Cost_Naive(new_floorplan);
            const double delta_cost = cost - last_cost;
            const double p = exp(-1 * delta_cost / T);
            // logger.log() << delta_cost << endl;
            if ((delta_cost <= 0) ||
                    (rand() / static_cast<double>(RAND_MAX) < p)) 
            {   
                if (delta_cost > 0)
                    num_up_hill++;
                num_accepted_floorplans++;
                floorplan = new_floorplan;
                last_cost = cost;

                const double outline_width = database->outline_width;
                const double outline_height = database->outline_height;
                if (new_floorplan.width1() <= outline_width &&
                    new_floorplan.height1() <= outline_height &&
                    new_floorplan.width2() <= outline_width &&
                    new_floorplan.height2() <= outline_height) 
                {
                    num_feasible_floorplans++;
                    if (cost < best_cost) {
                        best_floorplan_ = new_floorplan;
                        best_cost = cost;
                    }
                }
                else {
                    if (cost < best_cost) {
                        best_floorplan_invalid_ = new_floorplan;
                        best_cost = cost;
                    }
                }
            } 
        } //END FOR
        double run_time = runtime.elapsed();
        cout << "========== total time: " << run_time << " s ==========\n";
        if (is_verbose_) {
            logger.log() << "--------------- " << num_iteration << " ---------------" << endl;
            logger.log() << "    Moves:    " << num_M << endl;
            logger.log() << "    T:    " << T << endl;
            logger.log() << "    num_accepted_floorplans:  " << num_accepted_floorplans
                << endl;
            logger.log() << "    num_feasible_floorplans:  " << num_feasible_floorplans
                << endl;
            logger.log() << "    num_perturbations:    " << N << endl;
            logger.log() << "    adaptive_alpha: " << adaptive_alpha
                << "\tadaptive_beta: " << adaptive_beta << endl;
            logger.log() << "    Best area: " << best_floorplan_.area()
                << "\tBest wirelength: " << best_floorplan_.wirelength() << endl;
        }

        T *= r;
    }

    if (best_floorplan_.area() == 0) {
        logger.log() << "No feasible solution\n";
        logger.log() << fp_path_ << endl;
        best_floorplan_invalid_.PackInt();
        best_floorplan_invalid_.write(fp_path_);
    }
    else{
        best_floorplan_.PackInt();
        best_floorplan_.write(fp_path_);
    }
} //END MODULE

//---------------------------------------------------------------------

void Floorplan::FastSA() {
    const double P = 0.99;
    const double r = 0.9;
    const double init_T =
            average_uphill_cost_ / (-1 * log(P)) * lambda_;

    logger.log() << "Init cost:  " << average_uphill_cost_ << endl;
    logger.log() << "Init temperature:  " << init_T << endl;

    // TODO: solution space
    const int num_perturbations = 50 * database->nMacros;

    const double stop_T = init_T / 1.0e15;
    
    double adaptive_alpha_base = alpha_;
    double adaptive_beta_base = beta_;
    const int c = 100;
    const int k = 15;

    Packer floorplan(best_floorplan_);
    floorplan.PackInt();

    double adaptive_alpha = adaptive_alpha_base * balance_;
    double adaptive_beta = adaptive_beta_base * balance_;
    double best_cost = Calc_Cost(floorplan, adaptive_alpha, adaptive_beta);
    double last_cost = best_cost;
    double T = init_T;
    double p ;
    int num_iteration = 1;

    utils::timer runtime;
    double pack_time = 0;
    double pert_time = 0;
    double calc_time = 0;
    double anne_time = 0;
    
    double last_time = 0;
    double run_time = 0;
    // 1. Outter loop: T
    while (T > stop_T) {
         if (num_iteration >= 2 && num_iteration <= k) {
            logger.log() << "--------------- Greedy Iter: " << num_iteration << " ---------------" << endl;
        } else if(num_iteration > k) {
            logger.log() << "------- Hill Clibmbing Iter: " << num_iteration << " ---------------" << endl;
        } else {
            logger.log() << "----- High Temp Search Iter: " << num_iteration << " ---------------" << endl;
        }

        int num_M = 0;
        double total_delta_cost = 0.0;
        int num_accepted_floorplans = 0;
        int num_feasible_floorplans = 0;

        // adapt new values
        last_cost = Calc_Cost(floorplan, adaptive_alpha, adaptive_beta);

        // 2. Inner loop: solution space
        for (int i = 0; i < num_perturbations; ++i) {
            // 1. rotate
            // 2. swap
            // 3. delete and insert
            num_M++;
            Packer new_floorplan(floorplan);

            last_time = runtime.elapsed();
            new_floorplan.Perturb();
            run_time = runtime.elapsed();
            pert_time += (run_time - last_time);

            last_time = runtime.elapsed();
            new_floorplan.PackInt();
            run_time = runtime.elapsed();
            pack_time += (run_time - last_time);

            last_time = runtime.elapsed();
            const double cost =
                Calc_Cost(new_floorplan, adaptive_alpha, adaptive_beta);
            const double delta_cost = cost - last_cost;
            run_time = runtime.elapsed();
            calc_time += (run_time - last_time);

            last_time = runtime.elapsed();
            p = exp(-1 * delta_cost / T);
            if ((delta_cost < 0) || (rand() / static_cast<double>(RAND_MAX) < p)) {   
                total_delta_cost += delta_cost;
                num_accepted_floorplans++;
                floorplan = new_floorplan;
                last_cost = cost;

                const double outline_width = database->outline_width;
                const double outline_height = database->outline_height;
                if (new_floorplan.width1() <= outline_width &&
                    new_floorplan.height1() <= outline_height &&
                    new_floorplan.width2() <= outline_width &&
                    new_floorplan.height2() <= outline_height) {
                    num_feasible_floorplans++;
                    if (cost < best_cost) {
                        best_floorplan_ = new_floorplan;
                        best_cost = cost;
                    }
                } else {
                    if (cost < best_cost) {
                        best_floorplan_invalid_ = new_floorplan;
                        best_cost = cost;
                    }
                }
            }
            run_time = runtime.elapsed();
            anne_time += (run_time - last_time);
        } //END FOR
        logger.log() << "Best wirelength: " << best_floorplan_.wirelength() << endl;
        // if (is_verbose_) {
            logger.log() << "    Moves:    " << num_M << endl;
            logger.log() << "    T:    " << T << endl;
            logger.log() << "    stop T:    " << stop_T << endl;
            logger.log() << "    total_delta_cost: " << total_delta_cost << endl;
            // logger.log() << "    cost coef: " << abs(total_delta_cost / num_perturbations) << endl;
            logger.log() << "    num_accepted_floorplans:  " << num_accepted_floorplans
                << endl;
            logger.log() << "    num_feasible_floorplans:  " << num_feasible_floorplans
                << endl;
            logger.log() << "    lambda:    " << lambda_ << endl;
            logger.log() << "    adaptive_alpha: " << adaptive_alpha << "\tadaptive_beta: " << adaptive_beta << endl;
            logger.log() << "    Best area: " << best_floorplan_.area() << "\tBest wirelength: " << best_floorplan_.wirelength() << endl;
            logger.log() << "invalid Best area: " << best_floorplan_invalid_.area() << endl;
            logger.log() << "invalid Best wirelength: " << best_floorplan_invalid_.wirelength() << endl;
        // }


        // if (best_floorplan_.wirelength() != 0) {
        //     adaptive_alpha_base = alpha_ * exp(-(double)num_iteration/k);
        //     adaptive_beta_base = 1 - adaptive_alpha_base;

        //     adaptive_alpha = adaptive_alpha_base * balance_;
        //     adaptive_beta = adaptive_beta_base * balance_;
        // }

        // calculate temperature for next iteration
        num_iteration++;
        double sol_coef = (num_feasible_floorplans 
                    / static_cast<double>(num_perturbations) );
        const double average_delta_cost = abs(total_delta_cost / num_perturbations);
        logger.log() << "average_delta_cost: " << average_delta_cost << endl;
        if (num_iteration > 1 && num_iteration <= k) {
            T = init_T * average_delta_cost / (num_iteration * c);
        } else {
            T = init_T * average_delta_cost / num_iteration;
        }
        
    } //END WHILE
    
    // swap(alpha_, beta_);
    // alpha_ /= 2;
    // beta_ = 1 - alpha_;

    logger.log() << "========== runtime breakdown ==========\n";
    logger.log() << "    Pack:    " << pack_time << " s" << endl;
    logger.log() << "    Perturb:    " << pert_time << " s" << endl;
    logger.log() << "    Calc:    " << calc_time << " s" << endl;
    logger.log() << "    Annealing:    " << anne_time << " s" << endl;

    if (best_floorplan_.wirelength() == 0) {
        logger.log() << "No feasible solution\n";
        logger.log() << fp_path_ << endl;
        best_floorplan_invalid_.PackInt();
        best_floorplan_invalid_.write(fp_path_);
    }
    else{
        best_floorplan_.PackInt();
        best_floorplan_.write(fp_path_);
        
        for(int i=0;i<database->nMacros;i++)
        {
            logger.info("%d, %d, %d", i, 
            best_floorplan_.macro_bounding_box_by_id_[i].first.x(),
            best_floorplan_.macro_bounding_box_by_id_[i].first.y());
        }
    }
} //END MODULE

//---------------------------------------------------------------------

void Floorplan::Run() {
    if (sa_mode_ == "sa") {
        SA();
    } else if (sa_mode_ == "fast") {
        utils::timer runtime;
        double init_time = 0;
        init();
        init_time = runtime.elapsed();
        int nth_fsa = 0;
        while (true){
            nth_fsa++;
            FastSA();
            if(best_floorplan_.wirelength() > 0.0)
            {
                break;
            }
            if(nth_fsa>=5)
            {
                break;
            }
        }
        logger.log() << "    Init:     " << init_time << " s\n";
    } 
} //END MODULE

//---------------------------------------------------------------------
