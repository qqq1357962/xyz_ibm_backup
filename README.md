# Global Placer

## Build
Install libtorch 1.11.0 + cpu
```
bash install_thirdparty.sh
```

```
mkdir build
cd build 
cmake ../ && cmake --build ./ -j40 && cmake --install .
```

## Download contest benchmarks
```
cd data
python ../benchmarks/download_iccad22.py
```

## Run(2023 version)
```
case2:
./place data/2023_case2.txt output/case2.txt --version 23 --num_threads 8 --gpu 0 --partitioner gp3d --num_bin_x 512 --num_bin_y 512 --num_bin_z 10 --dp true --lg true --pp false --draw_placement true --num_den_layer 3 --force_coeff_2d 0 --round_recursion 1 --omni_int 1 --eval_params 0 --inner_iter 1500 --use_pre_gp 1 --stack_cells 1 --num_bin_3d 53 --cut_net_thres 12 --net_weight_coef 1 --net_weight_offset -0.01 --stop_overflow_3d 0.11 --quad_penalty 1 --wa_coeff 2 --density_weight 16e-5 --density_weight_coef 1.05 --magic_hpwl 70000 --sideline 100 --fp true --local_density_weight true --is_fp_permit_change_cross_chip true --num_part 1 --kernel_size 1

case3:
./place data/ProblemB_case3_20230720.txt output/case3.txt --version 23 --num_threads 8 --gpu 0 --partitioner gp3d --num_bin_x 512 --num_bin_y 512 --num_bin_z 10 --dp true --lg true --pp false --draw_placement true --num_den_layer 3 --force_coeff_2d 0 --round_recursion 1 --omni_int 1 --eval_params 0 --inner_iter 2700 --use_pre_gp 1 --stack_cells 1 --num_bin_3d 53 --cut_net_thres 12 --net_weight_coef 1 --net_weight_offset -0.01 --stop_overflow_3d 0.11 --quad_penalty 1 --wa_coeff 2 --density_weight 16e-4 --density_weight_coef 1.05 --magic_hpwl 35000 --sideline 400 --fp false --min_gp_step 1700 --gp_padding 0.07 --shrink_size 1.0

case4:
./place data/ProblemB_case4_20230816.txt output/case4_0816_35_7_7_7.txt --version 23 --num_threads 8 --gpu 0 --partitioner gp3d --num_bin_x 512 --num_bin_y 1024 --num_bin_z 10 --dp true --lg true --pp false --draw_placement false --num_den_layer 3 --force_coeff_2d 0 --round_recursion 1 --omni_int 1 --eval_params 0 --inner_iter 5000 --use_pre_gp 1 --stack_cells 1 --num_bin_3d 53 --cut_net_thres 12 --net_weight_coef 1 --net_weight_offset -0.01 --stop_overflow_3d 0.07 --quad_penalty 1 --wa_coeff 2 --density_weight 16e-5 --density_weight_coef 1.05 --magic_hpwl 70000 --sideline 2000 --fp false --num_part 1 --kernel_size 1

```



## Run
```
./place --verbose true --load_from_raw true --dataset ispd2005 --num_threads 20 --design_name adaptec1 --inner_iter 5000 --lg true --dp true --target_density 1

./place --verbose true --load_from_raw true --dataset ispd2015_without_fence --num_threads 20 --design_name mgc_des_perf_a --inner_iter 5000

./place --verbose true --load_from_raw true --dataset iccad2019 --num_threads 20 --design_name ispd18_test1 --inner_iter 5000 --num_bin_x 512 --num_bin_y 512

./place --verbose true --load_from_raw true --dataset iccad2019 --num_threads 20 --design_name ispd18_test2 --inner_iter 5000 --num_bin_x 512 --num_bin_y 512

./place "data/cad/iccad2022/Problem B_case1_0506.txt" output/case1.txt
./place "data/cad/iccad2022/Problem B_case2_0516.txt" output/case2.txt
./place "data/cad/iccad2022/Problem B_case3_0516.txt" output/case3.txt
./place "data/cad/iccad2022/Problem B_case4_0524.txt" output/case4.txt


```

#  case2
```
./place "data/Problem B_case2_0516.txt" output/case2.txt --partitioner gp3d --gpu 0 --num_bin_x 256 --num_bin_y 256  --num_bin_z 10 --dp true --lg true --pp true --draw_placement false --cut_net_thres 14 --net_weight_coef 1 --net_weight_offset -0.009 --num_den_layer 3 --num_bin_3d 5 --force_coeff_2d 0 --stop_overflow_3d 0.17 --use_pre_gp false --round_recursion 1 --magic_hpwl 3500 

./place "data/Problem B_case2_hidden.txt" output/case2_hidden.txt --partitioner gp3d --gpu 0 --num_bin_x 256 --num_bin_y 256  --num_bin_z 10 --dp true --lg true --pp true --draw_placement false --cut_net_thres 14 --net_weight_coef 1 --net_weight_offset -0.011 --num_den_layer 2 --num_bin_3d 6 --force_coeff_2d 0 --stop_overflow_3d 0.17 --use_pre_gp false --round_recursion 1 --omni_int 0 --magic_hpwl 3500  --block_row 0 
```

#  case3
```
./place "data/Problem B_case3_0516.txt" output/case3.txt --partitioner gp3d --gpu 0 --num_bin_x 512 --num_bin_y 512 --num_bin_z 10 --dp true --lg true --pp true --draw_placement false --cut_net_thres 10 --net_weight_coef 1 --net_weight_offset -0.1 --num_den_layer 3 --num_bin_3d 25 --force_coeff_2d 0 --stop_overflow_3d 0.13 --use_pre_gp false --round_recursion 1 --omni_int 0 --magic_hpwl 35000 --fmwl_area_coef 1

./place "data/Problem B_case3_hidden.txt" output/case3_hidden.txt --partitioner gp3d --gpu 0 --num_bin_x 512 --num_bin_y 512 --num_bin_z 10 --dp true --lg true --pp true --draw_placement false --cut_net_thres 10 --net_weight_coef 1 --net_weight_offset -0.05 --num_den_layer 3 --num_bin_3d 30 --force_coeff_2d 0 --stop_overflow_3d 0.11 --use_pre_gp false --round_recursion 1 --omni_int 1 --magic_hpwl 35000 --block_row 1
```

#  case4
```
<!-- ./place "data/Problem B_case4_0524.txt" output/case4.txt --partitioner gp3d --gpu 0 --num_bin_x 512 --num_bin_y 512 --num_bin_z 10 --dp true --lg true --pp false --draw_placement false --cut_net_thres 13 --net_weight_coef 1 --net_weight_offset -0.0095 --num_den_layer 3 --num_bin_3d 45 --force_coeff_2d 0 --stop_overflow_3d 0.14 --use_pre_gp false --round_recursion 1 --omni_int 1 --quad_penalty 0 --wa_coeff 4 --density_weight 16e-5 --density_weight_coef 1.04 --magic_hpwl 170000 --eval_params 0 --inner_iter 10000 -->

./place "data/Problem B_case4_0524.txt" output/case4.txt --partitioner gp3d --num_threads 20 --gpu 0 --num_bin_x 512 --num_bin_y 512 --num_bin_z 10 --dp true --lg true --pp false --draw_placement false --cut_net_thres 10 --net_weight_coef 1 --net_weight_offset -0.0085 --num_den_layer 3 --num_bin_3d 50 --force_coeff_2d 0 --stop_overflow_3d 0.11 --use_pre_gp 0 --round_recursion 1 --omni_int 1  --quad_penalty 1 --wa_coeff 2 --density_weight 16e-5 --density_weight_coef 1.06 --magic_hpwl 35000 --eval_params 0 --inner_iter 10000 --stack_cells 1


./place "data/Problem B_case4_hidden.txt" "output/case4_hidden.txt" --partitioner gp3d --gpu 1 --num_bin_x 512 --num_bin_y 512 --num_bin_z 10 --slice_direction 2 --dp true --lg true --pp false --draw_placement false --log_freq 100 --use_filler_3d true --cut_net_thres 15 --net_weight_coef 1 --net_weight_offset -0.0085 --num_den_layer 3 --use_pre_gp true --num_bin_3d 46 --shrink_size 1 --use_filler true --force_coeff_2d 0 --stop_overflow_3d 0.14 --omni_int 1 --round_recursion 1 --quad_penalty 0 --wa_coeff 4 --density_weight 8e-05 --density_weight_coef 1.05 --magic_hpwl 170000 --eval_params false --pt true --inner_iter 10000 --fmwl_iter 1 --fmwl_area_coef 1


```

# Configurations
| Parameter               | Description                          |
--------------------------|--------------------------------------|
|                   |                       |

## Evaluate
```
./evaluator "data/Problem B_case1_0506.txt" output/case1.txt
./evaluator "data/Problem B_case2_0516.txt" output/case2.txt
./evaluator "data/Problem B_case2_hidden.txt" output/case2_hidden.txt

./evaluator "data/Problem B_case3_0516.txt" output/case3.txt
./evaluator "data/Problem B_case3_hidden.txt" output/case3_hidden.txt

./evaluator "data/Problem B_case4_0524.txt" output/case4.txt
./evaluator "data/Problem B_case4_hidden.txt" output/case4_hidden.txt

```

## Submission
Change 
/utils.log.h write_log -> false
main.cpp log_verbose -> false


---------------------------------------------
