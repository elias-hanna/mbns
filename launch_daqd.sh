#!/bin/bash -l

# Things I don't really change
rep=1
ep=5
env_low_limit=-1
env_high_limit=1

# env=redundant_arm_no_walls_limited_angles
env=ball_in_cup
method=random-policies
transfer_selection=all
fitness_func=disagr_minimization

python gym_daqd_main.py --init-method $method --init-episodes $ep --environment $env --grid_shape [$env_low_limit,$env_high_limit]  --rep $rep --transfer-selection $transfer_selection --fitness-func $fitness_func --log_dir data/logs --max_evals 10000 --dump_period 1000 --num_cores 18 --b_size 200 --min-found-model 20
