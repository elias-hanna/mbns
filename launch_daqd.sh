#!/bin/bash -l

# Things I don't really change
rep=1
ep=20
env_low_limit=-1
env_high_limit=1

# env=redundant_arm_no_walls_limited_angles
env=fastsim_maze
method=brownian-motion
transfer_selection=disagr_bd
fitness_func=energy_minimization

python gym_daqd_main.py --init-method $method --init-episodes $ep --environment $env --grid_shape [$env_low_limit,$env_high_limit]  --rep $rep --transfer-selection $transfer_selection --fitness-func $fitness_func --log_dir data/logs --max_evals 1000 --dump_period 100 --num_cores 18 --b_size 10
