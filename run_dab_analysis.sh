#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################
reps=10

environments=(empty_maze half_cheetah walker2d) # Considered environments
model_types=(det det_ens) # Considered model types
m_horizons=(10 100) # Considered model horizons

search_method=(random-policies det det_ens)
sel_methods=(nov kmeans) # Archive Bootstraping methods

n_waypoints=1 # Number of waypoints for the BD (1 is last traj element)

a_sizes=10100 # saved archive sizes
rand_pol_evals=100 # number of random policies

daqd_folder=~/Documents/thesis/dev/model_init_exps/daqd

## for each environment, executes reps repetition of the given experiment
cpt=0
for env in "${environments[@]}"; do
    mkdir ${env}_dab_results; cd ${env}_dab_results
    echo "Processing following folder"; pwd

    python ${daqd_folder}/vis_dab_results.py --ab-methods ${ab_methods[*]}
	## execute analysis for environment and given model types and selection methods
	singularity exec --bind tmp/:/tmp --bind ./:/logs \
                ~/src/singularity/model_init_study.sif \
                python ${daqd_folder}/gym_rdds_main.py --log_dir /logs \
                -e $env --model-horizon ${m_horizon} --model-variant all_dynamics \
                --n-waypoints ${n_waypoints} --dump_period -1 \
                --max_evals ${max_evals} --algo ns --model-type det_ens
    cd ..
done
