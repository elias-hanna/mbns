#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################
reps=10

# environments=(empty_maze half_cheetah walker2d)
environments=(empty_maze)
model_types=(det det_ens)
m_horizons=(10 100)

n_waypoints=1

max_evals=10000 ## evals on model
rand_pol_evals=100

daqd_folder=~/Documents/thesis/dev/model_init_exps/daqd

## for each environment, executes reps repetition of the given experiment
cpt=0
for env in "${environments[@]}"; do
    mkdir ${env}_dab_results; cd ${env}_dab_results
    echo "Processing following folder"; pwd

    #### Diverse archive bootstrapping ####
    ## for each model type
    for model_type in "${model_types[@]}"; do
	    ## for each considered model horizon
	    for m_horizon in "${m_horizons[@]}"; do
            mkdir ${model_type}_h${m_horizon}
            cd ${model_type}_h${m_horizon}
	        ## execute one experiment with given index idx
	        for ((idx=0; idx<$reps; idx++)); do
		        mkdir $idx; cd $idx
		        mkdir tmp
		        singularity exec --bind tmp/:/tmp --bind ./:/logs \
                            ~/src/singularity/model_init_study.sif \
                            python ${daqd_folder}/gym_rdds_main.py --log_dir /logs \
                            -e $env --model-horizon ${m_horizon} --model-variant all_dynamics \
                            --n-waypoints ${n_waypoints} --dump_period -1 \
                            --max_evals ${max_evals} --algo ns --model-type $model_type
		        rm -r tmp/
                cd ..
	        done
            cd ..
	    done
    done

    
    #### Random policies archive bootstrapping ####
    mkdir random_policies; cd random_policies
    for ((idx=0; idx<$reps; idx++)); do
	    mkdir $idx; cd $idx
	    mkdir tmp
	    singularity exec --bind tmp/:/tmp --bind ./:/logs \
                    ~/src/singularity/model_init_study.sif \
                    python ${daqd_folder}/gym_rdds_main.py --log_dir /logs \
                    -e $env \
                    --dump_period -1 --max_evals ${rand_pol_evals} --random-policies
	    rm -r tmp/
	    cd ..
    done
    cd ..

    #### end of environment ####
    cd ..
    cpt=$((cpt+1))
    echo "finished experiment for $env"
done
