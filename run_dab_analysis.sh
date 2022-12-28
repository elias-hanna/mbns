#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################
reps=10

# environments=(empty_maze half_cheetah walker2d) # Considered environments
environments=(walker2d) # Considered environments
model_types=(det det_ens) # Considered model types
m_horizons=(10 100) # Considered model horizons
nb_divs=(10 100 1000)

search_methods=(random-policies det det_ens)
sel_methods=(random max nov kmeans) # Archive Bootstraping methods

asize=10100 # saved archive sizes
final_asize=100 # number of random policies

n_waypoints=1 # Number of waypoints for the BD (1 is last traj element)

daqd_folder=~/Documents/thesis/dev/model_init_exps/daqd

## for each environment, executes reps repetition of the given experiment
cpt=0
for env in "${environments[@]}"; do
    mkdir ${env}_dab_results; cd ${env}_dab_results
    echo "Processing following folder"; pwd

    for nb_div in "${nb_divs[@]}"; do
        python ${daqd_folder}/vis_dab_results.py --nb_div ${nb_div} \
               --search-methods ${search_methods[*]} -e $env \
               --m-horizons ${m_horizons[*]} --sel-methods ${sel_methods[*]} \
               --n-reps $reps --asize $asize --final-asize ${final_asize}
    done
    cd ..
done
