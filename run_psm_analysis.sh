#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################

######### Results from cluster ########
reps=10

## Environments
environments=(ball_in_cup hexapod_omni fastsim_maze fastsim_maze_traps empty_maze)
# environments=(fastsim_maze_traps)
environments=(hexapod_omni)

## considered policy search methods
psms=(ns qd_grid qd_unstructured daqd)
psms=(ns mbns_population_novelty qd_grid qd_unstructured qd_unstructured_adaptive daqd daqd_adaptive)
# psms=(ns mbns_population_novelty qd_unstructured_adaptive daqd_adaptive)
psms=(ns mbns_population_novelty qd_grid qd_unstructured daqd)

# psms=(mbns_archive_random mbns_archive_novelty mbns_population_random mbns_population_novelty mbns_test mbns_adaptative_test ns)
# psms=(daqd mbns_population_novelty)

## Other parameters
# nb_div: grid shape per dim for cov computing
nb_div=20 ## Good for hexapod, bic, empty
nb_div=40 ## Good for mazes

nb_divs=(20 20 40)

daqd_folder=~/src/daqd

div_cpt=0
for env in "${environments[@]}"; do
    cd ${env}_results
    echo "Processing following folder"; pwd

    python ${daqd_folder}/plot_utils.py --nb-div ${nb_divs[${div_cpt}]} \
           --ps-methods ${psms[*]} -e $env \
           --n-reps $reps
    
    div_cpt=$((div_cpt+1))
    wait
    cd ..
done
