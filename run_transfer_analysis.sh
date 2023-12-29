#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################

######### Results from cluster ########
reps=10

## Environments
environments=(ball_in_cup hexapod_omni empty_maze)

environments=(empty_maze)
# environments=(ball_in_cup)
# environments=(hexapod_omni)
# environments=(hexapod_omni empty_maze ball_in_cup)

## considered policy search methods
psms=(mbns_average_nov_novelty)
# psms=(mbns_population_novelty)

## Other parameters
daqd_folder=~/src/daqd

div_cpt=0
for env in "${environments[@]}"; do
    # cd ${env}_mbns_average_nov_novelty_results
    echo "Processing following folder"; pwd

    python ${daqd_folder}/plot_utils.py \
           --ps-methods ${psms[*]} -e $env \
           --n-reps $reps --only-transfer
    div_cpt=$((div_cpt+1))
    wait
    cd ..
done
