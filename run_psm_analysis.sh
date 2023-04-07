#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################

######### Results from cluster ########
reps=10

## Environments
environments=(ball_in_cup hexapod_omni fastsim_maze fastsim_maze_traps empty_maze)

environments=(hexapod_omni fastsim_maze)
#environments=(fastsim_maze_traps)

## considered policy search methods
psms=(ns qd_grid qd_unstructured daqd)
psms=(ns qd_grid qd_unstructured)

# psms=(ns qd_grid qd_unstructured)

## Other parameters
# nb_div: grid shape per dim for cov computing
nb_div=20 ## Good for hexapod
nb_div=40 ## Good for mazes

nb_divs=(20 40 40)

daqd_folder=~/Documents/thesis/dev/model_init_exps/daqd

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
