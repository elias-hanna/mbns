#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################

######### Results from cluster ########
reps=10

## Environments
environments=(ball_in_cup fastsim_maze fastsim_maze_traps redundant_arm_no_walls_limited_angles)
environments=(ball_in_cup fastsim_maze fastsim_maze_traps redundant_arm_no_walls_limited_angles)

## considered policy search methods
# inits=(no-init vanilla colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2 random-actions random-policies)
inits=(vanilla colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2 random-actions random-policies)
init_budget=10

## Other parameters
# nb_div: grid shape per dim for cov computing
nb_div=20 ## Good for hexapod, bic, empty
nb_div=40 ## Good for mazes, give good difference between plots on ball in cup but low cov
nb_div=30 ## seems good for hexapod

nb_divs=(40 40 40 40)

daqd_folder=~/src/daqd

div_cpt=0
for env in "${environments[@]}"; do
    cd ${env}_daqd_results
    echo "Processing following folder"; pwd

    python ${daqd_folder}/mis_impact_plot_utils.py --nb-div ${nb_divs[${div_cpt}]} \
           --ps-methods ${inits[*]} -e $env \
           --n-reps $reps --init-budget $init_budget
    
    div_cpt=$((div_cpt+1))
    wait
    cd ..
done
