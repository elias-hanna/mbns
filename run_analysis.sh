#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################

environments=(ball_in_cup redundant_arm_no_walls_limited_angles fastsim_maze fastsim_maze_traps)
pred_error_plot_upper_limits=(5 5 100 100) # warning needs to be in same order as envs
disagr_plot_upper_limits=(1 1 5 5) # warning needs to be in same order as envs

episodes=(20)
methods=(brownian-motion colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2 random-actions random-policies)

# ## Plot means (only means) over replications on same plot
# cpt=0
# for env in "${environments[@]}"; do
#     cd ${env}_daqd_results
#     echo "Processing following folder"; pwd
#     python ../../vis_repertoire_mis_all.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --environment $env --dump-path .
#     cd ..
#     cpt=$((cpt+1))
#     echo "finished archive analysis for $env"
# done

transfer_sels=(disagr disagr_bd)
nb_transfers=(10 1)

## Plot means (only means) over test replications on same plot
cpt=0
for env in "${environments[@]}"; do
    cd test_${env}_daqd_results
    echo "Processing following folder"; pwd
    python ../../vis_repertoire_mis_transfer_sel.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --transfer-selection ${transfer_sels[*]} --nb-transfer ${nb_transfers[*]} --environment $env --dump-path .
    cd ..
    cpt=$((cpt+1))
    echo "finished archive analysis for $env"
done
