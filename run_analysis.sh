#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################

environments=(ball_in_cup redundant_arm_no_walls_limited_angles fastsim_maze fastsim_maze_traps)
# environments=(ball_in_cup)
pred_error_plot_upper_limits=(5 5 100 100) # warning needs to be in same order as envs
disagr_plot_upper_limits=(1 1 5 5) # warning needs to be in same order as envs

episodes=(20)
methods=(random-actions brownian-motion colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2 random-policies vanilla no-init)
# fitnesses=(energy_minimization disagr_minimization)
fitnesses=(energy_minimization)

## Plot means (only means) over replications on same plot
# cpt=0
# for env in "${environments[@]}"; do
#     cd ${env}_daqd_results
#     echo "Processing following folder"; pwd
#     python ../../vis_repertoire_mis_all.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --fitness-funcs ${fitnesses[*]} --environment $env --dump-path .
#     cd ..
#     cpt=$((cpt+1))
#     echo "finished archive analysis for $env"
# done

# transfer_sels=(all disagr disagr_bd random)
# nb_transfers=(10 1)
# transfer_sels=(random disagr)
transfer_sels=(random)
nb_transfers=(10)
# dumps=(20 40 60 80 100)
dump_vals=(10 20 30 40 50 60 70 80 90 100)

methods=(random-actions brownian-motion colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2 random-policies no-init no-init--perfect-model)
sup_args=--no-training

## Plot means (only means) over test replications on same plot
cpt=0
for env in "${environments[@]}"; do
    cd test_${env}${sup_args}_daqd_results
    echo "Processing following folder"; pwd
    # python ../../vis_repertoire_mis_transfer_sel.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --fitness-funcs ${fitnesses[*]} --transfer-selection ${transfer_sels[*]} --nb-transfer ${nb_transfers[*]} --environment $env --dump-vals ${dump_vals[*]} --dump-path .
    python ../../../vis_repertoire_mis_transfer_sel.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --fitness-funcs ${fitnesses[*]} --transfer-selection ${transfer_sels[*]} --nb-transfer ${nb_transfers[*]} --environment $env --dump-vals ${dump_vals[*]} --dump-path .
    cd ..
    cpt=$((cpt+1))
    echo "finished archive analysis for $env"
done
