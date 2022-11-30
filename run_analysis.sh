#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################

#environments=(ball_in_cup redundant_arm_no_walls_limited_angles fastsim_maze fastsim_maze_traps hexapod_omni)
environments=(ball_in_cup redundant_arm_no_walls_limited_angles fastsim_maze fastsim_maze_traps)
# environments=(fastsim_maze fastsim_maze_traps)
# environments=(ball_in_cup)
pred_error_plot_upper_limits=(5 5 100 100) # warning needs to be in same order as envs
disagr_plot_upper_limits=(1 1 5 5) # warning needs to be in same order as envs

# environments=(ball_in_cup)
# environments=(hexapod_omni)
# pred_error_plot_upper_limits=(1) # warning needs to be in same order as envs
# disagr_plot_upper_limits=(1) # warning needs to be in same order as envs

episodes=(20)
#methods=(random-actions brownian-motion colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2 random-policies vanilla no-init)
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

methods=(random-actions brownian-motion colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2 random-policies vanilla no-init) # no-init--perfect-model)
# methods=(no-init)
sup_args=--no-training
# methods=(vanilla)
# transfer_sels=(all disagr disagr_bd random)
# nb_transfers=(10 1)

transfer_sels=(all)
nb_transfers=(10)

# dump_vals=(10 20 30 40 50 60 70 80 90 100)
dump_vals=(100 200 300 400 500 600 700 800 900 1000)
# dump_vals=(100 300 500 700 900 1100)
# dump_vals=(500 1100 1700 2300 2900 3500 4100 4700 5300 5900 6500 7100 7700 8300 8900 9500 10100)
# dump_vals=(10100 20300 30500 40100 49700)
dump_vals=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000)
# dump_vals=(100 200 300 400 500 600 700 800 900 1000 1100 1200)

#rand_budg=${dump_vals[-1]}
rand_budg=10000

## Plot means (only means) over test replications on same plot
cpt=0
for env in "${environments[@]}"; do
    cd ${env}${sup_args}_daqd_results
    echo "Processing following folder"; pwd
    # python ../../vis_repertoire_mis_transfer_sel.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --fitness-funcs ${fitnesses[*]} --transfer-selection ${transfer_sels[*]} --nb-transfer ${nb_transfers[*]} --environment $env --dump-vals ${dump_vals[*]} --dump-path .
    python ~/Documents/thesis/dev/model_init_exps/daqd/vis_repertoire_mis_transfer_sel.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --fitness-funcs ${fitnesses[*]} --transfer-selection ${transfer_sels[*]} --nb-transfer ${nb_transfers[*]} --environment $env --dump-vals ${dump_vals[*]} --dump-path . --random-budget $rand_budg
    cd ..
    cpt=$((cpt+1))
    echo "finished archive analysis for $env"
done
