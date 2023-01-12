#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################

######### Results from cluster ########
reps=10

environments=(empty_maze walker2d redundant_arm) # Considered environments
nb_divs=(1) ## not taken into account anymore

ens_sizes=(4)
search_methods=(random-policies det det_ens) # considered search methods
m_horizons=(10) # Considered model horizons
sel_methods=(random max nov kmeans) # selection methods

asize=15100 # saved archive sizes
asize=211000 # saved archive sizes
final_asizes=(100 200 500 1000) # number of random policies

n_waypoints=(1) # Number of waypoints for the BD (1 is last traj element)

daqd_folder=~/Documents/thesis/dev/model_init_exps/daqd

## for each environment, executes reps repetition of the given experiment
cpt=0
for env in "${environments[@]}"; do
    cd ${env}_dab_results
    echo "Processing following folder"; pwd

    for final_asize in "${final_asizes[@]}"; do 
        for nb_div in "${nb_divs[@]}"; do
            python ${daqd_folder}/vis_dab_results.py --nb_div ${nb_div} \
                   --search-methods ${search_methods[*]} -e $env \
                   --m-horizons ${m_horizons[*]} --sel-methods ${sel_methods[*]} \
                   --n-reps $reps --asize $asize --final-asize ${final_asize} \
                   --ens-sizes ${ens_sizes[*]} --n-waypoints ${n_waypoints[*]}
        done
    done
    wait
    cd ..
done

######### Results run from local pc ##############

# reps=10

# # environments=(empty_maze half_cheetah walker2d) # Considered environments
# environments=(half_cheetah walker2d) # Considered environments
# nb_divs=(10 100 1000)

# search_methods=(random-policies det det_ens) # considered search methods
# # m_horizons=(10 100) # Considered model horizons
# m_horizons=(10) # Considered model horizons
# sel_methods=(random max nov kmeans) # selection methods

# # environments=(half_cheetah) # Considered environments
# # nb_divs=(10 100 1000)

# # search_methods=(random-policies det det_ens) # considered search methods
# # m_horizons=(10) # Considered model horizons
# # sel_methods=(random max nov kmeans) # selection methods

# asize=10100 # saved archive sizes
# final_asize=100 # number of random policies

# n_waypoints=1 # Number of waypoints for the BD (1 is last traj element)

# daqd_folder=~/Documents/thesis/dev/model_init_exps/daqd

# ## for each environment, executes reps repetition of the given experiment
# cpt=0
# for env in "${environments[@]}"; do
#     mkdir ${env}_dab_results; cd ${env}_dab_results
#     echo "Processing following folder"; pwd

#     for nb_div in "${nb_divs[@]}"; do
#         python ${daqd_folder}/vis_dab_results.py --nb_div ${nb_div} \
#                --search-methods ${search_methods[*]} -e $env \
#                --m-horizons ${m_horizons[*]} --sel-methods ${sel_methods[*]} \
#                --n-reps $reps --asize $asize --final-asize ${final_asize} &
#     done
#     wait
#     cd ..
# done
