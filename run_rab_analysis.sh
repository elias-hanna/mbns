#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################

######### Results from cluster ########
reps=10

## Environments
environments=(empty_maze_laser fastsim_maze_laser) # Considered environments
nb_divs=(1) ## not taken into account anymore

## Model architectures
# architectures=(ffnn_2l_10n ffnn_2l_64n ffnn_10l_10n ffnn_10l_64n)
architectures=(rnn_2l_10h_2)

## Search methods
# search_methods=(random-policies det det_ens) # considered search methods
# search_methods=(random-policies_ffnn_2l_10 random-policies_ffnn_2l_64 random-policies_rnn_2l_10 random-policies_rnn_2l_64 ffnn_2l_10_det_ens ffnn_2l_64_det_ens rnn_2l_10_det_ens rnn_2l_64_det_ens) # considered search methods
search_methods=(random-policies_ffnn_2l_10 random-policies_ffnn_2l_64 ffnn_2l_10h_srf_ens ffnn_2l_64h_srf_ens) # considered search methods

# search methods params
ens_sizes=(4) # considered ensemble sizes
m_horizons=(10) # Considered model horizons

## Selection methods
# sel_methods=(random max nov kmeans) # selection methods
sel_methods=(random) # selection methods

## Considered archives sizes 
asize=15100 # saved archive sizes
asize=66500 # saved archive sizes

final_asizes=(5000) # number of random policies
sel_size=4995

n_waypoints=(1) # Number of waypoints for the BD (1 is last traj element)

daqd_folder=~/Documents/thesis/dev/model_init_exps/daqd

## for each environment, executes reps repetition of the given experiment
cpt=0
# for architecture in "${architectures[@]}"; do
    # cd $architecture
for env in "${environments[@]}"; do
    cd ${env}_dab_results
    echo "Processing following folder"; pwd

        for final_asize in "${final_asizes[@]}"; do 
            for nb_div in "${nb_divs[@]}"; do
                python ${daqd_folder}/vis_dab_results.py --nb_div ${nb_div} \
                       --search-methods ${search_methods[*]} -e $env \
                       --m-horizons ${m_horizons[*]} --sel-methods ${sel_methods[*]} \
                       --n-reps $reps --asize $asize --final-asize ${final_asize} \
                       --ens-sizes ${ens_sizes[*]} --n-waypoints ${n_waypoints[*]} \
                       --sel-size $sel_size
            done
        done
    wait
    cd ..
done
    # cd ..
# done
