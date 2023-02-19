#----------Algo imports--------#
from src.map_elites import common as cm
from src.map_elites import unstructured_container, cvt
from src.map_elites.qd import QD
from src.map_elites.ns import NS

#----------controller imports--------#
from model_init_study.controller.nn_controller \
    import NeuralNetworkController
from exps_utils import RNNController

#----------Environment imports--------#
import gym
from exps_utils import get_env_params, process_args
from exps_utils import WrappedEnv

#----------Data manipulation imports--------#
import numpy as np
import copy
import pandas as pd
import itertools
#----------Utils imports--------#
import os, sys
import argparse
import matplotlib.pyplot as plt

import random

import time
import tqdm

################################################################################
################################### MAIN #######################################
################################################################################
def main(args):
    bootstrap_archive = None
    px = \
    {
        # type of qd 'unstructured, grid, cvt'
        "type": args.qd_type,
        # arg for NS
        "pop_size": args.pop_size,
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        "cvt_use_cache": True,
        # we evaluate in batches to parallelize
        # "batch_size": args.b_size,
        "batch_size": args.pop_size*2,
        # proportion of total number of niches to be filled before starting
        "random_init": 0.005,  
        # batch for random initialization
        "random_init_batch": args.random_init_batch,
        # path of bootstrap archive
        'bootstrap_archive': bootstrap_archive,
        # when to write results (one generation = one batch)
        "dump_period": args.dump_period,
        # when to write results (budget = dump when dump period budget exhausted,
        # gen = dump at each generation)
        "dump_mode": args.dump_mode,

        # do we use several cores?
        "parallel": args.parallel,
        # min/max of genotype parameters - check mutation operators too
        # "min": 0.0,
        # "max": 1.0,
        "min": -5 if args.environment != 'hexapod_omni' else 0.0,
        "max": 5 if args.environment != 'hexapod_omni' else 1.0,
        
        #------------MUTATION PARAMS---------#
        # selector ["uniform", "random_search"]
        "selector" : args.selector,
        # mutation operator ["iso_dd", "polynomial", "sbx"]
        "mutation" : args.mutation,
    
        # probability of mutating each number in the genotype
        "mutation_prob": 0.2,

        # param for 'polynomial' mutation for variation operator
        "eta_m": 10.0,
        
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2,

        #--------UNSTURCTURED ARCHIVE PARAMS----#
        # l value - should be smaller if you want more individuals in the archive
        # - solutions will be closer to each other if this value is smaller.
        "nov_l": 0.015,
        # "nov_l": 1.5,
        "eps": 0.1, # usually 10%
        "k": 15,  # from novelty search
        "lambda": args.lambda_add, # For fixed ind add during runs (Gomes 2015)
        "arch_sel": args.arch_sel, # random, novelty

        #--------MODEL BASED PARAMS-------#
        "t_nov": 0.03,
        "t_qua": 0.0, 
        "k_model": 15,
        # Comments on model parameters:
        # t_nov is correlated to the nov_l value in the unstructured archive
        # If it is smaller than the nov_l value, we are giving the model more chances which might be more wasteful 
        # If it is larger than the nov_l value, we are imposing that the model must predict something more novel than we would normally have before even trying it out
        # fitness is always positive - so t_qua

        ## model parameters
        'model_type': "det", # always nn or whatever it is not used
        "model_variant": "dynamics", # always dynamics for real envs  
        "perfect_model_on": False,
        "ensemble_size": 1,

        #--------LOG/DUMP-------#
        "log_model_stats": False,
        "log_time_stats": False, 
        "log_ind_trajs": args.log_ind_trajs,
        "dump_ind_trajs": args.dump_ind_trajs,

        
        "norm_bd": False, # whatever value, it is not used
        "nov_ens": "sum", # whatever value, it is not used here
        # 0 for random emiiter, 1 for optimizing emitter
        # 2 for random walk emitter, 3 for model disagreement emitter
        "emitter_selection": 0,

        #--------EVAL FUNCTORS-------#
        "f_target": None,
        "f_training": None,

        #--------EXPS FLAGS-------#
        "include_target": False,
        
        "env_name": args.environment,
        ## for dump
        "ensemble_dump": False,
    }

    if args.algo == 'ns':
        px['type'] = 'fixed'
        
    #########################################################################
    ####################### Preparation of run ##############################
    #########################################################################
    
    ##TODO##
    env_params = get_env_params(args)
    nb_div = 50
    
    is_local_env = env_params['is_local_env'] 
    gym_args = env_params['gym_args']  
    env_register_id = env_params['env_register_id']
    a_min = env_params['a_min'] 
    a_max = env_params['a_max'] 
    ss_min = env_params['ss_min']
    ss_max = env_params['ss_max']
    init_obs = env_params['init_obs'] 
    state_dim = env_params['state_dim']
    obs_min = env_params['obs_min']
    obs_max = env_params['obs_max']
    dim_map = env_params['dim_map']
    bd_inds = env_params['bd_inds']
    nov_l = (1/100)*(np.max(ss_max[bd_inds]) - np.min(ss_min[bd_inds]))# 1% of BD space (maximum 100^bd_space_dim inds in archive)
    px['nov_l'] = nov_l
    print(f'INFO: nov_l param set to {nov_l} for environment {args.environment}')

    ## Get the environment task horizon, observation and action space dimensions
    if not is_local_env:
        gym_env = gym.make(env_register_id, **gym_args)

        try:
            max_step = gym_env._max_episode_steps
        except:
            try:
                max_step = gym_env.max_steps
            except:
                raise AttributeError("Env doesnt allow access to _max_episode_steps" \
                                     "or to max_steps")

        obs = gym_env.reset()
        if isinstance(obs, dict):
            obs_dim = gym_env.observation_space['observation'].shape[0]
        else:
            obs_dim = gym_env.observation_space.shape[0]
        act_dim = gym_env.action_space.shape[0]
    else:
        gym_env = None

    
    n_waypoints = args.n_waypoints
    dim_map *= n_waypoints
    px['dim_map'] = dim_map

    ## Set the type of controller we use
    if args.c_type == 'ffnn':
        controller_type = NeuralNetworkController
    elif args.c_type == 'rnn':
        controller_type = RNNController

    ## Controller parameters
    controller_params = \
    {
        'controller_input_dim': obs_dim,
        'controller_output_dim': act_dim,
        'n_hidden_layers': args.c_n_layers,
        'n_neurons_per_hidden': args.c_n_neurons,
        'time_open_loop': args.open_loop_control,
        'norm_input': args.norm_controller_input,
        'pred_mode': args.pred_mode,
    }
    ## Dynamics model parameters
    dynamics_model_params = \
    {
        'obs_dim': state_dim,
        'action_dim': act_dim,
        'layer_size': [500, 400],
        # 'layer_size': 500,
        'batch_size': 512,
        'learning_rate': 1e-3,
        'train_unique_trans': False,
        'model_type': "det",
        'model_horizon': max_step,
        'ensemble_size': 1,
    }
    observation_model_params = {}
    ## Surrogate model parameters 
    surrogate_model_params = \
    {
        'bd_dim': dim_map,
        'obs_dim': obs_dim,
        'action_dim': act_dim,
        'layer_size': 64,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'train_unique_trans': False,
    }
    ## General parameters
    params = \
    {
        ## general parameters
        'state_dim': state_dim,
        'obs_dim': obs_dim,
        'action_dim': act_dim,
        'dynamics_model_params': dynamics_model_params,
        'observation_model_params': observation_model_params,
        ## controller parameters
        'controller_type': controller_type,
        'controller_params': controller_params,
        'time_open_loop': controller_params['time_open_loop'],

        ## state-action space params
        'action_min': a_min,
        'action_max': a_max,

        'state_min': ss_min,
        'state_max': ss_max,

        'obs_min': obs_min,
        'obs_max': obs_max,
        'init_obs': init_obs,
        
        'clip_obs': False, # clip models predictions 
        'clip_state': False, # clip models predictions 
        ## env parameters
        'env': gym_env,
        'env_name': args.environment,
        'env_max_h': max_step,
        'use_obs_model': False,
        ## algo parameters
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,

        'fitness_func': args.fitness_func,
        'n_waypoints': n_waypoints,
        'num_cores': args.num_cores,
        'dim_map': dim_map,

        ## pretraining parameters
        'pretrain': False,
        ## srf parameters
        'srf_var': 0.001,
        'srf_cor': 0.01,

        ## Dump params/ memory gestion params
        "log_ind_trajs": args.log_ind_trajs,
    }
    px['dab_params'] = params
    px['min'] = params['policy_param_init_min']
    px['max'] = params['policy_param_init_max']
    ## Correct obs dim for controller if open looping on time
    if params['time_open_loop'] == True:
        controller_params['obs_dim'] = 1
    if init_obs is not None:
        params['init_obs'] = init_obs
    #########################################################################
    ####################### End of Preparation of run #######################
    #########################################################################

    if not is_local_env:
        env = WrappedEnv(params)
        dim_x = env.policy_representation_dim
    init_obs = params['init_obs']
    px['dim_x'] = dim_x

    if args.environment == 'hexapod_omni':
        env = HexapodEnv(dynamics_model=dynamics_model,
                         render=False,
                         record_state_action=True,
                         ctrl_freq=100)

    ## Define f_real and f_model
    f_real = env.evaluate_solution # maybe move f_real and f_model inside
    
    if args.algo == 'qd':
        algo = QD(dim_map, dim_x,
                f_real,
                n_niches=1000,
                params=px,
                log_dir=args.log_dir)

    elif args.algo == 'ns':
        algo = NS(dim_map, dim_x,
                  f_real,
                  params=px,
                  log_dir=args.log_dir)
    
    archive, n_evals = algo.compute(num_cores_set=args.num_cores,
                                    max_evals=args.max_evals)
    cm.save_archive(archive, f"{n_evals}_real_all", px, args.log_dir)
        
    ## Plot archive trajectories on real system
    if args.log_ind_trajs:
        ## Extract real sys BD data from s_list
        real_bd_traj_data = [s.obs_traj for s in archive]
        ## Format the bd data to plot with labels
        all_bd_traj_data = []

        all_bd_traj_data.append((real_bd_traj_data, 'real system'))
        ## Plot real archive and model(s) archive on plot
        total_plots = len(all_bd_traj_data)
        ## make it as square as possible
        rows = cols = round(np.sqrt(total_plots))
        ## Add a row in case closest square cannot take all plots in
        if total_plots > rows*cols:
            rows += 1

        if dim_map == 3:
            fig1 = plt.figure()
            fig2 = plt.figure()
            axs1 = []
            axs2 = []
            plt_num = 0
            #for plt_num in range(total_plots):
            for col in range(cols):
                axs1_cols = []
                axs2_cols = []
                for row in range(rows):
                    axs1_cols.append(fig1.add_subplot(cols, rows, plt_num+1, projection='3d'))
                    axs2_cols.append(fig2.add_subplot(cols, rows, plt_num+1, projection='3d'))
                    plt_num += 1
                axs1.append(axs1_cols)
                axs2.append(axs2_cols)
        else:
            fig1, axs1 = plt.subplots(rows, cols)
            fig2, axs2 = plt.subplots(rows, cols)
        m_cpt = 0
        
        for col in range(cols):
            for row in range(rows):
                try:
                    if not hasattr(axs1, '__len__'):
                    # if not 'ens' in args.model_type or args.perfect_model:
                        ax1 = axs1
                        ax2 = axs2
                    else:
                        ax1 = axs1[row][col]
                        ax2 = axs2[row][col]

                    loc_bd_traj_data, loc_system_name = all_bd_traj_data[m_cpt]

                    ## format to take care of trajs that end before max_step + 1
                    # (+1 because we store max_step transitions)
                    formatted_loc_bd_traj_data = []
                    traj_dim = loc_bd_traj_data[0].shape[1] # get traj dim
                    for loc_bd_traj in loc_bd_traj_data:
                        formatted_loc_bd_traj = np.empty((max_step+1,traj_dim))
                        formatted_loc_bd_traj[:] = np.nan
                        formatted_loc_bd_traj[:len(loc_bd_traj)] = loc_bd_traj
                        formatted_loc_bd_traj_data.append(formatted_loc_bd_traj)
                    loc_bd_traj_data = formatted_loc_bd_traj_data
                    loc_bd_traj_data = np.array(loc_bd_traj_data)
                    
                    ## Plot BDs
                    if dim_map == 3:
                        for loc_bd_traj in loc_bd_traj_data:
                            last_ind = (~np.isnan(loc_bd_traj)).cumsum(0).argmax(0)[0]
                            ax1.scatter(xs=loc_bd_traj[last_ind,0],
                                        ys=loc_bd_traj[last_ind,1],
                                        zs=loc_bd_traj[last_ind,2], s=3, alpha=0.1)
                        ax1.scatter(xs=init_obs[bd_inds[0]],ys=init_obs[bd_inds[1]],zs=init_obs[bd_inds[2]], s=10, c='red')
                        ## Plot trajectories
                        for bd_traj in loc_bd_traj_data:
                            ax2.plot(bd_traj[:,bd_inds[0]], bd_traj[:,bd_inds[1]], bd_traj[:,bd_inds[2]], alpha=0.1, markersize=1)
                    else:
                        for loc_bd_traj in loc_bd_traj_data:
                            last_ind = (~np.isnan(loc_bd_traj)).cumsum(0).argmax(0)[0]
                            ax1.scatter(x=loc_bd_traj[last_ind,bd_inds[0]],
                                        y=loc_bd_traj[last_ind,bd_inds[1]], s=3, alpha=0.1)
                        ax1.scatter(x=init_obs[bd_inds[0]],y=init_obs[bd_inds[1]], s=10, c='red')
                        ## Plot trajectories
                        for bd_traj in loc_bd_traj_data:
                            ax2.plot(bd_traj[:,bd_inds[0]], bd_traj[:,bd_inds[1]], alpha=0.1, markersize=1)
                    ax1.set_xlabel('x-axis')
                    ax1.set_ylabel('y-axis')
                    ax1.set_title(f'Archive coverage on {loc_system_name}')
                    if 'real' in loc_system_name:
                        ax1.set_xlim(ss_min[bd_inds[0]], ss_max[bd_inds[0]])
                        ax1.set_ylim(ss_min[bd_inds[1]], ss_max[bd_inds[1]])
                        if dim_map == 3:
                            ax1.set_ylabel('z-axis')
                            ax1.set_zlim(ss_min[bd_inds[2]], ss_max[bd_inds[2]])
                    else:
                        loc_bd_mins = np.min(loc_bd_traj_data[:,-1,:], axis=0) 
                        loc_bd_maxs = np.max(loc_bd_traj_data[:,-1,:], axis=0) 
                        ax1.set_xlim(loc_bd_mins[0], loc_bd_maxs[0])
                        ax1.set_ylim(loc_bd_mins[1], loc_bd_maxs[1])
                    if dim_map != 3:
                        ax1.set_aspect('equal', adjustable='box')
                    
                    ax2.set_xlabel('x-axis')
                    ax2.set_ylabel('y-axis')
                    ax2.set_title(f'Individuals trajectories on {loc_system_name}')
                    if 'real' in loc_system_name:
                        ax2.set_xlim(ss_min[bd_inds[0]], ss_max[bd_inds[0]])
                        ax2.set_ylim(ss_min[bd_inds[1]], ss_max[bd_inds[1]])
                        if dim_map == 3:
                            ax2.set_ylabel('z-axis')
                            ax2.set_zlim(ss_min[bd_inds[2]], ss_max[bd_inds[2]])
                    else:
                        loc_bd_mins = np.min(loc_bd_traj_data, axis=(0,1)) 
                        loc_bd_maxs = np.max(loc_bd_traj_data, axis=(0,1))
                        ax2.set_xlim(loc_bd_mins[0], loc_bd_maxs[0])
                        ax2.set_ylim(loc_bd_mins[1], loc_bd_maxs[1])
                    if dim_map != 3:
                        ax2.set_aspect('equal', adjustable='box')
                    m_cpt += 1
                except:
                    fig1.delaxes(axs1[row][col])
                    fig2.delaxes(axs2[row][col])


        fig1.set_size_inches(total_plots*2,total_plots*2)
        fig1.suptitle('Archive coverage after DAB')
        file_path = os.path.join(args.log_dir, f"{args.environment}_real_cov")
        fig1.savefig(file_path,
                    dpi=300, bbox_inches='tight')

        fig2.set_size_inches(total_plots*2,total_plots*2)
        fig2.suptitle('Individuals trajectories after DAB')
        file_path = os.path.join(args.log_dir, f"{args.environment}_ind_trajs")
        fig2.savefig(file_path,
                    dpi=300, bbox_inches='tight')


    ## Plot archive coverage evolution at each generation
    # we can do that by looking at coverage of individuals in archive taking
    # into account the first lambda, then adding to these the next lambda inds
    # etc...
    def get_data_bins(data, ss_min, ss_max, dim_map, bd_inds, nb_div):
        df_min = data.iloc[0].copy(); df_max = data.iloc[0].copy()
        
        for i in range(dim_map-1, dim_map-3, -1):
            df_min[f'bd{i}'] = ss_min[bd_inds[i%len(bd_inds)]]
            df_max[f'bd{i}'] = ss_max[bd_inds[i%len(bd_inds)]]
        ## Deprecated but oh well
        data = data.append(df_min, ignore_index = True)
        data = data.append(df_max, ignore_index = True)

        for i in range(dim_map-1, dim_map-3, -1):
            data[f'{i}_bin'] = pd.cut(x = data[f'bd{i}'],
                                      bins = nb_div, 
                                      labels = [p for p in range(nb_div)])

        ## assign data to bins
        data = data.assign(bins=pd.Categorical
                           (data.filter(regex='_bin')
                            .apply(tuple, 1)))

        ## remove df min and df max
        data.drop(data.tail(2).index,inplace=True)

        return data

    def compute_cov(data, ss_min, ss_max, dim_map, bd_inds, nb_div):
        ## add bins field to data
        data = get_data_bins(data, ss_min, ss_max, dim_map, bd_inds, nb_div)
        ## count number of bins filled
        counts = data['bins'].value_counts()
        total_bins = nb_div**(dim_map//args.n_waypoints)
        ## return coverage (number of bins filled)
        return len(counts[counts>=1])/total_bins

    archive_cov_by_gen = []
    bd_cols = [f'bd{i}' for i in range(dim_map)]
    bd_cols = bd_cols[-2:]
    for gen in range(1,len(archive)//px['lambda']):
        archive_at_gen = archive[:gen*px['lambda']]
        bds_at_gen = np.array([ind.desc[-2:] for ind in archive_at_gen])
        archive_at_gen_data = pd.DataFrame(bds_at_gen, columns=bd_cols)
        cov_at_gen = compute_cov(archive_at_gen_data, ss_min, ss_max, px['dim_map'], bd_inds, nb_div)
        archive_cov_by_gen.append(cov_at_gen)
    archive_cov_by_gen = np.array(archive_cov_by_gen)
    to_save = os.path.join(args.log_dir, 'archive_cov_by_gen')
    np.savez(to_save, archive_cov_by_gen=archive_cov_by_gen)

    print()
    print(f'Finished performing {args.algo} search successfully.')
    
################################################################################
############################## Params parsing ##################################
################################################################################
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    args = process_args()
    
    main(args)
