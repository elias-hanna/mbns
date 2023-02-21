#----------Algo imports--------#
from src.map_elites import common as cm
from src.map_elites import unstructured_container, cvt
from src.map_elites.mbqd import ModelBasedQD

from exps_utils import get_dynamics_model, get_surrogate_model, \
    get_observation_model, addition_condition, evaluate_, evaluate_all_, \
    process_args,plot_cov_and_trajs, save_archive_cov_by_gen

#----------Model imports--------#
from src.models.observation_models.deterministic_obs_model import DeterministicObsModel
from src.models.observation_models.srf_deterministic_obs_model import SrfDeterministicObsModel
from src.models.dynamics_models.srf_deterministic_ensemble import SrfDeterministicEnsemble
from src.models.dynamics_models.deterministic_model import DeterministicDynModel
from src.models.dynamics_models.deterministic_ensemble import DeterministicEnsemble
from src.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
from src.models.surrogate_models.det_surrogate import DeterministicQDSurrogate
from src.data_management.replay_buffers.simple_replay_buffer import SimpleReplayBuffer
import src.torch.pytorch_util as ptu
import torch

#----------controller imports--------#
from model_init_study.controller.nn_controller \
    import NeuralNetworkController
from exps_utils import RNNController

#----------Environment imports--------#
import gym
from exps_utils import get_env_params
from exps_utils import WrappedEnv

#----------Init methods imports--------#
from model_init_study.initializers.random_policy_initializer \
    import RandomPolicyInitializer
from model_init_study.initializers.random_actions_initializer \
    import RandomActionsInitializer
from model_init_study.initializers.random_actions_random_policies_hybrid_initializer \
    import RARPHybridInitializer
from model_init_study.initializers.brownian_motion \
    import BrownianMotion
from model_init_study.initializers.colored_noise_motion \
    import ColoredNoiseMotion

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

    px = \
    {
        # type of qd 'unstructured, grid, cvt'
        "type": args.qd_type,
        
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        "cvt_use_cache": True,
        # we evaluate in batches to parallelize
        "batch_size": args.b_size,
        # proportion of total number of niches to be filled before starting
        "random_init": 0.005,  
        # batch for random initialization
        "random_init_batch": args.random_init_batch,
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

        "model_variant": args.model_variant, # "dynamics" or "direct" or "all_dynamics"  
        "train_model_on": not args.no_training,
        "perfect_model_on": args.perfect_model,
        
        # "train_freq": 40, # train at a or condition between train freq and evals_per_train
        # "evals_per_train": 500,
        "train_freq": args.train_freq_gen, # train at a or condition between train freq and evals_per_train
        "evals_per_train": args.train_freq_eval,

        "log_model_stats": False,
        "log_time_stats": False, 
        "dump_ind_trajs": args.dump_ind_trajs,
        
        ## for dump
        "ensemble_dump": False,
        
        # 0 for random emiiter, 1 for optimizing emitter
        # 2 for random walk emitter, 3 for model disagreement emitter
        "emitter_selection": 0,

        "min_found_model": args.min_found_model,
        "transfer_selection": args.transfer_selection,
        "nb_transfer": args.nb_transfer,
        'env_name': args.environment,
        'init_method': args.init_method,
    }

    
    #########################################################################
    ####################### Preparation of run ##############################
    #########################################################################
    
    noise_beta = 2
    if args.init_method == 'random-policies':
        Initializer = RandomPolicyInitializer
    elif args.init_method == 'random-actions':
        Initializer = RandomActionsInitializer
    elif args.init_method == 'rarph':
        Initializer = RARPHybridInitializer
    elif args.init_method == 'brownian-motion':
        Initializer = BrownianMotion
    elif args.init_method == 'levy-flight':
        Initializer = LevyFlight
    elif args.init_method == 'colored-noise-beta-0':
        Initializer = ColoredNoiseMotion
        noise_beta = 0
    elif args.init_method == 'colored-noise-beta-1':
        Initializer = ColoredNoiseMotion
        noise_beta = 1
    elif args.init_method == 'colored-noise-beta-2':
        Initializer = ColoredNoiseMotion
        noise_beta = 2
    elif args.init_method == 'no-init':
        ## this will do uninitialized model daqd
        pass
    elif args.init_method == 'vanilla':
        ## This will do vanilla daqd
        pass
    else:
        raise Exception(f"Warning {args.init_method} isn't a valid initializer")
    
    env_params = get_env_params(args)

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
        ## for hexapod
        obs_dim = state_dim 
        act_dim = env_params['act_dim']
        max_step = 300
        dim_x = env_params['dim_x']
        
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
        'model_type': args.model_type,
        'model_horizon': args.model_horizon if args.model_horizon!=-1 else max_step,
        'ensemble_size': args.ens_size,
    }
    ## Observation model parameters
    if args.use_obs_model:
        observation_model_params = \
        {
            'obs_dim': obs_dim,
            'state_dim': state_dim,
            'layer_size': [500, 400],
            'batch_size': 512,
            'learning_rate': 1e-3,
            'train_unique_trans': False,
            'obs_model_type': args.obs_model_type,
            'ensemble_size': args.ens_size,
        }
    else:
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

        'clip_obs': args.clip_obs, # clip models predictions 
        'clip_state': args.clip_state, # clip models predictions 
        ## env parameters
        'env': gym_env,
        'env_name': args.environment,
        'env_max_h': max_step,
        'use_obs_model': args.use_obs_model,
        ## algo parameters
        'policy_param_init_min': px['min'],
        'policy_param_init_max': px['max'],

        'fitness_func': args.fitness_func,
        'n_waypoints': n_waypoints,
        'num_cores': args.num_cores,
        'dim_map': dim_map,
        'bd_inds': bd_inds,
        ## pretraining parameters
        'pretrain': args.pretrain,
        ## srf parameters
        'srf_var': 0.001,
        'srf_cor': 0.01,

        ## Dump params/ memory gestion params
        "log_ind_trajs": args.log_ind_trajs,
        "dump_ind_trajs": args.dump_ind_trajs,
        ## Model Init Study params
        'n_init_episodes': args.init_episodes,
        # 'n_test_episodes': int(.2*args.init_episodes), # 20% of n_init_episodes
        'n_test_episodes': 2,
        'action_init': 0,
        ## Random walks parameters
        'step_size': 0.1,
        'noise_beta': noise_beta,
        ## RA parameters
        'action_lasting_steps': 5,

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
    
    surrogate_model_params['gen_dim'] = dim_x
    
    ## Get the various models we need for the run
    dynamics_model, dynamics_model_trainer = get_dynamics_model(params)
    if observation_model_params:
        observation_model = get_observation_model(params)
    surrogate_model, surrogate_model_trainer = get_surrogate_model(surrogate_model_params)

    ## Initialize model with wnb from previous run if an init method is to be used
    if args.init_method != 'no-init' and args.init_method != 'vanilla':
        if args.init_data_path is not None:
            data_path = args.init_data_path
            path = f'{data_path}/{args.environment}_results/{args.rep}/'\
                f'{args.environment}_{args.init_method}_{args.init_episodes}_model_wnb.pt'
        else:
            import src
            path_to_src = src.__path__[0]
            module_path = f'{path_to_src}/..'
            path = f'{module_path}/data/{args.environment}_results/{args.rep}/'\
                f'{args.environment}_{args.init_method}_{args.init_episodes}_model_wnb.pt'
        dynamics_model.load_state_dict(torch.load(path))
        dynamics_model.eval()

    if not is_local_env:
        env.set_dynamics_model(dynamics_model)
    elif args.environment == 'hexapod_omni':
        from src.envs.hexapod_dart.hexapod_env import HexapodEnv ## Contains hexapod 
        env = HexapodEnv(dynamics_model=dynamics_model,
                         render=False,
                         record_state_action=True,
                         ctrl_freq=100)
        
    f_real = env.evaluate_solution # maybe move f_real and f_model inside

    ## If we evaluate directly on the real system f_model = f_real 
    if args.perfect_model:
        f_model = f_real
        px['model_variant'] = "dynamics"
    ## Else if we evaluate on the generated random models we use one of below
    elif args.model_variant == "dynamics" :
        if args.model_type == "det":
            f_model = env.evaluate_solution_model 
        elif args.model_type == "prob":
            f_model = env.evaluate_solution_model_ensemble
    elif args.model_variant == "all_dynamics":
        if args.model_type == "det":
            f_model = env.evaluate_solution_model_all
        elif args.model_type == "det_ens" or args.model_type == "srf_ens":
            f_model = env.evaluate_solution_model_det_ensemble_all
        elif args.model_type == "prob":
            f_model = env.evaluate_solution_model_ensemble_all

    # initialize replay buffer
    replay_buffer = SimpleReplayBuffer(
        max_replay_buffer_size=1000000,
        observation_dim=obs_dim,
        action_dim=act_dim,
        env_info_sizes=dict(),
    )
    
    mbqd = ModelBasedQD(dim_map, dim_x,
                        f_real, f_model,
                        surrogate_model, surrogate_model_trainer,
                        dynamics_model, dynamics_model_trainer,
                        replay_buffer, 
                        n_niches=args.n_niches,
                        params=px, log_dir=args.log_dir)

    #mbqd.compute(num_cores_set=cpu_count()-1, max_evals=args.max_evals)
    archive, n_evals = mbqd.compute(num_cores_set=args.num_cores, max_evals=args.max_evals)
    
    cm.save_archive(archive, f"{n_evals}_real_all", px, args.log_dir)
        
    # ## Plot archive trajectories on real system
    # if args.log_ind_trajs:
    #     ## Extract real sys BD data from s_list
    #     real_bd_traj_data = [s.obs_traj for s in archive]
    #     ## Format the bd data to plot with labels
    #     all_bd_traj_data = []

    #     all_bd_traj_data.append((real_bd_traj_data, 'real system'))
    #     ## Plot real archive and model(s) archive on plot
    #     total_plots = len(all_bd_traj_data)
    #     ## make it as square as possible
    #     rows = cols = round(np.sqrt(total_plots))
    #     ## Add a row in case closest square cannot take all plots in
    #     if total_plots > rows*cols:
    #         rows += 1

    #     if dim_map == 3:
    #         fig1 = plt.figure()
    #         fig2 = plt.figure()
    #         axs1 = []
    #         axs2 = []
    #         plt_num = 0
    #         #for plt_num in range(total_plots):
    #         for col in range(cols):
    #             axs1_cols = []
    #             axs2_cols = []
    #             for row in range(rows):
    #                 axs1_cols.append(fig1.add_subplot(cols, rows, plt_num+1, projection='3d'))
    #                 axs2_cols.append(fig2.add_subplot(cols, rows, plt_num+1, projection='3d'))
    #                 plt_num += 1
    #             axs1.append(axs1_cols)
    #             axs2.append(axs2_cols)
    #     else:
    #         fig1, axs1 = plt.subplots(rows, cols)
    #         fig2, axs2 = plt.subplots(rows, cols)
    #     m_cpt = 0
        
    #     for col in range(cols):
    #         for row in range(rows):
    #             try:
    #                 if not hasattr(axs1, '__len__'):
    #                 # if not 'ens' in args.model_type or args.perfect_model:
    #                     ax1 = axs1
    #                     ax2 = axs2
    #                 else:
    #                     ax1 = axs1[row][col]
    #                     ax2 = axs2[row][col]

    #                 loc_bd_traj_data, loc_system_name = all_bd_traj_data[m_cpt]

    #                 ## format to take care of trajs that end before max_step + 1
    #                 # (+1 because we store max_step transitions)
    #                 formatted_loc_bd_traj_data = []
    #                 traj_dim = loc_bd_traj_data[0].shape[1] # get traj dim
    #                 for loc_bd_traj in loc_bd_traj_data:
    #                     formatted_loc_bd_traj = np.empty((max_step+1,traj_dim))
    #                     formatted_loc_bd_traj[:] = np.nan
    #                     formatted_loc_bd_traj[:len(loc_bd_traj)] = loc_bd_traj
    #                     formatted_loc_bd_traj_data.append(formatted_loc_bd_traj)
    #                 loc_bd_traj_data = formatted_loc_bd_traj_data
    #                 loc_bd_traj_data = np.array(loc_bd_traj_data)
                    
    #                 ## Plot BDs
    #                 if dim_map == 3:
    #                     for loc_bd_traj in loc_bd_traj_data:
    #                         last_ind = (~np.isnan(loc_bd_traj)).cumsum(0).argmax(0)[0]
    #                         ax1.scatter(xs=loc_bd_traj[last_ind,0],
    #                                     ys=loc_bd_traj[last_ind,1],
    #                                     zs=loc_bd_traj[last_ind,2], s=3, alpha=0.1)
    #                     ax1.scatter(xs=init_obs[bd_inds[0]],ys=init_obs[bd_inds[1]],zs=init_obs[bd_inds[2]], s=10, c='red')
    #                     ## Plot trajectories
    #                     for bd_traj in loc_bd_traj_data:
    #                         ax2.plot(bd_traj[:,bd_inds[0]], bd_traj[:,bd_inds[1]], bd_traj[:,bd_inds[2]], alpha=0.1, markersize=1)
    #                 else:
    #                     for loc_bd_traj in loc_bd_traj_data:
    #                         last_ind = (~np.isnan(loc_bd_traj)).cumsum(0).argmax(0)[0]
    #                         ax1.scatter(x=loc_bd_traj[last_ind,bd_inds[0]],
    #                                     y=loc_bd_traj[last_ind,bd_inds[1]], s=3, alpha=0.1)
    #                     ax1.scatter(x=init_obs[bd_inds[0]],y=init_obs[bd_inds[1]], s=10, c='red')
    #                     ## Plot trajectories
    #                     for bd_traj in loc_bd_traj_data:
    #                         ax2.plot(bd_traj[:,bd_inds[0]], bd_traj[:,bd_inds[1]], alpha=0.1, markersize=1)
    #                 ax1.set_xlabel('x-axis')
    #                 ax1.set_ylabel('y-axis')
    #                 ax1.set_title(f'Archive coverage on {loc_system_name}')
    #                 if 'real' in loc_system_name:
    #                     ax1.set_xlim(ss_min[bd_inds[0]], ss_max[bd_inds[0]])
    #                     ax1.set_ylim(ss_min[bd_inds[1]], ss_max[bd_inds[1]])
    #                     if dim_map == 3:
    #                         ax1.set_ylabel('z-axis')
    #                         ax1.set_zlim(ss_min[bd_inds[2]], ss_max[bd_inds[2]])
    #                 else:
    #                     loc_bd_mins = np.min(loc_bd_traj_data[:,-1,:], axis=0) 
    #                     loc_bd_maxs = np.max(loc_bd_traj_data[:,-1,:], axis=0) 
    #                     ax1.set_xlim(loc_bd_mins[0], loc_bd_maxs[0])
    #                     ax1.set_ylim(loc_bd_mins[1], loc_bd_maxs[1])
    #                 if dim_map != 3:
    #                     ax1.set_aspect('equal', adjustable='box')
                    
    #                 ax2.set_xlabel('x-axis')
    #                 ax2.set_ylabel('y-axis')
    #                 ax2.set_title(f'Individuals trajectories on {loc_system_name}')
    #                 if 'real' in loc_system_name:
    #                     ax2.set_xlim(ss_min[bd_inds[0]], ss_max[bd_inds[0]])
    #                     ax2.set_ylim(ss_min[bd_inds[1]], ss_max[bd_inds[1]])
    #                     if dim_map == 3:
    #                         ax2.set_ylabel('z-axis')
    #                         ax2.set_zlim(ss_min[bd_inds[2]], ss_max[bd_inds[2]])
    #                 else:
    #                     loc_bd_mins = np.min(loc_bd_traj_data, axis=(0,1)) 
    #                     loc_bd_maxs = np.max(loc_bd_traj_data, axis=(0,1))
    #                     ax2.set_xlim(loc_bd_mins[0], loc_bd_maxs[0])
    #                     ax2.set_ylim(loc_bd_mins[1], loc_bd_maxs[1])
    #                 if dim_map != 3:
    #                     ax2.set_aspect('equal', adjustable='box')
    #                 m_cpt += 1
    #             except:
    #                 fig1.delaxes(axs1[row][col])
    #                 fig2.delaxes(axs2[row][col])

    #     fig1.set_size_inches(total_plots*2,total_plots*2)
    #     fig1.suptitle('Archive coverage after DAB')
    #     file_path = os.path.join(args.log_dir, f"{args.environment}_real_cov")
    #     fig1.savefig(file_path,
    #                 dpi=300, bbox_inches='tight')

    #     fig2.set_size_inches(total_plots*2,total_plots*2)
    #     fig2.suptitle('Individuals trajectories after DAB')
    #     file_path = os.path.join(args.log_dir, f"{args.environment}_ind_trajs")
    #     fig2.savefig(file_path,
    #                 dpi=300, bbox_inches='tight')

    ## Plot archive trajectories on real system
    if args.log_ind_trajs:
        ## Extract real sys BD data from s_list
        real_bd_traj_data = [s.obs_traj for s in archive]
        ## Format the bd data to plot with labels
        all_bd_traj_data = []

        all_bd_traj_data.append((real_bd_traj_data, 'real system'))

        plot_cov_and_trajs(all_bd_traj_data, args, params)

    ## Plot archive coverage at each generation (does not work for QD instances)
    ## will consider a gen = lambda indiv added to archive
    save_archive_cov_by_gen(archive, args, px, params)

    print()
    print(f'Finished performing mbqd search successfully.')
    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    args = process_args()

    main(args)
