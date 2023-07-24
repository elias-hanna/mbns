def main(args, pool):
    #########################################################################
    ####################### Preparation of run ##############################
    #########################################################################
    
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
    bins = env_params['bins'] ## for grid based qd

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
        'ensemble_size': 1 if args.model_type == 'det' else args.ens_size,
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
            'ensemble_size': 1 if args.model_type == 'det' else args.ens_size,
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
        'policy_param_init_min': -5 if args.environment != 'hexapod_omni' else 0.0,
        'policy_param_init_max': 5 if args.environment != 'hexapod_omni' else 1.0,

        'fitness_func': args.fitness_func,
        'n_waypoints': n_waypoints,
        'num_cores': args.num_cores,
        'dim_map': dim_map,
        'bd_inds': bd_inds,
        'bins': bins,

        'pretrain': args.pretrain,
        ## srf parameters
        'srf_var': 0.001,
        'srf_cor': 0.01,

        ## Dump params/ memory gestion params
        "log_ind_trajs": args.log_ind_trajs,
        "dump_ind_trajs": args.dump_ind_trajs,
    }
    px['dab_params'] = params
    px['min'] = params['policy_param_init_min']
    px['max'] = params['policy_param_init_max']
    ## Correct obs dim for controller if open looping on time
    if params['time_open_loop'] == True:
        controller_params['obs_dim'] = 1
    if init_obs is not None:
        params['init_obs'] = init_obs

    if not is_local_env:
        env = WrappedEnv(params)
        dim_x = env.policy_representation_dim
    init_obs = params['init_obs']
    px['dim_x'] = dim_x

    surrogate_model_params['gen_dim'] = dim_x

    #########################################################################
    ####################### End of Preparation of run #######################
    #########################################################################

    
    ## Train a model with x data sampled with y method in env
    # Sample training data (randomly gather transitions across all state space)
    # initialize replay buffer
    replay_buffer = SimpleReplayBuffer(
        max_replay_buffer_size=1000000,
        observation_dim=obs_dim,
        action_dim=act_dim,
        env_info_sizes=dict(),
    )
    
    # Create dynamics models with architectures specified in args
    dynamics_model, dynamics_model_trainer = get_dynamics_model(params)
        
    # Train each of the dynamics models on the same training data
    dynamics_model_trainer.train_from_buffer(replay_buffer, 
                                             holdout_pct=0.1,
                                             max_grad_steps=100000,
                                             verbose=True)
    ## Take 10 diverse trajectories from NS on that env
    # Get the expert trajectories
    path_to_examples = os.path.join(args.ns_examples_path,
                                    env+'_example_trajectories.npz')
    example_trajs_data = np.load(path_to_examples)
    # Evaluate model performance on these trajectories
    # initialize visualizers
    params['model'] = dynamics_model # to pass down to the visualizer routines

    test_traj_visualizer = TestTrajectoriesVisualization(params)

    n_step_visualizer = NStepErrorVisualization(params)

    # Visualize n step error and disagreement

    n_step_visualizer.set_n(1)

    examples_1_step_trajs, examples_1_step_disagrs, examples_1_step_pred_errors = n_step_visualizer.dump_plots(
        env,
        args.init_method,
        init_episode,
        'examples', dump_separate=True, no_sep=True)

    n_step_visualizer.set_n(max_step//10)
    
    examples_plan_h_step_trajs, examples_plan_h_step_disagrs, examples_plan_h_step_pred_errors = n_step_visualizer.dump_plots(
        env,
        args.init_method,
        init_episode,
        'examples', dump_separate=True, no_sep=True)

    ### Full recursive prediction visualizations ###
    examples_pred_trajs, examples_disagrs, examples_pred_errors = test_traj_visualizer.dump_plots(
        env,
        args.init_method,
        args.init_episode,
        'examples', dump_separate=True, no_sep=True)

    ### Full recursive prediction visualizations ###
    test_traj_visualizer.set_test_trajectories(test_trajectories)
    test_pred_trajs, test_disagrs, test_pred_errors = test_traj_visualizer.dump_plots(
        env,
        args.init_method,
        args.init_episode,
        'test', dump_separate=True, no_sep=True)

    
    return 0
    

if __name__ == "__main__":
    #----------Algo imports--------#
    from src.map_elites import common as cm
    from src.map_elites import unstructured_container, cvt
    from src.map_elites.mbns import ModelBasedNS

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
    # import torch.multiprocessing as multiprocessing
    import multiprocessing
    
    #----------controller imports--------#
    from model_init_study.controller.nn_controller \
        import NeuralNetworkController
    from exps_utils import RNNController

    #----------Environment imports--------#
    import gym
    from exps_utils import get_env_params
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

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    parser = argparse.ArgumentParser()

    ## Add ns_examples_path
    parser.add_argument('--ns-examples-path', type=str, default=None,
                        help='Path of examples trajectories foraged with NS')
    args = process_args(parser)

    multiprocessing.set_start_method('spawn')

    num_cores_set = args.num_cores
    # setup the parallel processing pool
    if num_cores_set == 0:
        num_cores = multiprocessing.cpu_count() - 1 # use all cores
    else:
        num_cores = num_cores_set
        
    # pool = multiprocessing.Pool(num_cores)
    pool = multiprocessing.get_context("spawn").Pool(num_cores)
    # pool = get_context("fork").Pool(num_cores)
    #pool = ThreadPool(num_cores)

    main(args, pool)
