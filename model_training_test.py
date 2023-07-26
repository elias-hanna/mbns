class WrappedDynamicsModel():
    def __init__(self, dynamics_model):
        self.dynamics_model = dynamics_model

    def forward_multiple(self, A, S, mean=True, disagr=True, ensemble=True, det_ens=False):
        ## Takes a list of actions A and a list of states S we want to query the model from
        ## Returns a list of the return of a forward call for each couple (action, state)
        assert len(A) == len(S)
        batch_len = len(A)
        if ensemble:
            ens_size = self.dynamics_model.ensemble_size
        else:
            ens_size = 1
        S_0 = np.empty((batch_len*ens_size, S.shape[1]))
        A_0 = np.empty((batch_len*ens_size, A.shape[1]))

        batch_cpt = 0
        for a, s in zip(A, S):
            S_0[batch_cpt*ens_size:batch_cpt*ens_size+ens_size,:] = \
            np.tile(s,(ens_size, 1))

            A_0[batch_cpt*ens_size:batch_cpt*ens_size+ens_size,:] = \
            np.tile(a,(ens_size, 1))
            batch_cpt += 1
        if ensemble:
            return self.forward(A_0, S_0, mean=mean, disagr=disagr, multiple=True)
        elif det_ens:
            s_0 = copy.deepcopy(S_0)
            a_0 = copy.deepcopy(A_0)
            s_0 = ptu.from_numpy(s_0)
            a_0 = ptu.from_numpy(a_0)
            return self.dynamics_model.output_pred_with_ts(
                    torch.cat((s_0, a_0), dim=-1),
                    mean=mean), [0]*len(s_0)
        else:
            s_0 = copy.deepcopy(S_0)
            a_0 = copy.deepcopy(A_0)
            s_0 = ptu.from_numpy(s_0)
            a_0 = ptu.from_numpy(a_0)
            return self.dynamics_model.output_pred(
                    torch.cat((s_0, a_0), dim=-1),
                    mean=mean), [0]*len(s_0)

    def forward(self, a, s, mean=True, disagr=True, multiple=False):
        s_0 = copy.deepcopy(s)
        a_0 = copy.deepcopy(a)

        if not multiple:
            s_0 = np.tile(s_0,(self.dynamics_model.ensemble_size, 1))
            a_0 = np.tile(a_0,(self.dynamics_model.ensemble_size, 1))

        s_0 = ptu.from_numpy(s_0)
        a_0 = ptu.from_numpy(a_0)

        # a_0 = a_0.repeat(self._dynamics_model.ensemble_size,1)

        # if probalistic dynamics model - choose output mean or sample
        if disagr:
            if not multiple:
                pred_delta_ns, disagreement = self.dynamics_model.sample_with_disagreement(
                    torch.cat((
                        self.dynamics_model._expand_to_ts_form(s_0),
                        self.dynamics_model._expand_to_ts_form(a_0)), dim=-1
                    ))#, disagreement_type="mean" if mean else "var")
                pred_delta_ns = ptu.get_numpy(pred_delta_ns)
                return pred_delta_ns, disagreement
            else:
                pred_delta_ns_list, disagreement_list = \
                self.dynamics_model.sample_with_disagreement_multiple(
                    torch.cat((
                        self.dynamics_model._expand_to_ts_form(s_0),
                        self.dynamics_model._expand_to_ts_form(a_0)), dim=-1
                    ))#, disagreement_type="mean" if mean else "var")
                for i in range(len(pred_delta_ns_list)):
                    pred_delta_ns_list[i] = ptu.get_numpy(pred_delta_ns_list[i])
                    disagreement_list[i] = ptu.get_numpy(disagreement_list[i])
                return pred_delta_ns_list, disagreement_list
        else:
            pred_delta_ns = self.dynamics_model.output_pred_ts_ensemble(s_0, a_0, mean=mean)
        return pred_delta_ns, 0
    

def sample(env, n_samples):
    '''
    This method samples n_samples transitions from a uniformely sampled state
    and action in their respective spaces.
    Returns the observed transitions and corresponding state deltas.
    '''
    env = copy.copy(env) ## copy gym env
    transitions = []
    deltas = []
    obs_shape = env.observation_space.shape

    for _ in range(n_samples):
        ## Reset environment
        env.reset()
        ## Sample an action in action space
        a = env.action_space.sample()
        qpos, qvel, s = env.sample_q_vectors()

        env.set_state(qpos, qvel)
        ## Perform a step in the environment
        ns, r, done, info = env.step(a)
        ## Add observed transition
        transitions.append((copy.copy(s),
                            copy.copy(a),
                            copy.copy(ns)))
        ## Sample a new action in action space
        a = env.action_space.sample()
        deltas.append(ns-s)
        s = ns

    return transitions, deltas

def add_transitions_to_buffer(transitions, replay_buffer):
    for trans in transitions:
        s = trans[0]
        a = trans[1]
        ns = trans[2]

        reward = 0
        done = 0
        info = {}
        replay_buffer.add_sample(s, a, reward, done, ns, info)
    return 1

def arch_to_str(arch_array):
    out = str(arch_array)
    out = out.replace(' ', '')
    out = out.replace(',', '_')
    return out

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

    ## Set the type of controller we use
    if args.c_type == 'ffnn':
        controller_type = NeuralNetworkController
    elif args.c_type == 'rnn':
        controller_type = RNNController

    path_to_examples = os.path.join(args.ns_examples_path,
                                    args.environment+'_example_trajectories.npz')

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
        'batch_size': 512,
        'learning_rate': 1e-3,
        'train_unique_trans': False,
        'model_type': args.model_type,
        'model_horizon': args.model_horizon if args.model_horizon!=-1 else max_step,
        'ensemble_size': 1 if args.model_type == 'det' else args.ens_size,
    }
    ## General parameters
    params = \
    {
        ## general parameters
        'state_dim': state_dim,
        'obs_dim': obs_dim,
        'action_dim': act_dim,
        'separator': None,
        'dynamics_model_params': dynamics_model_params,
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
        'path_to_test_trajectories': path_to_examples,
    }
    
    ## Correct obs dim for controller if open looping on time
    if params['time_open_loop'] == True:
        controller_params['obs_dim'] = 1
    if init_obs is not None:
        params['init_obs'] = init_obs

    if not is_local_env:
        env = WrappedEnv(params)
        gym_env = env._env
        dim_x = env.policy_representation_dim
    init_obs = params['init_obs']

    #########################################################################
    ####################### End of Preparation of run #######################
    #########################################################################

    
    ## Train a model with x data sampled with y method in env
    # Sample training data (randomly gather transitions across all state space)
    transitions, deltas = sample(gym_env, args.n_samples)
    # initialize replay buffer
    replay_buffer = SimpleReplayBuffer(
        max_replay_buffer_size=1000000,
        observation_dim=obs_dim,
        action_dim=act_dim,
        env_info_sizes=dict(),
    )

    add_transitions_to_buffer(transitions, replay_buffer)
    
    dynamics_models = []
    dynamics_models_trainers = []
    archs = []
    for arch in args.archs:
        arch_layers = [int(l) for l in arch.split(',')] 
        # Turn str arch layers into reg tab
        archs.append(arch_layers)

        params['dynamics_model_params']['layer_size'] = arch_layers
        # Create dynamics models with architectures specified in args
        dynamics_model, dynamics_model_trainer = get_dynamics_model(params)

        dynamics_models.append(dynamics_model)
        dynamics_models_trainers.append(dynamics_model_trainer)
        # Train each of the dynamics models on the same training data
        dynamics_model_trainer.train_from_buffer(replay_buffer, 
                                                 holdout_pct=0.1,
                                                 max_grad_steps=100000,
                                                 verbose=True)

    ## Take 10 diverse trajectories from NS on that env
    # Get the expert trajectories
    # the trajectories are passed with the 'path_to_test_trajectories' key
    # in the params dictionary
    
    # example_trajs_data = np.load(path_to_examples)

    examples_1_step_list = []
    examples_plan_h_list = []
    examples_full_list = []
    
    for (arch, dynamics_model) in zip(archs, dynamics_models):    
        # Evaluate model performance on these trajectories
        # initialize visualizers
        # to pass down to the visualizer routines
        params['model'] = WrappedDynamicsModel(dynamics_model) 
        test_traj_visualizer = TestTrajectoriesVisualization(params)

        n_step_visualizer = NStepErrorVisualization(params)

        # Visualize n step error and disagreement

        n_step_visualizer.set_n(1)

        examples_1_step_trajs, examples_1_step_disagrs, examples_1_step_pred_errors = n_step_visualizer.dump_plots(
            args.environment,
            'model arch ' +arch_to_str(arch),
            args.n_samples,
            'examples', dump_separate=True, no_sep=True)

        n_step_visualizer.set_n(max_step//10)

        examples_plan_h_step_trajs, examples_plan_h_step_disagrs, examples_plan_h_step_pred_errors = n_step_visualizer.dump_plots(
            args.environment,
            'arch_' + arch_to_str(arch),
            args.n_samples,
            'examples', dump_separate=True, no_sep=True)

        ### Full recursive prediction visualizations ###
        examples_pred_trajs, examples_disagrs, examples_pred_errors = test_traj_visualizer.dump_plots(
            args.environment,
            'model arch ' +arch_to_str(arch),
            args.n_samples,
            'examples', dump_separate=True, no_sep=True)

        examples_1_step_list.append((examples_1_step_trajs,
                                     examples_1_step_disagrs,
                                     examples_1_step_pred_errors))
        examples_plan_h_list.append((examples_plan_h_step_trajs,
                                          examples_plan_h_step_disagrs,
                                          examples_plan_h_step_pred_errors))
        examples_full_list.append((examples_pred_trajs,
                                   examples_disagrs,
                                   examples_pred_errors))

    ## Save the resulting data in a npz
    save_dict = {}
    for i in range(len(archs)):
        save_dict['1_step_pred_trajs_'+str(i)] = examples_1_step_list[i][0]
        save_dict['1_step_disagrs_'+str(i)] = examples_1_step_list[i][1]
        save_dict['1_step_pred_errors_'+str(i)] = examples_1_step_list[i][2]

        save_dict['plan_h_pred_trajs_'+str(i)] = examples_plan_h_list[i][0]
        save_dict['plan_h_disagrs_'+str(i)] = examples_plan_h_list[i][1]
        save_dict['plan_h_pred_errors_'+str(i)] = examples_plan_h_list[i][2]

        save_dict['full_pred_trajs_'+str(i)] = examples_full_list[i][0]
        save_dict['full_disagrs_'+str(i)] = examples_full_list[i][1]
        save_dict['full_pred_errors_'+str(i)] = examples_full_list[i][2]

    save_dict['archs'] = [arch_to_str(arch) for arch in archs]

    dump_path = os.path.join(args.log_dir, 'model_archs_pred_data')
    np.savez(dump_path, **save_dict)
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

    #----------Visualizing imports--------#
    from model_init_study.visualization.test_trajectories_visualization \
        import TestTrajectoriesVisualization
    from model_init_study.visualization.n_step_error_visualization \
        import NStepErrorVisualization
        
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
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Number of uniformely sampled transitions to '\
                        'use to initialize the models')
    parser.add_argument('--archs', type=str, nargs='*', default=[],
                        help='Model architectures to test, ex: 100,100,50'\
                        '10,10,5')
    args = process_args(parser)

    # multiprocessing.set_start_method('spawn')

    # num_cores_set = args.num_cores
    # # setup the parallel processing pool
    # if num_cores_set == 0:
    #     num_cores = multiprocessing.cpu_count() - 1 # use all cores
    # else:
    #     num_cores = num_cores_set
        
    # pool = multiprocessing.get_context("spawn").Pool(num_cores)

    pool = None
    main(args, pool)
