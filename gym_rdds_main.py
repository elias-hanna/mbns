#----------Algo imports--------#
from src.map_elites import common as cm
from src.map_elites import unstructured_container, cvt
from src.map_elites.qd import QD
from src.map_elites.ns import NS

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
# import diversity_algorithms.environments.env_imports ## Contains deterministic ant + fetch

#----------Pretraining imports--------#
from src.data_generation.srf_training import get_ensemble_training_samples
#----------Data manipulation imports--------#
import numpy as np
import copy
import pandas as pd
import itertools
#----------Utils imports--------#
import os, sys
import argparse
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import cpu_count
import random

import time
import tqdm

################################################################################
################################### MAIN #######################################
################################################################################
def main(args):
    bootstrap_archive = None
    ## Parameters that are passed to NS or QD instance
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
        'model_type': args.model_type,
        "model_variant": args.model_variant, # "dynamics" or "direct" or "all_dynamics"  
        "perfect_model_on": args.perfect_model,
        "ensemble_size": 1,
        
        "log_model_stats": False,
        "log_time_stats": False, 
        "log_ind_trajs": args.log_ind_trajs,
        "dump_ind_trajs": args.dump_ind_trajs,

        "norm_bd": args.norm_bd,
        "nov_ens": args.nov_ens,
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
        'model_type': args.model_type,
        'model_horizon': args.model_horizon if args.model_horizon!=-1 else max_step,
        'ensemble_size': 1 if args.model_type == 'det' else args.ens_size,
    }
    if args.use_obs_model:
    ## Observation model parameters
        observation_model_params = \
        {
            'obs_dim': obs_dim,
            'state_dim': state_dim,
            'layer_size': [500, 400],
            # 'layer_size': 500,
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

    surrogate_model_params['gen_dim'] = dim_x
    px['dim_x'] = dim_x

    ###### debug
    # params['obs_dim'] = 1
    # params['action_dim'] = 1
    # params['dynamics_model_params']['obs_dim'] = 1
    # params['dynamics_model_params']['action_dim'] = 1
    # params['state_min'] = params['state_min'][:1]
    # params['state_max'] = params['state_max'][:1]
    # params['action_min'] = params['action_min'][:1]
    # params['action_max'] = params['action_max'][:1]
    #######

    ## Get the various random models we need for the run
    dynamics_model, dynamics_model_trainer = get_dynamics_model(params)
    if observation_model_params:
        observation_model = get_observation_model(params)
    surrogate_model, surrogate_model_trainer = get_surrogate_model(surrogate_model_params)

    ## Pretrain the random models on some generated data
    if args.pretrain: ## won't go in with default value ''
        ## initialize replay buffer
        replay_buffer = SimpleReplayBuffer(
            max_replay_buffer_size=1000000,
            observation_dim=params['obs_dim'],
            action_dim=params['action_dim'],
            env_info_sizes=dict(),
        )
        ## Generate data
        if args.pretrain == 'srf':
            ens_size = 1 if args.model_type == 'det' else args.ens_size
            n_training_samples = args.pretrain_budget
            input_data, output_data = get_ensemble_training_samples(
                params,
                n_training_samples=n_training_samples, ensemble_size=ens_size
            )

            for i in range(ens_size):
                for k in range(n_training_samples):
                    ## access ith training data
                    in_train_data = input_data[i]
                    out_train_data = output_data[i]
                    ## add it to ith replay buffer
                    replay_buffer.add_sample(
                        in_train_data[k,:params['obs_dim']],
                        in_train_data[k,params['obs_dim']:],
                        0, False,
                        out_train_data[k],
                        {}
                    )

                ## train from buffer each model one by one
                dynamics_model_trainer.train_from_buffer(
                    replay_buffer, 
                    holdout_pct=0.2,
                    max_grad_steps=100000,
                    epochs_since_last_update=5,
                    verbose=True,
                )

                eval_stats = dynamics_model_trainer.get_diagnostics()
                print('###########################################################')
                print('################# Final training stats ####################')
                print('###########################################################')
                for key in eval_stats.keys():
                    print(f'{key}: {eval_stats[key]}')
                print('###########################################################')
                print('###########################################################')
                print('###########################################################')
                
                ## Plot generated srf
                fig, (ax1, ax2) = plt.subplots(1, 2)
                vmin = np.min(out_train_data[:,0]-in_train_data[:,0])
                vmax = np.max(out_train_data[:,0]-in_train_data[:,0])
                sc = ax1.scatter(in_train_data[:,0], in_train_data[:,1], c=out_train_data[:,0]-in_train_data[:,0],)
                                # vmin=vmin, vmax=vmax)
                plt.colorbar(sc)
                # plt.title('Generated srf data')
                ax1.set_title('Generated srf data')
                ## Plot learned srf
                # Query the nn for the input data:
                out_learned_data = dynamics_model.output_pred(ptu.from_numpy(in_train_data)) 
                # fig, ax = plt.subplots()
                sc = ax2.scatter(in_train_data[:,0], in_train_data[:,1], c=out_learned_data[:],
                                vmin=vmin, vmax=vmax) 
                                # vmin=np.min(out_learned_data), vmax=np.max(out_learned_data))
                plt.colorbar(sc)
                ax2.set_title('Generated srf data')
                plt.title('Generated vs Learned srf data')
                # plt.title('Learned srf data')

                plt.show()
                

    ## Set the models for the considered environments
    if not is_local_env:
        env.set_dynamics_model(dynamics_model)
        if args.use_obs_model:
            env.set_observation_model(observation_model)
    elif args.environment == 'hexapod_omni':
        env = HexapodEnv(dynamics_model=dynamics_model,
                         render=False,
                         record_state_action=True,
                         ctrl_freq=100)

    ## Define f_real and f_model
    f_real = env.evaluate_solution # maybe move f_real and f_model inside
    
    ## If we evaluate directly on the real system f_model = f_real 
    if args.perfect_model:
        f_model = f_real
        px['model_variant'] = "dynamics"
    ## Else if we evaluate on the generated random models we use one of below
    elif args.model_variant == "dynamics" :
        if args.model_type == "det":
            f_model = env.evaluate_solution_model 
        elif args.model_type == "prob" and args.environment == 'hexapod_omni':
            f_model = env.evaluate_solution_model_ensemble
    elif args.model_variant == "all_dynamics":
        if args.model_type == "det":
            f_model = env.evaluate_solution_model_all
        elif args.model_type == "det_ens" or args.model_type == "srf_ens":
            f_model = env.evaluate_solution_model_det_ensemble_all
        elif args.model_type == "prob":
            f_model = env.evaluate_solution_model_ensemble_all
    # elif args.model_variant == "direct":
        # f_model = env.evaluate_solution_model 

    ## Change some parameters that are passed to NS/QD instance if using an ensemble
    ## (changes the way behaviors are dumped and how novelty is evaluated) 
    if (args.model_type == "det_ens" or args.model_type == "srf_ens") \
       and not args.perfect_model:
        px["ensemble_dump"] = True
        px["ensemble_size"] = dynamics_model_params["ensemble_size"]

    ## Read the baseline archives obtained with NS (archive size: 4995)
    ns_data_ok = False
    archi = f"{args.c_type}_{args.c_n_layers}l_{args.c_n_neurons}n"
    filename = f'{os.getenv("HOME")}/ns_results/ns_results_{args.environment}_{archi}.dat' ## get archive path
    try:
        ns_data = pd.read_csv(filename)
        ns_data = ns_data.iloc[:,:-1]
        ns_data_ok = True
    except:
        print(f'Could not find file: {filename}. NS baseline won\'t be printed')
        
    ns_data_ok = False
    if ns_data_ok and 'ens' in args.model_type and not args.perfect_model and not args.random_policies and args.ens_size <= 10:
        ## Get real BD data from ns_data
        ns_bd_data = ns_data[['bd0','bd1']].to_numpy()
        ## Load archive genotypes
        gen_cols = [col for col in ns_data.columns if 'x' in col]
        ns_xs = []
        for index, row in ns_data.iterrows():
             ns_xs.append(row[gen_cols].to_numpy())
        to_evaluate = list(zip(ns_xs, itertools.repeat(f_model)))
        ## Evaluate on model(s)
        s_list = evaluate_all_(to_evaluate)
        ## Extract model(s) BD data from s_list
        model_bd_data = [s.desc for s in s_list]
        ## Format the bd data to plot with labels
        all_bd_data = []
        all_bd_data.append((ns_bd_data, 'real system'))
        for m_idx in range(px['ensemble_size']):
            all_bd_data.append(
                (np.array([bd[m_idx*dim_map:m_idx*dim_map+dim_map]
                  for bd in model_bd_data]), f'model n°{m_idx+1}')
            )
        ## Plot real archive and model(s) archive on plot
        total_plots = len(all_bd_data)
        ## make it as square as possible
        rows = cols = round(np.sqrt(total_plots))
        ## Add a row in case closest square cannot take all plots in
        if total_plots > rows*cols:
            rows += 1
        
        fig, axs = plt.subplots(rows, cols)

        bd_cpt = 0
        for col in range(cols):
            for row in range(rows):
                try:
                    loc_bd_data, loc_system_name = all_bd_data[bd_cpt]
                    axs[row][col].scatter(x=loc_bd_data[:,0],y=loc_bd_data[:,1], s=3)
                    axs[row][col].scatter(x=init_obs[0],y=init_obs[1], s=10, c='red')
                    axs[row][col].set_xlabel('x-axis')
                    axs[row][col].set_ylabel('y-axis')
                    axs[row][col].set_title(f'Archive coverage on {loc_system_name}')
                    loc_bd_mins = np.min(loc_bd_data, axis=0) 
                    loc_bd_maxs = np.max(loc_bd_data, axis=0) 
                    axs[row][col].set_xlim(loc_bd_mins[0], loc_bd_maxs[0])
                    axs[row][col].set_ylim(loc_bd_mins[1], loc_bd_maxs[1])
                    axs[row][col].set_aspect('equal', adjustable='box')
                    bd_cpt += 1
                except:
                    fig.delaxes(axs[row][col])
                    
        fig.suptitle('NS archive coverage obtained on real system shown on' \
                     'various dynamic systems', fontsize=16)

        fig.set_size_inches(total_plots*2,total_plots*2)
        file_path = os.path.join(args.log_dir, f"{args.environment}_real_to_model_cov")
        fig.tight_layout(pad=0.5)
        plt.savefig(file_path,
                    dpi=300)#, bbox_inches='tight')

    ## Perform the QD or NS search on f_model
    px['f_target'] = f_real
    px['f_training'] = f_model
    
    if args.algo == 'qd':
        algo = QD(dim_map, dim_x,
                f_model,
                n_niches=1000,
                params=px,
                log_dir=args.log_dir)

    elif args.algo == 'ns':
        algo = NS(dim_map, dim_x,
                  f_model,
                  params=px,
                  log_dir=args.log_dir)
    
    if not args.random_policies:
        model_archive, n_evals = algo.compute(num_cores_set=args.num_cores,
                                              max_evals=args.max_evals*args.multi_eval)
        cm.save_archive(model_archive, f"{n_evals}_model_all", px, args.log_dir)

    else:
        to_evaluate = []
        for i in range(0, args.max_evals):
            x = np.random.uniform(low=px['min'], high=px['max'], size=dim_x)
            to_evaluate += [(x, f_real)]
            n_evals = len(to_evaluate)

    ## Evaluate the found solutions on the model
    if args.perfect_model:
        px['model_variant'] = args.model_variant
        if args.model_variant == "dynamics" :
            if args.model_type == "det":
                f_real = env.evaluate_solution_model 
            elif args.model_type == "prob" and args.environment == 'hexapod_omni':
                f_real = env.evaluate_solution_model_ensemble
        elif args.model_variant == "all_dynamics":
            if args.model_type == "det":
                f_real = env.evaluate_solution_model_all
            elif args.model_type == "det_ens" or args.model_type == "srf_ens":
                px["ensemble_dump"] = True
                px["ensemble_size"] = dynamics_model_params["ensemble_size"]
                f_real = env.evaluate_solution_model_det_ensemble_all
            elif args.model_type == "prob":
                f_real = env.evaluate_solution_model_ensemble_all

    ## Evaluate the found solutions on the real system
    pool = multiprocessing.Pool(args.num_cores)

    ## Select the individuals to transfer onto real system
    ## Selection based on nov, clustering, other?
    
    ## Create to evaluate vector
    if not args.random_policies:
        to_evaluate = list(zip([ind.x.copy() for ind in model_archive], itertools.repeat(f_real)))

    if args.model_type == 'det_ens':
        px['parallel'] = False

    ## Evaluate on real sys or model(if perfect_model on)
    if args.perfect_model and px['model_variant'] == 'all_dynamics':
        s_list = evaluate_all_(to_evaluate)
    else:
        s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, px)

    pool.close()

    if (args.model_type == "det_ens" or args.model_type == "srf_ens") \
       and not args.perfect_model:
        px["ensemble_dump"] = False
    
    # real_archive = []
            
    # real_archive, add_list, _ = addition_condition(s_list, real_archive, px)
    # cm.save_archive(real_archive, f"{n_evals}_real_added", px, args.log_dir)
    cm.save_archive(s_list, f"{n_evals}_real_all", px, args.log_dir)

    ## Plot archive trajectories on model/real system
    # if not args.random_policies and not args.perfect_model:
    if args.log_ind_trajs and not args.random_policies:
        ## Extract real sys BD data from s_list
        real_bd_traj_data = [s.obs_traj for s in s_list]
        ## Format the bd data to plot with labels
        all_bd_traj_data = []

        all_bd_traj_data.append((real_bd_traj_data, 'real system'))
        if px['include_target'] == True:
            px['ensemble_size'] += 1
        for m_idx in range(px['ensemble_size']):
            loc_model_bd_traj_data= []
            for ind in model_archive:
                if not 'ens' in args.model_type or args.perfect_model:
                    bd_traj = ind.obs_traj
                else:
                    bd_traj = ind.obs_traj[:,m_idx]
                loc_model_bd_traj_data.append(bd_traj)

            all_bd_traj_data.append((loc_model_bd_traj_data, f'model n°{m_idx+1}'))

        plot_cov_and_trajs(all_bd_traj_data, args, params)
        
    # Exit if we just checked RP cov
    if args.random_policies:
        exit()
    ## Run again with bootstrap if we ran on model previously 
    if not args.perfect_model:
        ## Now boostrap a NS on the real system with individuals from the previously found archive

        ## Select individuals from model_archive to make a bootstrap archive
        bootstrap_size = px['pop_size']
        if args.bootstrap_selection == 'nov':
            sorted_archive = sorted(model_archive,
                                    key=lambda x:x.nov, reverse=True)
            bootstrap_archive = sorted_archive[:bootstrap_size]
        elif args.bootstrap_selection == 'final_pop':
            bootstrap_archive = algo.population
        elif args.bootstrap_selection == 'random':
            bootstrap_archive = np.random.choice(model_archive, size=bootstrap_size, replace=False)
        else:
            raise ValueError(f'args.bootstrap_selection: {args.bootstrap_selection} is not valid')
        # either take the final population, only make sense with NS
        # Set params['bootstrap_archive'] to an archive containing those individuals
        px['bootstrap_archive'] = bootstrap_archive
        px['model_variant'] = "dynamics"
        px['perfect_model_on'] = True
        # warning! Still need to evaluate them on the real system (done in NS)

        # Instanciate a new NS (to reinit all previously set internal variables)
        algo = NS(dim_map, dim_x,
                  f_real,
                  params=px,
                  log_dir=args.log_dir)
        # Run NS for eval budget on real sys(maybe increase the budget on model to
        # be like twice or thrice real NS one :thought:)

        real_archive, total_evals = algo.compute(num_cores_set=args.num_cores,
                                                 max_evals=args.max_evals)

    else:
        real_archive = model_archive
    
    ## Plot archive coverage at each generation (does not work for QD instances)
    ## will consider a gen = lambda indiv added to archive
    save_archive_cov_by_gen(real_archive, args, px, params)

    print()
    print('Finished dabbing successfully. ¯\_(ツ)_/¯')
################################################################################
############################## Params parsing ##################################
################################################################################
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    parser = argparse.ArgumentParser()
    args = process_args(parser)

    main(args)
