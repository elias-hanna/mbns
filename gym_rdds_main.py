import os, sys
import argparse
import numpy as np

import src.torch.pytorch_util as ptu

from src.map_elites.qd import QD

from src.models.dynamics_models.deterministic_model import DeterministicDynModel
from src.models.surrogate_models.det_surrogate import DeterministicQDSurrogate


#----------controller imports--------#
from model_init_study.controller.nn_controller \
    import NeuralNetworkController

#----------Environment imports--------#
import gym
import diversity_algorithms.environments.env_imports ## Contains deterministic ant + fetch

#----------Utils imports--------#
from multiprocessing import cpu_count
import copy
import numpy as np
import torch
import time
import tqdm

from src.data_management.replay_buffers.simple_replay_buffer import SimpleReplayBuffer

def get_dynamics_model(dynamics_model_params):
    obs_dim = dynamics_model_params['obs_dim']
    action_dim = dynamics_model_params['action_dim']
    
    from src.trainers.mbrl.mbrl_det import MBRLTrainer 
    dynamics_model = DeterministicDynModel(obs_dim=obs_dim,
                                           action_dim=action_dim,
                                           hidden_size=dynamics_model_params['layer_size'])
    dynamics_model_trainer = MBRLTrainer(model=dynamics_model,
                                         batch_size=dynamics_model_params['batch_size'],)

    return dynamics_model, dynamics_model_trainer

def get_surrogate_model(surrogate_model_params):
    from src.trainers.qd.surrogate import SurrogateTrainer
    model = DeterministicQDSurrogate(gen_dim=surrogate_model_params['gen_dim'],
                                     bd_dim=surrogate_model_params['bd_dim'],
                                     hidden_size=surrogate_model_params['layer_size'])
    model_trainer = SurrogateTrainer(model, batch_size=surrogate_model_params['batch_size'])

    return model, model_trainer

class WrappedEnv():
    def __init__(self, params):
        self._action_min = params['action_min']
        self._action_max = params['action_max']
        self._env_max_h = params['env_max_h']
        self._env = params['env']
        self._env_name = params['env_name']
        self._init_obs = self._env.reset()
        self._is_goal_env = False
        if isinstance(self._init_obs, dict):
            self._is_goal_env = True
            self._init_obs = self._init_obs['observation']
        self._obs_dim = params['obs_dim']
        self._act_dim = params['action_dim']
        self.fitness_func = params['fitness_func']
        self.nb_thread = cpu_count() - 1 or 1
        ## Get the controller and initialize it from params
        self.controller = params['controller_type'](params)
        ## Get policy parameters init min and max
        self._policy_param_init_min = params['policy_param_init_min'] 
        self._policy_param_init_max = params['policy_param_init_max']
        ## Get size of policy parameter vector
        self.policy_representation_dim = len(self.controller.get_parameters())
        self.dynamics_model = None
        self.time_open_loop = params['time_open_loop']
        
    def set_dynamics_model(self, dynamics_model):
        self.dynamics_model = dynamics_model

    ## Evaluate the individual on the REAL ENVIRONMENT
    def evaluate_solution(self, ctrl, render=False):
        """
        Input: ctrl (array of floats) the genome of the individual
        Output: Trajectory and actions taken
        """
        ## Create a copy of the controller
        controller = self.controller.copy()
        ## Verify that x and controller parameterization have same size
        # assert len(x) == len(self.controller.get_parameters())
        ## Set controller parameters
        controller.set_parameters(ctrl)
        env = copy.copy(self._env) ## need to verify this works
        obs = env.reset()
        act_traj = []
        obs_traj = []
        ## WARNING: need to get previous obs
        for t in range(self._env_max_h):
            if self._is_goal_env:
                obs = obs['observation']
            if self.time_open_loop:
                action = controller([t])
            else:
                action = controller(obs)
            action[action>self._action_max] = self._action_max
            action[action<self._action_min] = self._action_min
            obs_traj.append(obs)
            act_traj.append(action)
            obs, reward, done, info = self._env.step(action)
            if done:
                break
        if self._is_goal_env:
            obs = obs['observation']
        obs_traj.append(obs)

        desc = self.compute_bd(obs_traj)
        fitness = self.compute_fitness(obs_traj, act_traj)

        if render:
            print("Desc from simulation", desc)

        return fitness, desc, obs_traj, act_traj, 0 # 0 is disagr

    ## Evaluate the individual on the DYNAMICS MODEL
    def evaluate_solution_model(self, ctrl, render=False):
        """
        Input: ctrl (array of floats) the genome of the individual
        Output: Trajectory and actions taken
        """
        ## Create a copy of the controller
        controller = self.controller.copy()
        ## Verify that x and controller parameterization have same size
        # assert len(x) == len(self.controller.get_parameters())
        ## Set controller parameters
        controller.set_parameters(ctrl)
        env = copy.copy(self._env) ## need to verify this works
        obs = self._init_obs
        act_traj = []
        obs_traj = []
        ## WARNING: need to get previous obs
        for t in range(self._env_max_h):
            if self.time_open_loop:
                action = controller([t])
            else:
                action = controller(obs)
            action[action>self._action_max] = self._action_max
            action[action<self._action_min] = self._action_min
            obs_traj.append(obs)
            act_traj.append(action)

            s = ptu.from_numpy(np.array(obs))
            a = ptu.from_numpy(np.array(action))
            s = s.view(1,-1)
            a = a.view(1,-1)

            # Deterministic dynamics model
            pred_delta_ns = self.dynamics_model.output_pred(torch.cat((s, a), dim=-1))

            obs = pred_delta_ns[0] + obs # the [0] just seelect the row [1,state_dim]
        obs_traj.append(obs)

        desc = self.compute_bd(obs_traj)
        fitness = self.compute_fitness(obs_traj, act_traj)

        if render:
            print("Desc from model", desc)

        return fitness, desc, obs_traj, act_traj

    def compute_bd(self, obs_traj):
        bd = None
        last_obs = obs_traj[-1]
                
        if self._env_name == 'ball_in_cup':
            bd = last_obs[:3]
        if self._env_name == 'fastsim_maze':
            bd = last_obs[:2]
        if self._env_name == 'fastsim_maze_traps':
            bd = last_obs[:2]
        if self._env_name == 'redundant_arm_no_walls_limited_angles':
            bd = last_obs[-2:]
        return bd

    def energy_minimization_fit(self, actions, disagrs):
        return -np.sum(np.abs(actions))

    def disagr_minimization_fit(self, actions, disagrs):
        if disagrs is None: # was a real world eval
            return 0
        return -np.sum(disagrs)

    def compute_fitness(self, obs_traj, act_traj, disagr_traj=None, ensemble=False):
        fit = 0
        ## Energy minimization fitness
        if self.fitness_func == 'energy_minimization':
            fit_func = self.energy_minimization_fit
        elif self.fitness_func == 'disagr_minimization':
            fit_func = self.disagr_minimization_fit
        if self._env_name == 'ball_in_cup':
            fit = fit_func(act_traj, disagr_traj)
        if self._env_name == 'fastsim_maze':
            fit = fit_func(act_traj, disagr_traj)
        if self._env_name == 'fastsim_maze_traps':
            fit = fit_func(act_traj, disagr_traj)
        if self._env_name == 'redundant_arm_no_walls_limited_angles':
            fit = fit_func(act_traj, disagr_traj)
        return fit

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
        "parallel": True,
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
        "perfect_model_on": args.perfect_model,
        
        "log_model_stats": False,
        "log_time_stats": False, 

        # 0 for random emiiter, 1 for optimizing emitter
        # 2 for random walk emitter, 3 for model disagreement emitter
        "emitter_selection": 0,

        "transfer_selection": args.transfer_selection,
        "nb_transfer": args.nb_transfer,
        'env_name': args.environment,
    }

    
    #########################################################################
    ####################### Preparation of run ##############################
    #########################################################################
    
    ### Environment initialization ###
    env_register_id = 'BallInCup3d-v0'
    gym_args = {}
    is_local_env = False
    if args.environment == 'ball_in_cup':
        import mb_ge ## Contains ball in cup
        env_register_id = 'BallInCup3d-v0'
        ss_min = -0.4
        ss_max = 0.4
        dim_map = 3
    elif args.environment == 'redundant_arm':
        import redundant_arm ## contains redundant arm
        env_register_id = 'RedundantArmPos-v0'
        ss_min = -1
        ss_max = 1
        dim_map = 2
    elif args.environment == 'redundant_arm_no_walls':
        env_register_id = 'RedundantArmPosNoWalls-v0'
        ss_min = -1
        ss_max = 1
        dim_map = 2
    elif args.environment == 'redundant_arm_no_walls_no_collision':
        env_register_id = 'RedundantArmPosNoWallsNoCollision-v0'
        ss_min = -1
        ss_max = 1
        dim_map = 2
    elif args.environment == 'redundant_arm_no_walls_limited_angles':
        env_register_id = 'RedundantArmPosNoWallsLimitedAngles-v0'
        ss_min = -1
        ss_max = 1
        dim_map = 2
        gym_args['dof'] = 100
    elif args.environment == 'fastsim_maze':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        ss_min = -10
        ss_max = 10
        dim_map = 2
    elif args.environment == 'fastsim_maze_traps':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        ss_min = -10
        ss_max = 10
        dim_map = 2
        gym_args['physical_traps'] = True
    elif args.environment == 'hexapod_omni':
        from src.envs.hexapod_dart.hexapod_env import HexapodEnv ## Contains hexapod 
        is_local_env = True
        max_step = 300 # ctrl_freq = 100Hz, sim_time = 3.0 seconds 
        obs_dim = 48
        act_dim = 18
        dim_x = 36
        ss_min = -1
        ss_max = 1
        dim_map = 2
    else:
        raise ValueError(f"{args.environment} is not a defined environment")
    
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
        
    controller_params = \
    {
        'controller_input_dim': obs_dim,
        'controller_output_dim': act_dim,
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 10,
        'time_open_loop': False,
    }
    dynamics_model_params = \
    {
        'obs_dim': obs_dim,
        'action_dim': act_dim,
        'layer_size': 500,
        'batch_size': 512,
        'learning_rate': 1e-3,
        'train_unique_trans': False,
    }
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
    params = \
    {
        'obs_dim': obs_dim,
        'action_dim': act_dim,

        'controller_type': NeuralNetworkController,
        'controller_params': controller_params,
        'time_open_loop': controller_params['time_open_loop'],
        
        'dynamics_model_params': dynamics_model_params,

        'action_min': -1,
        'action_max': 1,

        'state_min': ss_min,
        'state_max': ss_max,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
        
        'env': gym_env,
        'env_name': args.environment,
        'env_max_h': max_step,
        'fitness_func': args.fitness_func,
    }
    ## Correct obs dim for controller if open looping on time
    if params['time_open_loop']:
        controller_params['obs_dim'] = 1
        
    #########################################################################
    ####################### End of Preparation of run #######################
    #########################################################################


    if not is_local_env:
        env = WrappedEnv(params)
        dim_x = env.policy_representation_dim

    surrogate_model_params['gen_dim'] = dim_x


    dynamics_model, dynamics_model_trainer = get_dynamics_model(dynamics_model_params)
    surrogate_model, surrogate_model_trainer = get_surrogate_model(surrogate_model_params)

    if not is_local_env:
        env.set_dynamics_model(dynamics_model)
    elif args.environment == 'hexapod_omni':
        env = HexapodEnv(dynamics_model=dynamics_model,
                         render=False,
                         record_state_action=True,
                         ctrl_freq=100)
        
    f_real = env.evaluate_solution # maybe move f_real and f_model inside

    if args.perfect_model:
        f_model = f_real
    elif args.model_variant == "dynamics":
        f_model = env.evaluate_solution_model 
    # elif args.model_variant == "direct":
        # f_model = env.evaluate_solution_model 
        
    qd = QD(dim_map, dim_x,
            f_model,
            n_niches=1000,
            params=px,
            log_dir=args.log_dir)

    archive = qd.compute(num_cores_set=args.num_cores, max_evals=args.max_evals)
    
    # mbqd = ModelBasedQD(dim_map, dim_x,
    #                     f_real, f_model,
    #                     surrogate_model, surrogate_model_trainer,
    #                     dynamics_model, dynamics_model_trainer,
    #                     replay_buffer, 
    #                     n_niches=args.n_niches,
    #                     params=px, log_dir=args.log_dir)

    # #mbqd.compute(num_cores_set=cpu_count()-1, max_evals=args.max_evals)
    # mbqd.compute(num_cores_set=args.num_cores, max_evals=args.max_evals)
        

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    parser = argparse.ArgumentParser()
    #-----------------Type of QD---------------------#
    # options are 'cvt', 'grid' and 'unstructured'
    parser.add_argument("--qd_type", type=str, default="unstructured")
    
    #---------------CPU usage-------------------#
    parser.add_argument("--num_cores", type=int, default=6)
    
    #-----------Store results + analysis-----------#
    parser.add_argument("--log_dir", type=str)
    
    #-----------QD params for cvt or GRID---------------#
    # ONLY NEEDED FOR CVT OR GRID MAP ELITES - not needed for unstructured archive
    parser.add_argument("--grid_shape", default=[100,100], type=list) # num discretizat
    parser.add_argument("--n_niches", default=3000, type=int)

    #----------population params--------#
    parser.add_argument("--random-init-batch", default=100, type=int) # Number of inds to initialize the archive
    parser.add_argument("--b_size", default=200, type=int) # For paralellization - 
    parser.add_argument("--dump_period", default=5000, type=int) 
    parser.add_argument("--dump-mode", type=str, default="budget")
    parser.add_argument("--max_evals", default=1e6, type=int) # max number of evaluation
    parser.add_argument("--selector", default="uniform", type=str)
    parser.add_argument("--mutation", default="iso_dd", type=str)

    #-------------DAQD params-----------#
    parser.add_argument('--transfer-selection', type=str, default='all')
    parser.add_argument('--fitness-func', type=str, default='energy_minimization')
    parser.add_argument('--nb-transfer', type=int, default=1)

    parser.add_argument('--model-variant', type=str, default='dynamics')
    parser.add_argument('--perfect-model', action='store_true')

    #----------model init study params--------#
    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')
    parser.add_argument('--rep', type=int, default='1')
    
    args = parser.parse_args()

    main(args)
