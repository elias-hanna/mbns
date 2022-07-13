import os, sys
import argparse
import numpy as np

import src.torch.pytorch_util as ptu

from src.map_elites.mbqd import ModelBasedQD

from src.models.dynamics_models.deterministic_model import DeterministicDynModel
from src.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
from src.models.surrogate_models.det_surrogate import DeterministicQDSurrogate


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

#----------controller imports--------#
from model_init_study.controller.nn_controller \
    import NeuralNetworkController

#----------Separator imports--------#
from model_init_study.visualization.fetch_pick_and_place_separator \
    import FetchPickAndPlaceSeparator
from model_init_study.visualization.ant_separator \
    import AntSeparator
from model_init_study.visualization.ball_in_cup_separator \
    import BallInCupSeparator
from model_init_study.visualization.redundant_arm_separator \
    import RedundantArmSeparator
from model_init_study.visualization.fastsim_separator \
    import FastsimSeparator

#----------Environment imports--------#
import gym
import diversity_algorithms.environments.env_imports ## Contains deterministic ant + fetch
import mb_ge ## Contains ball in cup
import redundant_arm ## contains redundant arm

#----------Utils imports--------#
from multiprocessing import cpu_count
import copy
import numpy as np
import torch

# added in get dynamics model section
#from src.trainers.mbrl.mbrl_det import MBRLTrainer
#from src.trainers.mbrl.mbrl import MBRLTrainer
#from src.trainers.qd.surrogate import SurrogateTrainer

from src.data_management.replay_buffers.simple_replay_buffer import SimpleReplayBuffer

def get_dynamics_model(dynamics_model_type, act_dim, obs_dim):
    obs_dim = obs_dim
    action_dim = act_dim
    
    ## INIT MODEL ##
    if dynamics_model_type == "prob":
        from src.trainers.mbrl.mbrl import MBRLTrainer
        variant = dict(
            mbrl_kwargs=dict(
                ensemble_size=4,
                layer_size=500,
                learning_rate=1e-3,
                batch_size=512,
            )
        )
        M = variant['mbrl_kwargs']['layer_size']
        dynamics_model = ProbabilisticEnsemble(
            ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M]
        )
        dynamics_model_trainer = MBRLTrainer(
            ensemble=dynamics_model,
            **variant['mbrl_kwargs'],
        )

        # ensemble somehow cant run in parallel evaluations
    elif dynamics_model_type == "det":
        from src.trainers.mbrl.mbrl_det import MBRLTrainer 
        dynamics_model = DeterministicDynModel(obs_dim=obs_dim,
                                               action_dim=action_dim,
                                               hidden_size=500)
        dynamics_model_trainer = MBRLTrainer(
            model=dynamics_model,
            batch_size=512,)


    return dynamics_model, dynamics_model_trainer

def get_surrogate_model(dim):
    from src.trainers.qd.surrogate import SurrogateTrainer
    dim_x=dim # genotype dimnesion    
    model = DeterministicQDSurrogate(gen_dim=dim_x, bd_dim=2, hidden_size=64)
    model_trainer = SurrogateTrainer(model, batch_size=32)

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
        self.nb_thread = cpu_count() - 1 or 1
        ## Get the controller and initialize it from params
        self.controller = params['controller_type'](params)
        ## Get policy parameters init min and max
        self._policy_param_init_min = params['policy_param_init_min'] 
        self._policy_param_init_max = params['policy_param_init_max']
        ## Get size of policy parameter vector
        self.policy_representation_dim = len(self.controller.get_parameters())
        self.dynamics_model = None

    def set_dynamics_model(self, dynamics_model):
        self.dynamics_model = dynamics_model
        
    ## For each env, do a BD + Fitness based on traj
    ## mb best solution is to put it in the envs directly
    ## check what obs_traj and act_traj looks like in src/envs/hexapod_env.py
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
            action = controller(obs)
            action[action>self._action_max] = self._action_max
            action[action<self._action_min] = self._action_min
            obs_traj.append(obs)
            act_traj.append(action)
            obs, reward, done, info = self._env.step(action)
            if done:
                break
        # if self._is_goal_env:
            # obs = obs['observation']
        # obs_traj.append(obs)

        desc = self.compute_bd(obs_traj)
        fitness = self.compute_fitness(obs_traj, act_traj)

        if render:
            print("Desc from simulation", desc)

        return fitness, desc, obs_traj, act_traj

    def evaluate_solution_model(self, ctrl, mean=False, det=True, render=False):
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
            action = controller(obs)
            action[action>self._action_max] = self._action_max
            action[action<self._action_min] = self._action_min
            obs_traj.append(obs)
            act_traj.append(action)

            s = ptu.from_numpy(np.array(obs))
            a = ptu.from_numpy(np.array(action))
            s = s.view(1,-1)
            a = a.view(1,-1)

            if det:
                # if deterministic dynamics model
                pred_delta_ns = self.dynamics_model.output_pred(torch.cat((s, a), dim=-1))
            else:
                # if probalistic dynamics model - choose output mean or sample
                pred_delta_ns = self.dynamics_model.output_pred(torch.cat((s, a), dim=-1), mean=mean)
            obs = pred_delta_ns[0] + obs # the [0] just seelect the row [1,state_dim]
            
        obs_traj.append(obs)

        desc = self.compute_bd(obs_traj)
        fitness = self.compute_fitness(obs_traj, act_traj)

        if render:
            print("Desc from model", desc)

        return fitness, desc, obs_traj, act_traj

    def evaluate_solution_model_ensemble(self, ctrl, mean=True, disagr=True, render=False):
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
        disagr_traj = []
        obs = np.tile(obs,(self.dynamics_model.ensemble_size, 1))
        ## WARNING: need to get previous obs
        for t in range(self._env_max_h):
            ## Get mean obs to determine next action
            # mean_obs = [np.mean(obs[:,i]) for i in range(len(obs[0]))]
            action = controller(obs)
            action[action>self._action_max] = self._action_max
            action[action<self._action_min] = self._action_min
            # if t == 0:
                # obs = np.tile(obs,(self.dynamics_model.ensemble_size, 1))
            obs_traj.append(obs)
            act_traj.append(action)

            s = ptu.from_numpy(np.array(obs))
            a = ptu.from_numpy(np.array(action))

            # if t ==0:
                # a = a.repeat(self.dynamics_model.ensemble_size,1)
            
            # if probalistic dynamics model - choose output mean or sample
            if disagr:
                pred_delta_ns, _ = self.dynamics_model.sample_with_disagreement(torch.cat((
                    self.dynamics_model._expand_to_ts_form(s),
                    self.dynamics_model._expand_to_ts_form(a)), dim=-1))#,
                    # disagreement_type="mean" if mean else "var")
                pred_delta_ns = ptu.get_numpy(pred_delta_ns)
                disagreement = self.compute_abs_disagreement(obs, pred_delta_ns)
                # print("Disagreement: ", disagreement.shape)
                # print("Disagreement: ", disagreement)
                disagreement = ptu.get_numpy(disagreement) 
                #disagreement = ptu.get_numpy(disagreement[0,3]) 
                #disagreement = ptu.get_numpy(torch.mean(disagreement)) 
                disagr_traj.append(disagreement)
                
            else:
                pred_delta_ns = self.dynamics_model.output_pred_ts_ensemble(s,a, mean=mean)

            # mean_pred = [np.mean(pred_delta_ns[:,i]) for i in range(len(pred_delta_ns[0]))]
            
            obs = pred_delta_ns + obs # This keeps all model predictions separated
            # obs = mean_pred + obs # This uses mean prediction

        # obs_traj.append(obs)

        obs_traj = np.array(obs_traj)
        act_traj = np.array(act_traj)
        
        desc = self.compute_bd(obs_traj, ensemble=True)
        fitness = self.compute_fitness(obs_traj, act_traj, ensemble=True)

        if render:
            print("Desc from model", desc)

        return fitness, desc, obs_traj, act_traj, disagr_traj

    def compute_abs_disagreement(self, cur_state, pred_delta_ns):
        '''
        Computes absolute state dsiagreement between models in the ensemble
        cur state is [4,48]
        pred delta ns [4,48]
        '''
        next_state = pred_delta_ns + cur_state
        next_state = ptu.from_numpy(next_state)
        mean = next_state

        sample=False
        if sample: 
            inds = torch.randint(0, mean.shape[0], next_state.shape[:1]) #[4]
            inds_b = torch.randint(0, mean.shape[0], next_state.shape[:1]) #[4]
            inds_b[inds == inds_b] = torch.fmod(inds_b[inds == inds_b] + 1, mean.shape[0]) 
        else:
            inds = torch.tensor(np.array([0,0,0,1,1,2]))
            inds_b = torch.tensor(np.array([1,2,3,2,3,3]))

        # Repeat for multiplication
        inds = inds.unsqueeze(dim=-1).to(device=ptu.device)
        inds = inds.repeat(1, mean.shape[1])
        inds_b = inds_b.unsqueeze(dim=-1).to(device=ptu.device)
        inds_b = inds_b.repeat(1, mean.shape[1])

        means_a = (inds == 0).float() * mean[0]
        means_b = (inds_b == 0).float() * mean[0]
        for i in range(1, mean.shape[0]):
            means_a += (inds == i).float() * mean[i]
            means_b += (inds_b == i).float() * mean[i]
            
        disagreements = torch.mean(torch.sqrt((means_a - means_b)**2), dim=-2, keepdim=True)
        #disagreements = torch.mean((means_a - means_b) ** 2, dim=-1, keepdim=True)

        return disagreements

    def compute_bd(self, obs_traj, ensemble=False, mean=True):
        bd = None
        last_obs = obs_traj[-1]
        if ensemble:
            if mean:
                last_obs = np.mean(last_obs, axis=0)
            else:
                last_obs = last_obs[np.random.randint(self.dynamics_model.ensemble_size)]
                
        if self._env_name == 'ball_in_cup':
            bd = last_obs[:3]
        if self._env_name == 'fastsim_maze':
            bd = last_obs[:2]
        if self._env_name == 'fastsim_maze_traps':
            bd = last_obs[:2]
        if self._env_name == 'redundant_arm_no_walls_limited_angles':
            bd = last_obs[-2:]
        return bd

    def compute_fitness(self, obs_traj, act_traj, ensemble=False):
        fit = 0
        if self._env_name == 'ball_in_cup':
            fit = 0
        if self._env_name == 'fastsim_maze':
            fit = 0
        if self._env_name == 'fastsim_maze_traps':
            fit = 0
        if self._env_name == 'redundant_arm_no_walls_limited_angles':
            fit = 0
            
        return fit

def main(args):

    px = \
    {
        # type of qd 'unstructured, grid, cvt'
        "type": args.qd_type,
        
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to parallelize
        "batch_size": args.b_size,
        # proportion of total number of niches to be filled before starting
        "random_init": 0.005,  
        # batch for random initialization
        "random_init_batch": 100,
        # when to write results (one generation = one batch)
        "dump_period": args.dump_period,

        # do we use several cores?
        "parallel": True,
        # min/max of genotype parameters - check mutation operators too
        # "min": 0.0,
        # "max": 1.0,
        "min": -5,
        "max": 5,
        
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

        "model_variant": "dynamics", #"direct", # "dynamics" or "direct"  
        "train_model_on": True, #                                                                              
        "train_freq": 40, # train at a or condition between train freq and evals_per_train
        "evals_per_train": 500,
        "log_model_stats": False,
        "log_time_stats": False, 

        # 0 for random emiiter, 1 for optimizing emitter
        # 2 for random walk emitter, 3 for model disagreement emitter
        "emitter_selection": 0,

        "transfer_selection": args.transfer_selection,
        'env_name': args.environment,
        
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
    elif args.init_method == 'colored-noise-beta-1':
        Initializer = ColoredNoiseMotion
        noise_beta = 1
    elif args.init_method == 'colored-noise-beta-2':
        Initializer = ColoredNoiseMotion
        noise_beta = 2
    else:
        raise Exception(f"Warning {args.init_method} isn't a valid initializer")
    
    
    ### Environment initialization ###
    env_register_id = 'BallInCup3d-v0'
    gym_args = {}
    if args.environment == 'ball_in_cup':
        env_register_id = 'BallInCup3d-v0'
        separator = BallInCupSeparator
        ss_min = -0.4
        ss_max = 0.4
    elif args.environment == 'redundant_arm':
        env_register_id = 'RedundantArmPos-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
    elif args.environment == 'redundant_arm_no_walls':
        env_register_id = 'RedundantArmPosNoWalls-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
    elif args.environment == 'redundant_arm_no_walls_no_collision':
        env_register_id = 'RedundantArmPosNoWallsNoCollision-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
    elif args.environment == 'redundant_arm_no_walls_limited_angles':
        env_register_id = 'RedundantArmPosNoWallsLimitedAngles-v0'
        separator = RedundantArmSeparator
        ss_min = -1
        ss_max = 1
    elif args.environment == 'fastsim_maze':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        separator = FastsimSeparator
        ss_min = -10
        ss_max = 10
    elif args.environment == 'fastsim_maze_traps':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        separator = FastsimSeparator
        ss_min = -10
        ss_max = 10
        gym_args['physical_traps'] = True
    else:
        raise ValueError(f"{args.environment} is not a defined environment")
    
    
    # if args.environment == 'fetch_pick_and_place':
    #     env_register_id = 'FetchPickAndPlaceDeterministic-v1'
    #     separator = FetchPickAndPlaceSeparator
    #     ss_min = -1
    #     ss_max = 1
    # if args.environment == 'ant':
    #     env_register_id = 'AntBulletEnvDeterministicPos-v0'
    #     separator = AntSeparator
    #     ss_min = -10
    #     ss_max = 10
        
    gym_env = gym.make(env_register_id, **gym_args)

    try:
        max_step = gym_env._max_episode_steps
    except:
        try:
            max_step = gym_env.max_steps
        except:
            raise AttributeError("Env does not allow access to _max_episode_steps or to max_steps")

    obs = gym_env.reset()
    if isinstance(obs, dict):
        obs_dim = gym_env.observation_space['observation'].shape[0]
    else:
        obs_dim = gym_env.observation_space.shape[0]
    act_dim = gym_env.action_space.shape[0]

    controller_params = \
    {
        'controller_input_dim': obs_dim,
        'controller_output_dim': act_dim,
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 10
    }
    dynamics_model_params = \
    {
        'obs_dim': obs_dim,
        'action_dim': act_dim,
        'dynamics_model_type': 'prob', # possible values: prob, det
        'ensemble_size': 4, # only used if dynamics_model_type == prob
        'layer_size': 500,
        'batch_size': 512,
        'learning_rate': 1e-3,
        'train_unique_trans': False,
    }
    params = \
    {
        'obs_dim': obs_dim,
        'action_dim': act_dim,

        'separator': separator,
        
        'n_init_episodes': args.init_episodes,
        # 'n_test_episodes': int(.2*args.init_episodes), # 20% of n_init_episodes
        'n_test_episodes': 2,
        
        'controller_type': NeuralNetworkController,
        'controller_params': controller_params,

        'dynamics_model_params': dynamics_model_params,

        'action_min': -1,
        'action_max': 1,
        'action_init': 0,

        ## Random walks parameters
        'step_size': 0.1,
        'noise_beta': noise_beta,
        
        'action_lasting_steps': 5,

        'state_min': ss_min,
        'state_max': ss_max,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
        
        # 'dump_path': args.dump_path,
        # 'path_to_test_trajectories': 'examples/'+args.environment+'_example_trajectories.npz',
        # 'path_to_test_trajectories': path_to_examples,

        'env': gym_env,
        'env_name': args.environment,
        'env_max_h': max_step,
    }

    #########################################################################
    ####################### End of Preparation of run #######################
    #########################################################################

    env = WrappedEnv(params)

    dim_x = env.policy_representation_dim
    obs_dim = obs_dim
    action_dim = act_dim
    
    # Deterministic = "det", Probablistic = "prob" 
    dynamics_model_type = "prob"

    print("Dynamics model type: ", dynamics_model_type) 
    dynamics_model, dynamics_model_trainer = get_dynamics_model(dynamics_model_type,
                                                                action_dim, obs_dim)
    surrogate_model, surrogate_model_trainer = get_surrogate_model(dim_x)

    ## Initialize model with wnb from previous run if an init method is to be used
    if args.init_method != 'no-init':
        path = f'data/{args.environment}_results/{args.rep}/'\
               f'{args.environment}_{args.init_method}_{args.init_episodes}_model_wnb.pt'
        dynamics_model.load_state_dict(torch.load(path))
        dynamics_model.eval()
        env.set_dynamics_model(dynamics_model)
    
    f_real = env.evaluate_solution # maybe move f_real and f_model inside

    if dynamics_model_type == "det":
        f_model = env.evaluate_solution_model 
    elif dynamics_model_type == "prob":
        f_model = env.evaluate_solution_model_ensemble
        
    # initialize replay buffer
    replay_buffer = SimpleReplayBuffer(
        max_replay_buffer_size=1000000,
        observation_dim=obs_dim,
        action_dim=action_dim,
        env_info_sizes=dict(),
    )
    
    mbqd = ModelBasedQD(args.dim_map, dim_x,
                        f_real, f_model,
                        surrogate_model, surrogate_model_trainer,
                        dynamics_model, dynamics_model_trainer,
                        replay_buffer, 
                        n_niches=args.n_niches,
                        params=px, log_dir=args.log_dir)

    mbqd.compute(num_cores_set=args.num_cores, max_evals=args.max_evals)
        

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    parser = argparse.ArgumentParser()
    #-----------------Type of QD---------------------#
    # options are 'cvt', 'grid' and 'unstructured'
    parser.add_argument("--qd_type", type=str, default="unstructured")
    
    #---------------CPU usage-------------------#
    parser.add_argument("--num_cores", type=int, default=8)
    
    #-----------Store results + analysis-----------#
    parser.add_argument("--log_dir", type=str)
    
    #-----------QD params for cvt or GRID---------------#
    # ONLY NEEDED FOR CVT OR GRID MAP ELITES - not needed for unstructured archive
    parser.add_argument("--dim_map", default=2, type=int) # Dim of behaviour descriptor
    parser.add_argument("--grid_shape", default=[100,100], type=list) # num discretizat
    parser.add_argument("--n_niches", default=3000, type=int)

    #----------population params--------#
    parser.add_argument("--b_size", default=200, type=int) # For paralellization - 
    parser.add_argument("--dump_period", default=5000, type=int) 
    parser.add_argument("--max_evals", default=1e6, type=int) # max number of evaluation
    parser.add_argument("--selector", default="uniform", type=str)
    parser.add_argument("--mutation", default="iso_dd", type=str)

    #----------model init study params--------#
    parser.add_argument('--environment', '-e', type=str, default='ball_in_cup')
    parser.add_argument('--init-method', type=str, default='random-policies')
    parser.add_argument('--init-episodes', type=int, default='10')
    parser.add_argument('--rep', type=int, default='1')
    parser.add_argument('--transfer-selection', type=str, default='all')

    args = parser.parse_args()

    main(args)
