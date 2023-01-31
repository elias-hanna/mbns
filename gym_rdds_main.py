#----------Algo imports--------#
from src.map_elites import common as cm
from src.map_elites import unstructured_container, cvt
from src.map_elites.qd import QD
from src.map_elites.ns import NS

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
from model_init_study.controller.controller import Controller ## Superclass for controllers
from model_init_study.controller.nn_controller \
    import NeuralNetworkController

#----------Environment imports--------#
import gym
import diversity_algorithms.environments.env_imports ## Contains deterministic ant + fetch

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

max_obs = None
min_obs = None
################################################################################
################################ QD methods ####################################
################################################################################
def addition_condition(s_list, archive, params):
    add_list = [] # list of solutions that were added
    discard_list = []
    for s in s_list:
        if params['type'] == "unstructured":
            success = unstructured_container.add_to_archive(s, archive, params)
        else:
            success = cvt.add_to_archive(s, s.desc, archive, kdt)
        if success:
            add_list.append(s)
        else:
            discard_list.append(s) #not important for alogrithm but to collect stats

    return archive, add_list, discard_list

def evaluate_(t):
    # evaluate a single vector (x) with a function f and return a species
    # evaluate z with function f - z is the genotype and f is the evalution function
    # t is the tuple from the to_evaluate list
    z, f = t
    fit, desc, obs_traj, act_traj, disagr = f(z) 
    ## warning: commented the lines below, as in my case I don't see the use..
    # becasue it somehow returns a list in a list (have to keep checking sometimes)
    # desc = desc[0] # important - if not it fails the KDtree for cvt and grid map elites
    # desc_ground = desc
    # return a species object (containing genotype, descriptor and fitness)
    # return cm.Species(z, desc, fit, obs_traj=None, act_traj=None)
    return cm.Species(z, desc, fit, obs_traj=obs_traj, act_traj=act_traj)

def evaluate_all_(T):
    # same as the above evaluate but this takes in the disagreement also
    # - useful if you want to make use disargeement value
    # needs two types because no such thing as disagreemnt for real eval
    Z = [T[i][0] for i in range(len(T))]
    f = T[0][1]
    fit_list, desc_list, obs_traj_list, act_traj_list, disagr_list = f(Z) 
    
    # becasue it somehow returns a list in a list (have to keep checking sometimes)
    # desc = desc[0] # important - if not it fails the KDtree for cvt and grid map elites
    
    # return a species object (containing genotype, descriptor and fitness)
    inds = []
    for i in range(len(T)):
        inds.append(cm.Species(Z[i], desc_list[i], fit_list[i], obs_traj=obs_traj_list[i],
                               act_traj=act_traj_list[i], model_dis=disagr_list[i]))
    return inds

################################################################################
############################## Model methods ###################################
################################################################################

def get_dynamics_model(params):
    dynamics_model_params = params['dynamics_model_params']
    obs_dim = dynamics_model_params['obs_dim']
    action_dim = dynamics_model_params['action_dim']
    dynamics_model_type = dynamics_model_params['model_type']
    use_minmax_norm = False if params['pretrain'] != '' else True
    ## INIT MODEL ##
    if dynamics_model_type == "prob":
        from src.trainers.mbrl.mbrl import MBRLTrainer
        variant = dict(
            mbrl_kwargs=dict(
                ensemble_size=dynamics_model_params['ensemble_size'],
                layer_size=dynamics_model_params['layer_size'],
                learning_rate=dynamics_model_params['learning_rate'],
                batch_size=dynamics_model_params['batch_size'],
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
        dynamics_model = DeterministicDynModel(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_size=dynamics_model_params['layer_size'],
                sa_min=np.concatenate((params['state_min'],
                                       params['action_min'])),
                sa_max=np.concatenate((params['state_max'],
                                       params['action_max'])),
                use_minmax_norm=use_minmax_norm)
        dynamics_model_trainer = MBRLTrainer(
            model=dynamics_model,
            learning_rate=dynamics_model_params['learning_rate'],
            batch_size=dynamics_model_params['batch_size'],)

    elif dynamics_model_type == "det_ens":
        from src.trainers.mbrl.mbrl_det import MBRLTrainer 
        dynamics_model = DeterministicEnsemble(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=dynamics_model_params['layer_size'],
            ensemble_size=dynamics_model_params['ensemble_size'],
            sa_min=np.concatenate((params['state_min'],
                                   params['action_min'])),
            sa_max=np.concatenate((params['state_max'],
                                   params['action_max'])),
            use_minmax_norm=use_minmax_norm)
        ## warning same trainer for all (need to call it n times) 
        # dynamics_model_trainer = MBRLTrainer(
        #     model=dynamics_model,
        #     learning_rate=dynamics_model_params['learning_rate'],
        #     batch_size=dynamics_model_params['batch_size'],)
        dynamics_model_trainer = None
    elif dynamics_model_type == "srf_ens":
        dynamics_model = SrfDeterministicEnsemble(
            obs_dim=obs_dim,
            action_dim=action_dim,
            sa_min=np.concatenate((params['state_min'],
                                   params['action_min'])),
            sa_max=np.concatenate((params['state_max'],
                                   params['action_max'])),
            var=params['srf_var'],
            len_scale=params['srf_cor'],
            ensemble_size=dynamics_model_params['ensemble_size'],
            use_minmax_norm=use_minmax_norm)
        dynamics_model_trainer = None
    return dynamics_model, dynamics_model_trainer

def get_observation_model(params):
    observation_model_params = params['observation_model_params']
    state_dim = observation_model_params['state_dim']
    obs_dim = observation_model_params['obs_dim']
    observation_model_type = observation_model_params['obs_model_type']
    use_minmax_norm = False if params['pretrain'] != '' else True

    if observation_model_params['obs_model_type'] == 'nn':
        observation_model = DeterministicObsModel(
            state_dim=state_dim,
            obs_dim=obs_dim,
            hidden_size=observation_model_params['layer_size'],
            so_min=params['state_min'],
            so_max=params['state_max'],
            use_minmax_norm=use_minmax_norm)

    elif observation_model_params['obs_model_type'] == 'srf':
        observation_model = SrfDeterministicObsModel(
            state_dim=state_dim,
            obs_dim=obs_dim,
            s_min=params['state_min'],
            s_max=params['state_max'],
            o_min=params['obs_min'],
            o_max=params['obs_max'],
            var=params['srf_var'],
            len_scale=params['srf_cor'],
            use_minmax_norm=use_minmax_norm)
        
    return observation_model

def get_surrogate_model(surrogate_model_params):
    from src.trainers.qd.surrogate import SurrogateTrainer
    model = DeterministicQDSurrogate(
            gen_dim=surrogate_model_params['gen_dim'],
            bd_dim=surrogate_model_params['bd_dim'],
            hidden_size=surrogate_model_params['layer_size'])
    model_trainer = SurrogateTrainer(
            model,
            batch_size=surrogate_model_params['batch_size'])

    return model, model_trainer

class RNNLinearOutput(torch.nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 num_layers,
                 pred_mode):
        super(RNNLinearOutput, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.pred_mode = pred_mode
        
        self.rnn = torch.nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                nonlinearity='tanh',
                                batch_first=True)
        
        self.fc_out = torch.nn.Linear(self.hidden_size, self.output_size)

        # input h of shape (num layers, batch_size, hidden_size)
        self.prev_h = None

        ## First need to get n_params
        self.layers_sizes = []
        self.params = self.get_parameters()
        self.n_params = len(self.params)

    def set_parameters(self, flat_parameters):
        self.prev_h = None # reset prev h since we change NN params
        self.params = flat_parameters
        ## ih layers
        # first ih layer has shape (hidden_size, input_size)
        # all other ih layers have shape (hidden_size, hidden_size)
        ## hh layers: all hh layers have shape (hidden_size, hidden_size)
        ## Biases: all biases have shape (hidden_size)
        assert len(flat_parameters) == self.n_params
        layers_names = copy.copy(self.rnn._flat_weights_names)
        layers_names += ['fc_out']
        params_cpt = 0
        
        for (layer_name, layer_size) in zip(layers_names, self.layers_sizes):
            flat_to_set = flat_parameters[params_cpt:params_cpt+layer_size]
            params_cpt += layer_size

            if 'weight' in layer_name:
                if 'ih_l0' in layer_name:
                    to_set = np.resize(flat_to_set, (self.hidden_size, self.input_size))
                else:
                    to_set = np.resize(flat_to_set, (self.hidden_size, self.hidden_size))
            elif 'bias' in layer_name:
                to_set = np.resize(flat_to_set, (self.hidden_size))
            elif 'fc_out' in layer_name:
                to_set = np.resize(flat_to_set, (self.output_size, self.hidden_size))

            to_set = ptu.from_numpy(to_set)
            with torch.no_grad():
                if 'fc_out' in layer_name:
                    setattr(self.fc_out, 'weight', torch.nn.Parameter(to_set))
                else:
                    setattr(self.rnn, layer_name, torch.nn.Parameter(to_set))
        
    def get_parameters(self):
        layers_flat_weights = copy.copy(self.rnn._flat_weights)
        layers_names = copy.copy(self.rnn._flat_weights_names)
        layers_flat_weights += [self.fc_out.weight]
        layers_names += ['fc_out']
        self.layers_sizes = []
        flat_weights = []
        for (layer_flat_weights, layer_name) in zip(layers_flat_weights, layers_names):
            loc_layer_flat_weights = list(ptu.get_numpy(layer_flat_weights).flatten())
            flat_weights += loc_layer_flat_weights
            self.layers_sizes.append(len(loc_layer_flat_weights))
        return flat_weights

    def predict(self, x, h0=None):
        ## transform x from list to np array
        x = np.array(x)
        ## If x shape is dim 1 -> sequence with a single element
        ## If x shape is dim 2 -> either batch of sequences with single element
        ## or single sequence with multiple elements
        ## If x shape is dim 3 -> batch of sequences with multiple elements
        if self.pred_mode == 'single':
            if len(x.shape) == 2:
                x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
            elif len(x.shape) == 1:
                x = np.reshape(x, (1, len(x)))
            else:
                raise RuntimeError('RNNLinearOutput predict error: x of shape 3 but pred_model is "single"')
        ## Transform x from np array to torch tensor
        x = ptu.from_numpy(x)
        if self.prev_h is not None:
            output, hn = self.rnn(x, self.prev_h)
        else:
            output, hn = self.rnn(x) ## h0 is 0 tensor in this case
        self.prev_h = hn
        ## output contains the final hidden state for each element in the sequence
        ## hn contains the final hidden state for the last element in the sequence for each layer
        h = output[-1]

        out = self.fc_out(h)
        out = ptu.get_numpy(out)
        return out

class RNNController(Controller):
    def __init__(self, params):
        super().__init__(params)
        controller_params = params['controller_params']
        self.hidden_size = controller_params['n_neurons_per_hidden']
        self.n_hidden_layers = controller_params['n_hidden_layers']
        self.pred_mode = controller_params['pred_mode']

        self.rnnlo = RNNLinearOutput(self.input_dim,
                                     self.output_dim,
                                     self.hidden_size,
                                     self.n_hidden_layers,
                                     self.pred_mode)
        self.params = self.get_parameters()
        self.n_params = len(self.params)
        
    def set_parameters(self, flat_parameters):
        assert len(flat_parameters) == self.n_params
        self.rnnlo.set_parameters(flat_parameters)
    
    def get_parameters(self):
        return self.rnnlo.get_parameters()

    def __call__(self, x):
        """Calling the controller calls predict"""
        return self.rnnlo.predict(x)
    
    
class WrappedEnv():
    def __init__(self, params):
        self._action_min = params['action_min']
        self._action_max = params['action_max']
        self._state_min = params['state_min']
        self._state_max = params['state_max']
        self._obs_min = params['obs_min']
        self._obs_max = params['obs_max']
        self._sa_min = np.concatenate((params['state_min'], params['action_min']))
        self._sa_max = np.concatenate((params['state_max'], params['action_max']))
        self._dim_map = params['dim_map']
        self._env_max_h = params['env_max_h']
        self._env = params['env']
        self._env_name = params['env_name']
        self._init_obs = self._env.reset()
        self._is_goal_env = False
        if isinstance(self._init_obs, dict):
            self._is_goal_env = True
            self._init_obs = self._init_obs['observation']
        if 'init_obs' in params:
            self._init_obs = params['init_obs']
        self._state_dim = params['state_dim']
        self._obs_dim = params['obs_dim']
        self._act_dim = params['action_dim']
        self.fitness_func = params['fitness_func']
        self.nb_thread = cpu_count() - 1 or 1
        ## Get the controller and initialize it from params
        self.controller = params['controller_type'](params)
        self.pred_mode = params['controller_params']['pred_mode']
        self.time_open_loop = params['time_open_loop']
        self._norm_c_input = params['controller_params']['norm_input']
        ## Get policy parameters init min and max
        self._policy_param_init_min = params['policy_param_init_min'] 
        self._policy_param_init_max = params['policy_param_init_max']
        ## Get size of policy parameter vector
        self.policy_representation_dim = len(self.controller.get_parameters())
        ## Dynamics model parameters
        self.dynamics_model = None
        self.observation_model = None
        self.use_obs_model = params['use_obs_model']
        self._model_max_h = params['dynamics_model_params']['model_horizon']
        self._ens_size = params['dynamics_model_params']['ensemble_size']
        self.n_wps = params['n_waypoints']
        self.log_ind_trajs = params["log_ind_trajs"]
        self.clip_obs = params['clip_obs']
        self.clip_state = params['clip_state']
        print('###############################################################')
        print('################ Environment parameters: ######################')
        print(f'###### - env name:        {self._env_name}                    ')
        print(f'###### - task horizon:    {self._env_max_h}                   ')
        print(f'###### - model horizon:    {self._model_max_h}                ')
        print(f'###### - controller type: {params["controller_type"]}         ')
        print('###############################################################')

    def set_dynamics_model(self, dynamics_model):
        self.dynamics_model = dynamics_model

    def set_observation_model(self, observation_model):
        self.observation_model = observation_model
        
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
        env = copy.copy(self._env)
        obs = env.reset()
        act_traj = []
        obs_traj = []
        info_traj = []
        cum_act = np.zeros((self._act_dim))
        cum_delta_pos = np.zeros((2,))
        c_input_traj = []
        ## WARNING: need to get previous obs
        for t in range(self._env_max_h):
            if self._is_goal_env:
                obs = obs['observation']
            c_input = None
            if self.time_open_loop:
                if self._norm_c_input:
                    c_input = [(t/self._env_max_h)*(1+1) - 1]
                else:
                    c_input = [t]
            else:
                if self._norm_c_input:
                    c_input = self.normalize_inputs_o_minmax(obs)
                else:
                    c_input = obs
                    
            if self.pred_mode == 'single':
                action = controller(c_input)
            elif self.pred_mode == 'all':
                action = controller(c_input_traj)
            elif self.pred_mode == 'window':
                action = controller(c_input_traj[-10:])
            
            action = np.clip(action, self._action_min, self._action_max)
            cum_act += action
            # if 'fastsim' in self._env_name or 'maze' in self._env_name:
            #     vr = action[0]
            #     vl = action[1]
            #     l = 10
            #     delta_x = ( ( (vr+vl)/(vr-vl) ) * l/2 ) * np.sin((vr-vl)/l)
            #     delta_y = ( ( (vr+vl)/(vr-vl) ) * l/2 ) * (np.cos((vr-vl)/l) +1)
            #     cum_delta_pos += np.array([delta_x, delta_y])
                
            # action = np.random.uniform(low=-1, high=1, size=(self._act_dim,))
            # action[action>self._action_max] = self._action_max
            # action[action<self._action_min] = self._action_min
            obs_traj.append(obs)
            c_input_traj.append(c_input)
            act_traj.append(action)
            obs, reward, done, info = env.step(action)
            info_traj.append(info)
            # print(np.array(obs_traj[-1]) - np.array(obs))
            if done:
                break
        if self._is_goal_env:
            obs = obs['observation']
        obs_traj.append(obs)

        desc = self.compute_bd(obs_traj)
        if 'maze' in self._env_name:
            try:
                wp_idxs = [i for i in range(len(info_traj)//self.n_wps, len(info_traj),
                                            len(info_traj)//self.n_wps)][:self.n_wps-1]
            except:
                import pdb; pdb.set_trace()
            wp_idxs += [-1]

            info_wps = np.take(info_traj, wp_idxs, axis=0)
            desc = []
            for info_wp in info_wps:
                desc += info_wp['robot_pos'][:2]
            obs_traj = np.array([info['robot_pos'][:2] for info in info_traj])

        fitness = self.compute_fitness(obs_traj, act_traj)

        if render:
            print("Desc from simulation", desc)


        ## snippet to gather min and max obs
        # global max_obs, min_obs
        # obs_traj = np.array(obs_traj)
        # if max_obs is None:
        #     max_obs = np.max(obs_traj, axis=0)
        # else:
        #     max_obs = np.reshape(max_obs, (1,-1))
        #     max_obs = np.max(np.concatenate((max_obs, obs_traj), axis=0), axis=0)
        # if min_obs is None:
        #     min_obs = np.min(obs_traj, axis=0)
        # else:
        #     min_obs = np.reshape(min_obs, (1,-1))
        #     min_obs = np.min(np.concatenate((min_obs, obs_traj), axis=0), axis=0)
        if not self.log_ind_trajs:
            obs_traj = None
            act_traj = None

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
        # for t in range(self._env_max_h):
        for t in range(self._model_max_h):
            c_input = None
            if self.time_open_loop:
                if self._norm_c_input:
                    c_input = [(t/self._env_max_h)*(1+1) - 1]
                else:
                    c_input = [t]
            else:
                if self._norm_c_input:
                    c_input = self.normalize_inputs_s_minmax(obs)
                else:
                    c_input = obs
                if self.use_obs_model:
                    c_input = self.observation_model.output_pred(ptu.from_numpy(c_input))
            if self.pred_mode == 'single':
                action = controller(c_input)
            elif self.pred_mode == 'all':
                action = controller(c_input_traj)
            elif self.pred_mode == 'window':
                action = controller(c_input_traj[-10:])

            action = np.clip(action, self._action_min, self._action_max)
            obs_traj.append(obs)
            c_input_traj.append(c_input)
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
            
        disagr = 0

        if not self.log_ind_trajs:
            obs_traj = None
            act_traj = None

        return fitness, desc, obs_traj, act_traj, disagr

    def evaluate_solution_model_all(self, ctrls, render=False):
        """
        Input: ctrl (array of floats) the genome of the individual
        Output: Trajectory and actions taken
        """
        controller_list = []
        traj_list = []
        actions_list = []
        disagreements_list = []
        obs_list = []
        c_input_traj_list = []
        
        env = copy.copy(self._env) ## need to verify this works
        obs = self._init_obs

        cpt = 0
        for ctrl in ctrls:
            ## Create a copy of the controller
            controller_list.append(self.controller.copy())
            ## Set controller parameters
            controller_list[-1].set_parameters(ctrl)
            traj_list.append([])
            traj_list[cpt].append(obs.copy())
            cpt += 1
            c_input_traj_list.append([])
            actions_list.append([])
            disagreements_list.append([])
            obs_list.append(obs.copy())

        ## WARNING: need to get previous obs
        # S = np.tile(prev_element.trajectory[-1].copy(), (len(X)))
        S = np.tile(obs, (len(ctrls), 1))
        A = np.empty((len(ctrls), self.controller.output_dim))

        for t in tqdm.tqdm(range(self._model_max_h), total=self._model_max_h):
            for i in range(len(ctrls)):
                c_input = None
                if self.time_open_loop:
                    if self._norm_c_input:
                        c_input = [(t/self._env_max_h)*(1+1) - 1]
                    else:
                        c_input = [t]
                else:
                    if self._norm_c_input:
                        c_input = self.normalize_inputs_s_minmax(S[i])
                    else:
                        c_input = S[i]
                    if self.use_obs_model:
                        c_input = self.observation_model.output_pred(ptu.from_numpy(c_input))
                        if self.clip_obs:
                            c_input = np.clip(c_input, self._obs_min, self._obs_max)
                        c_input = self.normalize_inputs_o_minmax(c_input)
                        
                if self.pred_mode == 'single':
                    A[i] = controller_list[i](c_input)
                elif self.pred_mode == 'all':
                    A[i] = controller_list[i](c_input_traj_list[i])
                elif self.pred_mode == 'window':
                    A[i] = controller_list[i](c_input_traj_list[i][-10:])

                c_input_traj_list[i].append(c_input)

                A[i] = np.clip(A[i], self._action_min, self._action_max)

            start = time.time()
            batch_pred_delta_ns, batch_disagreement = self.forward_multiple(A, S, ensemble=False)
            for i in range(len(ctrls)):
                ## Compute mean prediction from model samples
                next_step_pred = batch_pred_delta_ns[i]
                S[i,:] += next_step_pred.copy()
                if self.clip_state:
                    S[i,:] = np.clip(S[i,:], self._state_min, self._state_max)
                traj_list[i].append(S[i,:].copy())
                disagreements_list[i].append(batch_disagreement[i])
                actions_list[i].append(A[i,:])
        bd_list = []
        fit_list = []

        obs_trajs = np.array(traj_list)
        act_trajs = np.array(actions_list)
        disagr_trajs = np.array(disagreements_list)

        for i in range(len(ctrls)):
            obs_traj = obs_trajs[i]
            act_traj = act_trajs[i]
            disagr_traj = disagr_trajs[i]

            desc = self.compute_bd(obs_traj)
            fitness = self.compute_fitness(obs_traj, act_traj,
                                           disagr_traj=disagr_traj)

            fit_list.append(fitness)
            bd_list.append(desc)
        if not self.log_ind_trajs:
            obs_trajs = [None]*len(ctrls)
            act_trajs = [None]*len(ctrls)
            disagr_trajs = [None]*len(ctrls)
            
        return fit_list, bd_list, obs_trajs, act_trajs, disagr_trajs

    def evaluate_solution_model_det_ensemble_all(self, ctrls, mean=False, render=False):
        """
        Input: ctrl (array of floats) the genome of the individual
        Output: Trajectory and actions taken
        """
        controller_list = []
        traj_list = []
        actions_list = []
        disagreements_list = []
        obs_list = []
        c_input_traj_list = []
        
        env = copy.copy(self._env) ## need to verify this works
        obs = self._init_obs
        ens_size = self._ens_size

        cpt = 0
        for ctrl in ctrls:
            ## Create a copy of the controller
            controller_list.append(self.controller.copy())
            ## Set controller parameters
            controller_list[-1].set_parameters(ctrl)
            traj_list.append([])
            if mean:
                traj_list[cpt].append(obs.copy())
            else:
                traj_list[cpt].append(np.tile(obs, (ens_size, 1)))
            cpt += 1
            c_input_traj_list.append([])
            actions_list.append([])
            disagreements_list.append([])
            obs_list.append(obs.copy())
        
        if mean:
            S = np.tile(obs, (len(ctrls), 1))
            A = np.empty((len(ctrls), self.controller.output_dim))
        else:
            S = np.tile(obs, (ens_size*len(ctrls), 1))
            A = np.empty((ens_size*len(ctrls),
                          self.controller.output_dim))

        for t in tqdm.tqdm(range(self._model_max_h), total=self._model_max_h):
            for i in range(len(ctrls)):
                if mean:
                    c_input = None
                    if self.time_open_loop:
                        if self._norm_c_input:
                            c_input = [(t/self._env_max_h)*(1+1) - 1]
                        else:
                            c_input = [t]
                    else:
                        if self._norm_c_input:
                            c_input = self.normalize_inputs_s_minmax(S[i])
                        else:
                            c_input = S[i]
                        if self.use_obs_model:
                            c_input = self.observation_model.output_pred(ptu.from_numpy(c_input))
                            if self.clip_obs:
                                c_input = np.clip(c_input, self._obs_min, self._obs_max)
                            c_input = self.normalize_inputs_o_minmax(c_input)
                            
                    if self.pred_mode == 'single':
                        A[i] = controller_list[i](c_input)
                    elif self.pred_mode == 'all':
                        A[i] = controller_list[i](c_input_traj_list[i])
                    elif self.pred_mode == 'window':
                        A[i] = controller_list[i](c_input_traj_list[i][-10:])

                    c_input_traj_list[i].append(c_input)

                else:
                    c_input = None
                    if self.time_open_loop:
                        if self._norm_c_input:
                            c_input = [(t/self._env_max_h)*(1+1) - 1]
                        else:
                            c_input = [t]
                        c_input = np.reshape(np.array([c_input]*ens_size), (-1,1))
                    else:
                        if self._norm_c_input:
                            c_input = self.normalize_inputs_s_minmax(
                                S[i*ens_size:i*ens_size+ens_size])
                        else:
                            c_input = S[i*ens_size:i*ens_size+ens_size]
                        if self.use_obs_model:
                            c_input = self.observation_model.output_pred(ptu.from_numpy(c_input))
                            if self.clip_obs:
                                c_input = np.clip(c_input, self._obs_min, self._obs_max)
                            c_input = self.normalize_inputs_o_minmax(c_input)

                    if self.pred_mode == 'single':
                        A[i*ens_size:i*ens_size+ens_size] = \
                            controller_list[i](c_input)
                    elif self.pred_mode == 'all':
                        A[i*ens_size:i*ens_size+ens_size] = \
                            controller_list[i](c_input_traj_list[i])
                    elif self.pred_mode == 'window':
                        A[i*ens_size:i*ens_size+ens_size] = \
                            controller_list[i](c_input_traj_list[i][-10:])

                    c_input_traj_list[i].append(c_input)

                A[i*ens_size:i*ens_size+ens_size] = np.clip(
                    A[i*ens_size:i*ens_size+ens_size],
                    self._action_min,
                    self._action_max)

            start = time.time()
            if mean:
                batch_pred_delta_ns, batch_disagreement = self.forward_multiple(
                        A,
                        S,
                        mean=mean,
                        ensemble=False)
            else:
                batch_pred_delta_ns, batch_disagreement = self.forward_multiple(
                        A,
                        S,
                        mean=mean,
                        ensemble=False,
                        det_ens=True)

            for i in range(len(ctrls)):
                if mean:
                    ## Compute mean prediction from model samples
                    next_step_pred = batch_pred_delta_ns[i]
                    S[i,:] += next_step_pred.copy()
                    if self.clip_state:
                        S[i,:] = np.clip(S[i,:],
                                         self._state_min, self._state_max)
                    traj_list[i].append(S[i,:].copy())
                    disagreements_list[i].append(batch_disagreement[i])
                    actions_list[i].append(A[i,:])
                else:

                    S[i*ens_size:i*ens_size+ens_size] += batch_pred_delta_ns[:,i]
                    if self.clip_state:
                        S[i*ens_size:i*ens_size+ens_size] = np.clip(
                            S[i*ens_size:i*ens_size+ens_size],
                            self._state_min, self._state_max)
                    traj_list[i].append(S[i*ens_size:i*ens_size+ens_size].copy())
                    actions_list[i].append(A[i*ens_size:i*ens_size+ens_size])
        
        bd_list = []
        fit_list = []

        obs_trajs = np.array(traj_list)
        act_trajs = np.array(actions_list)
        disagr_trajs = np.array(disagreements_list)

        for i in range(len(ctrls)):
            obs_traj = obs_trajs[i]
            act_traj = act_trajs[i]
            disagr_traj = disagr_trajs[i]

            desc = self.compute_bd(obs_traj, ensemble=True, mean=mean)
            fitness = self.compute_fitness(obs_traj, act_traj,
                                           disagr_traj=disagr_traj)

            fit_list.append(fitness)
            bd_list.append(desc)

        if not self.log_ind_trajs:
            obs_trajs = [None]*len(ctrls)
            act_trajs = [None]*len(ctrls)
            disagr_trajs = [None]*len(ctrls)

        return fit_list, bd_list, obs_trajs, act_trajs, disagr_trajs

    def evaluate_solution_model_ensemble_all(self, ctrls, mean=True, disagr=True,
                                                 render=False, use_particules=True):
        """
        Input: ctrl (array of floats) the genome of the individual
        Output: Trajectory and actions taken
        """
        controller_list = []
        traj_list = []
        actions_list = []
        disagreements_list = []
        obs_list = []

        env = copy.copy(self._env) ## need to verify this works
        obs = self._init_obs

        for ctrl in ctrls:
            ## Create a copy of the controller
            controller_list.append(self.controller.copy())
            ## Set controller parameters
            controller_list[-1].set_parameters(ctrl)
            traj_list.append([])
            actions_list.append([])
            disagreements_list.append([])
            obs_list.append(obs.copy())

        ens_size = self.dynamics_model.ensemble_size

        if use_particules:
            S = np.tile(obs, (ens_size*len(ctrls), 1))
            A = np.empty((ens_size*len(ctrls),
                          self.controller.output_dim))
        else:
            S = np.tile(obs, (len(ctrls), 1))
            A = np.empty((len(ctrls), self.controller.output_dim))

        for t in tqdm.tqdm(range(self._model_max_h), total=self._model_max_h):
            for i in range(len(ctrls)):
                # A[i, :] = controller_list[i](S[i,:])
                if use_particules:
                    if self.time_open_loop:
                        if self._norm_c_input:
                            norm_t = (t/self._env_max_h)*(1+1) - 1
                            A[i*ens_size:i*ens_size+ens_size] = \
                            controller_list[i]([norm_t]*ens_size)
                        else:
                            A[i*ens_size:i*ens_size+ens_size] = \
                            controller_list[i]([t]*ens_size)
                    else:
                        if self._norm_c_input:
                            norm_s = self.normalize_inputs_s_minmax(
                                    S[i*ens_size:i*ens_size+ens_size])
                            A[i*ens_size:i*ens_size+ens_size] = \
                            controller_list[i](norm_s)
                        else:
                            A[i*ens_size:i*ens_size+ens_size] = \
                            controller_list[i](S[i*ens_size:i*ens_size+ens_size])
                else:
                    if self.time_open_loop:
                        if self._norm_c_input:
                            norm_t = (t/self._env_max_h)*(1+1) - 1
                            A[i] = controller_list[i]([norm_t])
                        else:
                            A[i] = controller_list[i]([t])
                    else:
                        if self._norm_c_input:
                            norm_s = self.normalize_inputs_s_minmax(S[i])
                            A[i] = controller_list[i](norm_s)
                        else:
                            A[i] = controller_list[i](S[i])
                    A[i] = np.clip(A[i], self._action_min, self._action_max)
                            
            start = time.time()
            if use_particules:
                batch_pred_delta_ns, batch_disagreement = self.forward(A, S, mean=mean,
                                                                       disagr=disagr,
                                                                       multiple=True)
            else:
                batch_pred_delta_ns, batch_disagreement = self.forward_multiple(A, S,
                                                                                mean=True,
                                                                                disagr=True)
            # print(f"Time for inference {time.time()-start}")
            for i in range(len(ctrls)):
                if use_particules:
                    ## Don't use mean predictions and keep each particule trajectory
                    # Be careful, in that case there is no need to repeat each state in
                    # forward multiple function
                    disagreement = self.compute_abs_disagreement(S[i*ens_size:i*ens_size+ens_size]
                                                                 , batch_pred_delta_ns[i])
                    # print("Disagreement: ", disagreement.shape)
                    # print("Disagreement: ", disagreement)
                    disagreement = ptu.get_numpy(disagreement)
                    
                    disagreements_list[i].append(disagreement.copy())
                    
                    S[i*ens_size:i*ens_size+ens_size] += batch_pred_delta_ns[i]
                    if self.clip_state:
                        S[i*ens_size:i*ens_size+ens_size] = np.clip(
                            S[i*ens_size:i*ens_size+ens_size],
                            self._state_min, self._state_max)
                    traj_list[i].append(S[i*ens_size:i*ens_size+ens_size].copy())

                    # disagreements_list[i].append(batch_disagreement[i])
                    actions_list[i].append(A[i*ens_size:i*ens_size+ens_size])

                else:
                    ## Compute mean prediction from model samples
                    next_step_pred = batch_pred_delta_ns[i]
                    mean_pred = [np.mean(next_step_pred[:,i]) for i
                                 in range(len(next_step_pred[0]))]
                    S[i,:] += mean_pred.copy()
                    if self.clip_state:
                        S[i,:] = np.clip(S[i,:], self._state_min, self._state_max)
                    traj_list[i].append(S[i,:].copy())
                    disagreements_list[i].append(batch_disagreement[i])
                    actions_list[i].append(A[i,:])

        bd_list = []
        fit_list = []

        obs_trajs = np.array(traj_list)
        act_trajs = np.array(actions_list)
        disagr_trajs = np.array(disagreements_list)

        for i in range(len(ctrls)):
            obs_traj = obs_trajs[i]
            act_traj = act_trajs[i]
            disagr_traj = disagr_trajs[i]

            desc = self.compute_bd(obs_traj, ensemble=True, mean=not use_particules)
            fitness = self.compute_fitness(obs_traj, act_traj,
                                           disagr_traj=disagr_traj,
                                           ensemble=True)

            fit_list.append(fitness)
            bd_list.append(desc)

        if not self.log_ind_trajs:
            obs_trajs = [None]*len(ctrls)
            act_trajs = [None]*len(ctrls)
            disagr_trajs = [None]*len(ctrls)

        return fit_list, bd_list, obs_trajs, act_trajs, disagr_trajs

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
                return pred_delta_ns_list, disagreement_list
        else:
            pred_delta_ns = self.dynamics_model.output_pred_ts_ensemble(s_0, a_0, mean=mean)
        return pred_delta_ns, 0
    
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
        bd = [0]*self._dim_map

        try:
            wp_idxs = [i for i in range(len(obs_traj)//self.n_wps, len(obs_traj),
                                        len(obs_traj)//self.n_wps)][:self.n_wps-1]
        except:
            import pdb; pdb.set_trace()
        wp_idxs += [-1]

        obs_wps = np.take(obs_traj, wp_idxs, axis=0)
        if ensemble:
            if mean:
                obs_wps = np.mean(obs_wps_obs, axis=0)
            else:
                ## Return bd for each model and flatten it all
                if self._env_name == 'ball_in_cup':
                    bd = obs_wps[:,:,:3].flatten()
                if 'maze' in self._env_name:
                    bd = obs_wps[:,:,:2].flatten()
                if 'redundant_arm' in self._env_name:
                    bd = obs_wps[:,:,-2:].flatten()
                if self._env_name == 'half_cheetah':
                    bd = obs_wps[:,:,:1].flatten()
                if self._env_name == 'walker2d':
                    bd = obs_wps[:,:,:1].flatten()
                return bd

        if self._env_name == 'ball_in_cup':
            bd = obs_wps[:,:3].flatten()
        if 'maze' in self._env_name:
            bd = obs_wps[:,:2].flatten()
        if 'redundant_arm' in self._env_name:
            bd = obs_wps[:,-2:].flatten()
        if self._env_name == 'half_cheetah':
            bd = obs_wps[:,:1].flatten()
        if self._env_name == 'walker2d':
            bd = obs_wps[:,:1].flatten()
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
        if self._env_name == 'empty_maze':
            fit = fit_func(act_traj, disagr_traj)
        if self._env_name == 'fastsim_maze_traps':
            fit = fit_func(act_traj, disagr_traj)
        if 'redundant_arm' in self._env_name:
            fit = fit_func(act_traj, disagr_traj)
        if self._env_name == 'half_cheetah':
            fit = fit_func(act_traj, disagr_traj)
        if self._env_name == 'walker2d':
            fit = fit_func(act_traj, disagr_traj)
        return fit

    def normalize_inputs_s_minmax(self, data):
        data_norm = (data - self._state_min)/(self._state_max - self._state_min)
        rescaled_data_norm = data_norm * (1 + 1) - 1 ## Rescale between -1 and 1
        return rescaled_data_norm

    def normalize_inputs_o_minmax(self, data):
        data_norm = (data - self._obs_min)/(self._obs_max - self._obs_min)
        rescaled_data_norm = data_norm * (1 + 1) - 1 ## Rescale between -1 and 1
        return rescaled_data_norm

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

        
        "env_name": args.environment,
        ## for dump
        "ensemble_dump": False,
    }

    
    #########################################################################
    ####################### Preparation of run ##############################
    #########################################################################
    
    ### Environment initialization ###
    env_register_id = 'BallInCup3d-v0'
    gym_args = {}
    is_local_env = False
    init_obs = None
    if args.environment == 'ball_in_cup':
        import mb_ge ## Contains ball in cup
        env_register_id = 'BallInCup3d-v0'
        a_min = np.array([-1, -1, -1])
        a_max = np.array([1, 1, 1])
        ss_min = -0.4
        ss_max = 0.4
        dim_map = 3
    elif args.environment == 'redundant_arm':
        import redundant_arm ## contains classic redundant arm
        gym_args['dof'] = 20
        env_register_id = 'RedundantArmPos-v0'
        a_min = np.array([-1]*gym_args['dof'])
        a_max = np.array([1]*gym_args['dof'])
        ss_min = np.array([-0.5]*gym_args['dof']+[0,0])
        ss_max = np.array([0.8]*gym_args['dof']+[1,1])
        # ss_min = np.array([-2.91586418e-01, -2.91059290e-01, -4.05994661e-01,
        #                    -3.43161155e-01, -4.48797687e-01, -3.42430607e-01,
        #                    -4.64587165e-01, -4.57486040e-01, -4.40965296e-01,
        #                    -3.74359165e-01, -4.73628034e-01, -3.64009843e-01,
        #                    -4.78609985e-01, -4.22113313e-01, -5.27555361e-01,
        #                    -5.18617559e-01, -4.36935815e-01, -5.31945509e-01,
        #                    -4.44923835e-01, -5.36581457e-01, 2.33058244e-05,
        #                    7.98103927e-05])
        # ss_max = np.array([0.8002732,  0.74879046, 0.68724849, 0.76289724,
        #                    0.66943127, 0.77772601, 0.67210694, 0.56392794,
        #                    0.65394265, 0.74616584, 0.61193007, 0.73037668,
        #                    0.59987872, 0.71458412, 0.58088037, 0.60106068,
        #                    0.66026566, 0.58433874, 0.64901992, 0.44800244,
        #                    0.99999368, 0.99999659])
        dim_map = 2
    elif args.environment == 'redundant_arm_no_walls':
        env_register_id = 'RedundantArmPosNoWalls-v0'
        a_min = np.array([-1]*20)
        a_max = np.array([1]*20)
        ss_min = -1
        ss_max = 1
        dim_map = 2
    elif args.environment == 'redundant_arm_no_walls_no_collision':
        env_register_id = 'RedundantArmPosNoWallsNoCollision-v0'
        a_min = np.array([-1]*20)
        a_max = np.array([1]*20)
        ss_min = -1
        ss_max = 1
        dim_map = 2
    elif args.environment == 'redundant_arm_no_walls_limited_angles':
        env_register_id = 'RedundantArmPosNoWallsLimitedAngles-v0'
        a_min = np.array([-1]*100)
        a_max = np.array([1]*100)
        ss_min = -1
        ss_max = 1
        dim_map = 2
        gym_args['dof'] = 100
    elif args.environment == 'fastsim_maze_laser':
        env_register_id = 'FastsimSimpleNavigation-v0'
        a_min = np.array([-1, -1])
        a_max = np.array([1, 1])
        ss_min = np.array([0, 0, -1, -1, -1, -1])
        ss_max = np.array([600, 600, 1, 1, 1, 1])
        init_obs = np.array([60., 450., 0., 0., 0. , 0.])
        state_dim = 6
        obs_min = np.array([0, 0, 0, 0, 0])
        obs_max = np.array([100, 100, 100, 1, 1])
        dim_map = 2
        bd_inds = [0, 1]
        nb_div = 50
        args.use_obs_model = True
    elif args.environment == 'empty_maze_laser':
        env_register_id = 'FastsimEmptyMapNavigation-v0'
        a_min = np.array([-1, -1])
        a_max = np.array([1, 1])
        ss_min = np.array([0, 0, -1, -1, -1, -1])
        ss_max = np.array([600, 600, 1, 1, 1, 1])
        init_obs = np.array([300., 300., 0., 0., 0. , 0.])
        state_dim = 6
        obs_min = np.array([0, 0, 0, 0, 0])
        obs_max = np.array([100, 100, 100, 1, 1])
        dim_map = 2
        bd_inds = [0, 1]
        nb_div = 50
        args.use_obs_model = True
    elif args.environment == 'fastsim_maze':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        a_min = np.array([-1, -1])
        a_max = np.array([1, 1])
        obs_min = ss_min = np.array([0, 0, -1, -1, -1, -1])
        obs_max = ss_max = np.array([600, 600, 1, 1, 1, 1])
        state_dim = 6
        # init_obs = np.array([60., 450., 0., 0., 0. , 0.])
        dim_map = 2
        bd_inds = [0, 1]
    elif args.environment == 'empty_maze':
        env_register_id = 'FastsimEmptyMapNavigationPos-v0'
        a_min = np.array([-1, -1])
        a_max = np.array([1, 1])
        obs_min = ss_min = np.array([0, 0, -1, -1, -1, -1])
        obs_max = ss_max = np.array([600, 600, 1, 1, 1, 1])
        state_dim = 6
        # init_obs = np.array([300., 300., 0., 0., 0. , 0.])
        dim_map = 2
        bd_inds = [0, 1]
        nb_div = 50
    elif args.environment == 'fastsim_maze_traps':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        a_min = np.array([-1, -1])
        a_max = np.array([1, 1])
        ss_min = np.array([0, 0, -1, -1, -1, -1])
        ss_max = np.array([600, 600, 1, 1, 1, 1])
        state_dim = 6
        dim_map = 2
        gym_args['physical_traps'] = True
        bd_inds = [0, 1]
        nb_div = 50
    elif args.environment == 'half_cheetah':
        env_register_id = 'HalfCheetah-v3'
        a_min = np.array([-1, -1, -1, -1, -1, -1])
        a_max = np.array([1, 1, 1, 1, 1, 1])
        ## Got these with NS 100 000 eval budget
        ss_min = np.array([-49.02189923, -0.61095456, -16.64607454, -0.70108701,
                           -1.00943152, -0.65815842, -1.19701832, -1.28944137,
                           -0.76604915, -5.2375874, -5.51574707, -10.4422284,
                           -26.43682609, -31.22491269, -31.96452725,
                           -26.68346276, -32.95576583, -32.70174356])
        ss_max = np.array([32.47642872, 0.83392967, 38.93965081, 1.14752425,
                           0.93195033, 0.95062493, 0.88961483, 1.11808423,
                           0.76134696, 4.81465142, 4.9208565, 10.81297147,
                           25.82911106, 28.41785798, 24.95866255, 31.30177305,
                           34.88956652, 30.07857634])

        init_obs = np.array([0.]*18)
        dim_map = 1
        gym_args['exclude_current_positions_from_observation'] = False
        gym_args['reset_noise_scale'] = 0
    elif args.environment == 'walker2d':
        env_register_id = 'Walker2d-v3'
        a_min = np.array([-1, -1, -1, -1, -1, -1])
        a_max = np.array([1, 1, 1, 1, 1, 1])
        ## Got these with NS 100 000 eval budget
        ss_min = np.array([-4.26249395, 0.75083099, -1.40787207, -2.81284653,
                           -2.93150238, -1.5855295, -3.04205169, -2.91603065,
                           -1.62175821, -7.11379591, -10., -10., -10., -10.,
                           -10., -10., -10., -10.])
        ss_max = np.array([1.66323372, 1.92256493, 1.15429141, 0.43140988,
                           0.49341738, 1.50477799, 0.47811355, 0.63702984,
                           1.50380045, 4.98763458, 4.00820283, 10., 10., 10.,
                           10., 10., 10., 10.])

        ## Got these with NS 100 000 eval budget
        ss_min = np.array([-5.62244541, 0.7439814, -1.41163676, -3.1294922,
                           -2.97025984, -1.67482138, -3.1644274, -3.01373681,
                           -1.78557467, -8.55243269, -10., -10., -10., -10.,
                           -10., -10., -10., -10.])
        ss_max = np.array([1.45419434, 1.98069464, 1.1196152, 0.5480219,
                           0.65664259, 1.54582436, 0.53905455, 0.61275703,
                           1.5541609, 6.12093722, 5.9363082, 10., 10., 10.,
                           10., 10., 10., 10.])
        init_obs = np.array([0.]*18)
        dim_map = 1
        gym_args['exclude_current_positions_from_observation'] = False
        gym_args['reset_noise_scale'] = 0
    elif args.environment == 'hexapod_omni':
        from src.envs.hexapod_dart.hexapod_env import HexapodEnv ## Contains hexapod 
        is_local_env = True
        max_step = 300 # ctrl_freq = 100Hz, sim_time = 3.0 seconds 
        obs_dim = 48
        act_dim = 18
        dim_x = 36
        ## Need to check the dims for hexapod
        ss_min = -1
        ss_max = 1
        dim_map = 2
    else:
        raise ValueError(f"{args.environment} is not a defined environment")

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
        'ensemble_size': 1 if not 'ens' in args.model_type else args.ens_size,
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
            'ensemble_size': 1 if not 'ens' in args.model_type else args.ens_size,
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
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,

        'fitness_func': args.fitness_func,
        'n_waypoints': n_waypoints,
        'num_cores': args.num_cores,
        'dim_map': dim_map,
        
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
        
    if ns_data_ok and 'ens' in args.model_type:
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

    ## Evaluate on real sys
    if args.perfect_model and px['model_variant'] == 'all_dynamics':
        s_list = evaluate_all_(to_evaluate)
    else:
        s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, px)

    pool.close()

    # if px['type'] == "fixed":
        # px['type'] = "unstructured"

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
        for m_idx in range(px['ensemble_size']):
            loc_model_bd_traj_data= []
            for ind in model_archive:
                if not 'ens' in args.model_type or args.perfect_model:
                    bd_traj = ind.obs_traj[:,:2]
                else:
                    bd_traj = ind.obs_traj[:,m_idx,:2]
                loc_model_bd_traj_data.append(bd_traj)

            all_bd_traj_data.append((loc_model_bd_traj_data, f'model n°{m_idx+1}'))

        ## Plot real archive and model(s) archive on plot
        total_plots = len(all_bd_traj_data)
        ## make it as square as possible
        rows = cols = round(np.sqrt(total_plots))
        ## Add a row in case closest square cannot take all plots in
        if total_plots > rows*cols:
            rows += 1
        
        fig1, axs1 = plt.subplots(rows, cols)
        fig2, axs2 = plt.subplots(rows, cols)
        m_cpt = 0
        
        # ind_idxs = np.random.permutation(len(model_archive))[:100]

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
                    loc_bd_traj_data = np.array(loc_bd_traj_data)
                    
                    ## Plot BDs
                    ax1.scatter(x=loc_bd_traj_data[:,-1,0],y=loc_bd_traj_data[:,-1,1], s=3)
                    ax1.scatter(x=init_obs[0],y=init_obs[1], s=10, c='red')
                    ## Plot trajectories
                    # for ind_idx in range(len(model_archive)):
                    for bd_traj in loc_bd_traj_data:
                        ax2.plot(bd_traj[:,0], bd_traj[:,1], alpha=0.1, markersize=1)

                    ax1.set_xlabel('x-axis')
                    ax1.set_ylabel('y-axis')
                    ax1.set_title(f'Archive coverage on {loc_system_name}')
                    if 'real' in loc_system_name:
                        ax1.set_xlim(0, 600)
                        ax1.set_ylim(0, 600)
                    else:
                        loc_bd_mins = np.min(loc_bd_traj_data[:,-1,:], axis=0) 
                        loc_bd_maxs = np.max(loc_bd_traj_data[:,-1,:], axis=0) 
                        ax1.set_xlim(loc_bd_mins[0], loc_bd_maxs[0])
                        ax1.set_ylim(loc_bd_mins[1], loc_bd_maxs[1])
                    ax1.set_aspect('equal', adjustable='box')
                    
                    ax2.set_xlabel('x-axis')
                    ax2.set_ylabel('y-axis')
                    ax2.set_title(f'Individuals trajectories on {loc_system_name}')
                    if 'real' in loc_system_name:
                        ax2.set_xlim(0, 600)
                        ax2.set_ylim(0, 600)
                    else:
                        loc_bd_mins = np.min(loc_bd_traj_data, axis=(0,1)) 
                        loc_bd_maxs = np.max(loc_bd_traj_data, axis=(0,1))
                        ax2.set_xlim(loc_bd_mins[0], loc_bd_maxs[0])
                        ax2.set_ylim(loc_bd_mins[1], loc_bd_maxs[1])
                    ax2.set_aspect('equal', adjustable='box')

                    m_cpt += 1
                except:
                    fig1.delaxes(axs1[row][col])
                    fig2.delaxes(axs2[row][col])


        fig1.set_size_inches(total_plots*2,total_plots*2)
        fig1.suptitle('Archive coverage after DAB')
        file_path = os.path.join(args.log_dir, f"{args.environment}_model_to_real_cov")
        fig1.savefig(file_path,
                    dpi=300, bbox_inches='tight')

        fig2.set_size_inches(total_plots*2,total_plots*2)
        fig2.suptitle('Individuals trajectories after DAB')
        file_path = os.path.join(args.log_dir, f"{args.environment}_model_ind_trajs")
        fig2.savefig(file_path,
                    dpi=300, bbox_inches='tight')


    ## Exit if we already ran on real model or if it was a RP run
    if args.perfect_model or args.random_policies:
        exit()

    ## otherwise...
    
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

    ## Plot archive coverage evolution at each generation
    # we can do that by looking at coverage of individuals in archive taking
    # into account the first lambda, then adding to these the next lambda inds
    # etc...
    def get_data_bins(data, ss_min, ss_max, dim_map, bd_inds, nb_div):
        df_min = data.iloc[0].copy(); df_max = data.iloc[0].copy()

        for i in range(dim_map):
            df_min[f'bd{i}'] = ss_min[bd_inds[i%len(bd_inds)]]
            df_max[f'bd{i}'] = ss_max[bd_inds[i%len(bd_inds)]]

        ## Deprecated but oh well
        data = data.append(df_min, ignore_index = True)
        data = data.append(df_max, ignore_index = True)

        for i in range(dim_map):
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
        total_bins = nb_div**dim_map
        ## return coverage (number of bins filled)
        return len(counts[counts>=1])/total_bins

    archive_cov_by_gen = []
    bd_cols = [f'bd{i}' for i in range(dim_map)]
    bd_cols = bd_cols[-2:]
    for gen in range(1,len(real_archive)//px['lambda']):
        archive_at_gen = real_archive[:gen*px['lambda']]
        bds_at_gen = np.array([ind.desc[-2:] for ind in archive_at_gen])
        archive_at_gen_data = pd.DataFrame(bds_at_gen, columns=bd_cols)
        cov_at_gen = compute_cov(archive_at_gen_data, ss_min, ss_max, px['dim_map'], bd_inds, nb_div)
        archive_cov_by_gen.append(cov_at_gen)
    archive_cov_by_gen = np.array(archive_cov_by_gen)
    to_save = os.path.join(args.log_dir, 'archive_cov_by_gen')
    np.savez(to_save, archive_cov_by_gen=archive_cov_by_gen)
    
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

    #-----------------Type of algo---------------------#
    # options are 'qd', 'ns'
    parser.add_argument("--algo", type=str, default="qd")
    #-----------------Type of QD---------------------#
    # options are 'cvt', 'grid', 'unstructured', 'fixed'
    parser.add_argument("--qd_type", type=str, default="fixed")
    
    #---------------CPU usage-------------------#
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num_cores", type=int, default=6)
    
    #-----------Store results + analysis-----------#
    parser.add_argument("--log_dir", type=str)
    parser.add_argument('--log-ind-trajs', action="store_true") ## Store trajs during run
    parser.add_argument('--dump-ind-trajs', action="store_true") ## Dump traj in archive
    
    #-----------QD params for cvt or GRID---------------#
    # ONLY NEEDED FOR CVT OR GRID MAP ELITES - not needed for unstructured archive
    parser.add_argument("--grid_shape", default=[100,100], type=list) # num discretizat
    parser.add_argument("--n_niches", default=3000, type=int) # qd number niches

    #----------population params--------#
    parser.add_argument("--random-init-batch", default=100, type=int) # Number of inds to initialize the archive
    parser.add_argument("--b_size", default=200, type=int) # For paralellization - 
    parser.add_argument("--dump_period", default=5000, type=int) 
    parser.add_argument("--dump-mode", type=str, default="budget")
    parser.add_argument("--max_evals", default=1e6, type=int) # max number of evaluation
    parser.add_argument("--multi-eval", default=1., type=float) # max number of evaluation
    parser.add_argument("--selector", default="uniform", type=str)
    # possible values: iso_dd, polynomial or sbx
    parser.add_argument("--mutation", default="iso_dd", type=str)

    #-------------Algo params-----------#
    parser.add_argument('--pop-size', default=100, type=int) # 1 takes BD on last obs
    parser.add_argument('--bootstrap-archive-path', type=str, default='')
    parser.add_argument('--bootstrap-selection', type=str, default='final_pop') # final_pop, nov, random
    parser.add_argument('--fitness-func', type=str, default='energy_minimization')
    parser.add_argument('--n-waypoints', default=1, type=int) # 1 takes BD on last obs
    ## Gen max_evals random policies and evaluate them
    parser.add_argument('--random-policies', action="store_true") 
    parser.add_argument('--environment', '-e', type=str, default='empty_maze')
    parser.add_argument('--lambda-add', type=int, default=15)
    parser.add_argument('--arch-sel', type=str, default='random')
    parser.add_argument('--rep', type=int, default='1')

    #-----------Controller params--------#
    parser.add_argument('--c-type', type=str, default='ffnn') # Type of controller to use
    parser.add_argument('--norm-controller-input', type=int, default=1) # minmax Normalize input space
    parser.add_argument('--open-loop-control', type=int, default=0) # open loop (time) or closed loop (state) control
    parser.add_argument('--c-n-layers', type=int, default=2) # Number of hidden layers
    parser.add_argument('--c-n-neurons', type=int, default=10) # Number of neurons per hidden layer
    ## RNN inputs: (batch,seq_len,input_dim)
    parser.add_argument('--pred-mode', type=str, default='single') # RNN prediction mode (single; all; window)
    
    #-------------Model params-----------#
    parser.add_argument('--obs-model-type', type=str, default='nn') # nn, srf
    
    parser.add_argument('--model-variant', type=str, default='dynamics') # dynamics, surrogate
    parser.add_argument('--model-type', type=str, default='det') # prob, det, det_ens
    parser.add_argument('--ens-size', type=int, default='4') # when using ens
    parser.add_argument('--model-horizon', type=int, default=-1) # model eval horizon
    parser.add_argument('--perfect-model', action='store_true')
    parser.add_argument('--norm-bd', type=int, default=1) # minmax Normalize BD space
    parser.add_argument('--nov-ens', type=str, default='sum') # min, mean, sum
    parser.add_argument('--use-obs-model', action='store_true')
    ## '' does not pretrain, srf pretrains using data generated with
    ## spatial random fields
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--pretrain-budget', type=int, default=10000)
    parser.add_argument('--clip-obs', action='store_true')
    parser.add_argument('--clip-state', action='store_true')
    
    args = parser.parse_args()

    main(args)
