import os, sys
import argparse
import numpy as np

import src.torch.pytorch_util as ptu

from src.map_elites import common as cm
from src.map_elites import unstructured_container, cvt
from src.map_elites.qd import QD
from src.map_elites.ns import NS

from src.models.dynamics_models.deterministic_model import DeterministicDynModel
from src.models.dynamics_models.deterministic_ensemble import DeterministicEnsemble
from src.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
from src.models.surrogate_models.det_surrogate import DeterministicQDSurrogate

#----------controller imports--------#
from model_init_study.controller.nn_controller \
    import NeuralNetworkController

#----------Environment imports--------#
import gym
import diversity_algorithms.environments.env_imports ## Contains deterministic ant + fetch

#----------Utils imports--------#
import multiprocessing
from multiprocessing import cpu_count
import copy
import numpy as np
import torch
import time
import tqdm

from src.data_management.replay_buffers.simple_replay_buffer import SimpleReplayBuffer

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

################################################################################
############################## Model methods ###################################
################################################################################

def get_dynamics_model(params):
    dynamics_model_params = params['dynamics_model_params']
    obs_dim = dynamics_model_params['obs_dim']
    action_dim = dynamics_model_params['action_dim']
    dynamics_model_type = dynamics_model_params['model_type']

    ## INIT MODEL ##
    if dynamics_model_type == "prob":
        from src.trainers.mbrl.mbrl import MBRLTrainer
        variant = dict(
            mbrl_kwargs=dict(
                ensemble_size=dynamics_model_params['ensemble_size'],
                layer_size=dynamics_model_params['layer_size'],
                learning_rate=1e-3,
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
                use_minmax_norm=True)
        dynamics_model_trainer = MBRLTrainer(
                model=dynamics_model,
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
                use_minmax_norm=True)
        dynamics_model_trainer = None ## Not trainable for now
        
    return dynamics_model, dynamics_model_trainer

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

class WrappedEnv():
    def __init__(self, params):
        self._action_min = params['action_min']
        self._action_max = params['action_max']
        self._state_min = params['state_min']
        self._state_max = params['state_max']
        self._sa_min = np.concatenate((params['state_min'], params['action_min']))
        self._sa_max = np.concatenate((params['state_max'], params['action_max']))
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
        self._model_max_h = params['dynamics_model_params']['model_horizon']
        self._ens_size = params['dynamics_model_params']['ensemble_size']
        self.time_open_loop = params['time_open_loop']
        self._norm_c_input = params['controller_params']['norm_input']
        self.n_wps = params['n_waypoints']
        
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
                if self._norm_c_input:
                    norm_t = (t/self._env_max_h)*(1+1) - 1
                    action = controller([norm_t])
                else:
                    action = controller([t])
            else:
                if self._norm_c_input:
                    norm_obs = self.normalize_inputs_s_minmax(obs)
                    action = controller(norm_obs)
                else:
                    action = controller(obs)
            action = np.clip(action, self._action_min, self._action_max)
            # action = np.random.uniform(low=-1, high=1, size=(self._act_dim,))
            # action[action>self._action_max] = self._action_max
            # action[action<self._action_min] = self._action_min
            obs_traj.append(obs)
            act_traj.append(action)
            obs, reward, done, info = env.step(action)
            # print(np.array(obs_traj[-1]) - np.array(obs))
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
        # for t in range(self._env_max_h):
        for t in range(self._model_max_h):
            if self.time_open_loop:
                if self._norm_c_input:
                    norm_t = (t/self._env_max_h)*(1+1) - 1
                    action = controller([norm_t])
                else:
                    action = controller([t])
            else:
                if self._norm_c_input:
                    norm_obs = self.normalize_inputs_s_minmax(obs)
                    action = controller(norm_obs)
                else:
                    action = controller(obs)
            action = np.clip(action, self._action_min, self._action_max)
            # action[action>self._action_max] = self._action_max
            # action[action<self._action_min] = self._action_min
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
            
        disagr = 0
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
            actions_list.append([])
            disagreements_list.append([])
            obs_list.append(obs.copy())

        ## WARNING: need to get previous obs
        # S = np.tile(prev_element.trajectory[-1].copy(), (len(X)))
        S = np.tile(obs, (len(ctrls), 1))
        A = np.empty((len(ctrls), self.controller.output_dim))

        for t in tqdm.tqdm(range(self._model_max_h), total=self._model_max_h):
            for i in range(len(ctrls)):
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
            batch_pred_delta_ns, batch_disagreement = self.forward_multiple(A, S, ensemble=False)
            for i in range(len(ctrls)):
                ## Compute mean prediction from model samples
                next_step_pred = batch_pred_delta_ns[i]
                # import pdb; pdb.set_trace()
                S[i,:] += next_step_pred.copy()
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
                else:
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
                A[i] = np.clip(A[i], self._action_min, self._action_max)

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
                    traj_list[i].append(S[i,:].copy())
                    disagreements_list[i].append(batch_disagreement[i])
                    actions_list[i].append(A[i,:])
                else:
                    S[i*ens_size:i*ens_size+ens_size] += batch_pred_delta_ns[:,i]
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
                    traj_list[i].append(S[i*ens_size:i*ens_size+ens_size].copy())

                    # disagreements_list[i].append(batch_disagreement[i])
                    actions_list[i].append(A[i*ens_size:i*ens_size+ens_size])

                else:
                    ## Compute mean prediction from model samples
                    next_step_pred = batch_pred_delta_ns[i]
                    mean_pred = [np.mean(next_step_pred[:,i]) for i
                                 in range(len(next_step_pred[0]))]
                    S[i,:] += mean_pred.copy()
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
        bd = None

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
                if self._env_name == 'fastsim_maze':
                    bd = obs_wps[:,:,:2].flatten()
                if self._env_name == 'empty_maze':
                    bd = obs_wps[:,:,:2].flatten()
                if self._env_name == 'fastsim_maze_traps':
                    bd = obs_wps[:,:,:2].flatten()
                if self._env_name == 'redundant_arm_no_walls_limited_angles':
                    bd = obs_wps[:,:,-2:].flatten()
                if self._env_name == 'half_cheetah':
                    bd = obs_wps[:,:,:1].flatten()
                if self._env_name == 'walker2d':
                    bd = obs_wps[:,:,:1].flatten()
                return bd

        if self._env_name == 'ball_in_cup':
            bd = obs_wps[:,:3].flatten()
        if self._env_name == 'fastsim_maze':
            bd = obs_wps[:,:2].flatten()
        if self._env_name == 'empty_maze':
            bd = obs_wps[:,:2].flatten()
        if self._env_name == 'fastsim_maze_traps':
            bd = obs_wps[:,:2].flatten()
        if self._env_name == 'redundant_arm_no_walls_limited_angles':
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
        if self._env_name == 'redundant_arm_no_walls_limited_angles':
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

################################################################################
################################### MAIN #######################################
################################################################################
def main(args):

    px = \
    {
        # type of qd 'unstructured, grid, cvt'
        "type": args.qd_type,
        # arg for NS
        "pop_size": 100,
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
        # "nov_l": 1.5,
        "eps": 0.1, # usually 10%
        "k": 15,  # from novelty search
        "lambda": 15, # For fixed ind add during runs (Gomes 2015)

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
        import redundant_arm ## contains redundant arm
        env_register_id = 'RedundantArmPos-v0'
        a_min = np.array([-1]*20)
        a_max = np.array([1]*20)
        ss_min = -1
        ss_max = 1
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
    elif args.environment == 'fastsim_maze':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        # ss_min = -10
        # ss_max = 10
        a_min = np.array([-1, -1])
        a_max = np.array([1, 1])
        ss_min = np.array([0, 0, -1, -1, -1, -1])
        ss_max = np.array([600, 600, 1, 1, 1, 1])
        init_obs = np.array([300., 300., 0., 0., 0. , 0.])
        dim_map = 2
    elif args.environment == 'empty_maze':
        env_register_id = 'FastsimEmptyMapNavigationPos-v0'
        # ss_min = -10
        # ss_max = 10
        a_min = np.array([-1, -1])
        a_max = np.array([1, 1])
        ss_min = np.array([0, 0, -1, -1, -1, -1])
        ss_max = np.array([600, 600, 1, 1, 1, 1])
        init_obs = np.array([300., 300., 0., 0., 0. , 0.])
        dim_map = 2
    elif args.environment == 'fastsim_maze_traps':
        env_register_id = 'FastsimSimpleNavigationPos-v0'
        # ss_min = -10
        # ss_max = 10
        a_min = np.array([-1, -1])
        a_max = np.array([1, 1])
        ss_min = np.array([0, 0, -1, -1, -1, -1])
        ss_max = np.array([600, 600, 1, 1, 1, 1])
        dim_map = 2
        gym_args['physical_traps'] = True
    elif args.environment == 'half_cheetah':
        env_register_id = 'HalfCheetah-v3'
        a_min = np.array([-1, -1, -1, -1, -1, -1])
        a_max = np.array([1, 1, 1, 1, 1, 1])
        # ss_min = np.array([-10]*18)
        # ss_max = np.array([10]*18)
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
        # ss_min = np.array([-10]*18)
        # ss_max = np.array([10]*18)
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
    
    controller_params = \
    {
        'controller_input_dim': obs_dim,
        'controller_output_dim': act_dim,
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 10,
        'time_open_loop': False,
        'norm_input': False,
    }
    dynamics_model_params = \
    {
        'obs_dim': obs_dim,
        'action_dim': act_dim,
        'layer_size': 500,
        'batch_size': 512,
        'learning_rate': 1e-3,
        'train_unique_trans': False,
        'model_type': args.model_type,
        'model_horizon': args.model_horizon if args.model_horizon is not None else max_step,
        'ensemble_size': args.ens_size,
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

        # 'action_min': -1,
        # 'action_max': 1,
        'action_min': a_min,
        'action_max': a_max,

        'state_min': ss_min,
        'state_max': ss_max,
        
        # 'policy_param_init_min': -0.1,
        # 'policy_param_init_max': 0.1,
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
        
        'env': gym_env,
        'env_name': args.environment,
        'env_max_h': max_step,
        'fitness_func': args.fitness_func,
        'n_waypoints': n_waypoints,
    }
    ## Correct obs dim for controller if open looping on time
    if params['time_open_loop']:
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

    # dynamics_model, dynamics_model_trainer = get_dynamics_model(dynamics_model_params)
    dynamics_model, dynamics_model_trainer = get_dynamics_model(params)
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
    elif args.model_variant == "dynamics" :
        if args.model_type == "det":
            f_model = env.evaluate_solution_model 
        elif args.model_type == "prob" and args.environment == 'hexapod_omni':
            f_model = env.evaluate_solution_model_ensemble
    elif args.model_variant == "all_dynamics":
        if args.model_type == "det":
            f_model = env.evaluate_solution_model_all
        elif args.model_type == "det_ens":
            f_model = env.evaluate_solution_model_det_ensemble_all
        elif args.model_type == "prob":
            f_model = env.evaluate_solution_model_ensemble_all
    # elif args.model_variant == "direct":
        # f_model = env.evaluate_solution_model 

    if args.model_type == "det_ens":
        px["ensemble_dump"] = True
        px["ensemble_size"] = dynamics_model_params["ensemble_size"]

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
                                              max_evals=args.max_evals)
    else:
        to_evaluate = []
        for i in range(0, args.max_evals):
            x = np.random.uniform(low=px['min'], high=px['max'], size=dim_x)
            to_evaluate += [(x, f_real)]
            n_evals = len(to_evaluate)
            # fit, desc, obs, act, disagr = f_real(x, render=False)
            # import pdb; pdb.set_trace()
            # exit()
            
    ## Evaluate the found solutions on the real system
    
    ## If search was done on the real system already then no need to test the
    ## found behaviorss
    if args.perfect_model:
        exit()

    pool = multiprocessing.Pool(args.num_cores)

    ## Select the individuals to transfer onto real system
    ## Selection based on nov, clustering, other?
    
    ## Create to evaluate vector
    import itertools
    if not args.random_policies:
        to_evaluate = list(zip([ind.x.copy() for ind in model_archive], itertools.repeat(f_real)))

    if args.model_type == 'det_ens':
        px['parallel'] = False
        
    ## Evaluate on real sys
    s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, px)

    pool.close()

    if px['type'] == "fixed":
        px['type'] = "unstructured"

    px["ensemble_dump"] = False
    
    real_archive = []
            
    real_archive, add_list, _ = addition_condition(s_list, real_archive, px)

    cm.save_archive(real_archive, f"{n_evals}_real_added", px, args.log_dir)
    cm.save_archive(s_list, f"{n_evals}_real_all", px, args.log_dir)
    
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
    parser.add_argument("--num_cores", type=int, default=6)
    
    #-----------Store results + analysis-----------#
    parser.add_argument("--log_dir", type=str)
    
    #-----------QD params for cvt or GRID---------------#
    # ONLY NEEDED FOR CVT OR GRID MAP ELITES - not needed for unstructured archive
    parser.add_argument("--grid_shape", default=[100,100], type=list) # num discretizat
    parser.add_argument("--n_niches", default=3000, type=int)
    parser.add_argument("--n-waypoints", default=1, type=int) # 1 takes BD on last obs

    #----------population params--------#
    parser.add_argument("--random-init-batch", default=100, type=int) # Number of inds to initialize the archive
    parser.add_argument("--b_size", default=200, type=int) # For paralellization - 
    parser.add_argument("--dump_period", default=5000, type=int) 
    parser.add_argument("--dump-mode", type=str, default="budget")
    parser.add_argument("--max_evals", default=1e6, type=int) # max number of evaluation
    parser.add_argument("--selector", default="uniform", type=str)
    # possible values: iso_dd, polynomial or sbx
    parser.add_argument("--mutation", default="iso_dd", type=str)

    #-------------DAQD params-----------#
    parser.add_argument('--transfer-selection', type=str, default='all')
    parser.add_argument('--fitness-func', type=str, default='energy_minimization')
    parser.add_argument('--nb-transfer', type=int, default=1)

    parser.add_argument('--model-variant', type=str, default='dynamics') # dynamics, surrogate
    parser.add_argument('--model-type', type=str, default='det') # prob, det, det_ens
    parser.add_argument('--model-horizon', type=int, default=None) # whatever suits you
    parser.add_argument('--perfect-model', action='store_true')

    #----------model init study params--------#
    parser.add_argument('--environment', '-e', type=str, default='empty_maze')
    parser.add_argument('--rep', type=int, default='1')
    parser.add_argument('--random-policies', action="store_true") ## Gen max_evals random policies and evaluate them
    parser.add_argument('--ens-size', type=int, default='4')
    args = parser.parse_args()

    main(args)
