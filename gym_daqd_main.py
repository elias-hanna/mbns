#----------Algo imports--------#
from src.map_elites import common as cm
from src.map_elites import unstructured_container, cvt
from src.map_elites.mbqd import ModelBasedQD

from exps_utils import get_dynamics_model, get_surrogate_model, \
    get_observation_model, addition_condition, evaluate_, evaluate_all_, \
    process_args

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

# added in get dynamics model section
#from src.trainers.mbrl.mbrl_det import MBRLTrainer
#from src.trainers.mbrl.mbrl import MBRLTrainer
#from src.trainers.qd.surrogate import SurrogateTrainer

# def get_dynamics_model(dynamics_model_type, act_dim, obs_dim):
#     obs_dim = obs_dim
#     action_dim = act_dim
    
#     ## INIT MODEL ##
#     if dynamics_model_type == "prob":
#         from src.trainers.mbrl.mbrl import MBRLTrainer
#         variant = dict(
#             mbrl_kwargs=dict(
#                 ensemble_size=4,
#                 layer_size=500,
#                 learning_rate=1e-3,
#                 batch_size=512,
#             )
#         )
#         M = variant['mbrl_kwargs']['layer_size']
#         dynamics_model = ProbabilisticEnsemble(
#             ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
#             obs_dim=obs_dim,
#             action_dim=action_dim,
#             hidden_sizes=[M, M]
#         )
#         dynamics_model_trainer = MBRLTrainer(
#             ensemble=dynamics_model,
#             **variant['mbrl_kwargs'],
#         )

#         # ensemble somehow cant run in parallel evaluations
#     elif dynamics_model_type == "det":
#         from src.trainers.mbrl.mbrl_det import MBRLTrainer 
#         dynamics_model = DeterministicDynModel(obs_dim=obs_dim,
#                                                action_dim=action_dim,
#                                                hidden_size=500)
#         dynamics_model_trainer = MBRLTrainer(
#             model=dynamics_model,
#             batch_size=512,)


#     return dynamics_model, dynamics_model_trainer

# def get_surrogate_model(dim):
#     from src.trainers.qd.surrogate import SurrogateTrainer
#     dim_x=dim # genotype dimnesion    
#     model = DeterministicQDSurrogate(gen_dim=dim_x, bd_dim=2, hidden_size=64)
#     model_trainer = SurrogateTrainer(model, batch_size=32)

#     return model, model_trainer


# class WrappedEnv():
#     def __init__(self, params):
#         self._action_min = params['action_min']
#         self._action_max = params['action_max']
#         self._env_max_h = params['env_max_h']
#         self._env = params['env']
#         self._env_name = params['env_name']
#         self._init_obs = self._env.reset()
#         self._is_goal_env = False
#         if isinstance(self._init_obs, dict):
#             self._is_goal_env = True
#             self._init_obs = self._init_obs['observation']
#         self._obs_dim = params['obs_dim']
#         self._act_dim = params['action_dim']
#         self.fitness_func = params['fitness_func']
#         self.nb_thread = cpu_count() - 1 or 1
#         ## Get the controller and initialize it from params
#         self.controller = params['controller_type'](params)
#         ## Get policy parameters init min and max
#         self._policy_param_init_min = params['policy_param_init_min'] 
#         self._policy_param_init_max = params['policy_param_init_max']
#         ## Get size of policy parameter vector
#         self.policy_representation_dim = len(self.controller.get_parameters())
#         self.dynamics_model = None
        
        
#     def set_dynamics_model(self, dynamics_model):
#         self.dynamics_model = dynamics_model
        
#     ## For each env, do a BD + Fitness based on traj
#     ## mb best solution is to put it in the envs directly
#     ## check what obs_traj and act_traj looks like in src/envs/hexapod_env.py
#     def evaluate_solution(self, ctrl, render=False):
#         """
#         Input: ctrl (array of floats) the genome of the individual
#         Output: Trajectory and actions taken
#         """
#         ## Create a copy of the controller
#         controller = self.controller.copy()
#         ## Verify that x and controller parameterization have same size
#         # assert len(x) == len(self.controller.get_parameters())
#         ## Set controller parameters
#         controller.set_parameters(ctrl)
#         env = copy.copy(self._env) ## need to verify this works
#         obs = env.reset()
#         act_traj = []
#         obs_traj = []
#         ## WARNING: need to get previous obs
#         for t in range(self._env_max_h):
#             if self._is_goal_env:
#                 obs = obs['observation']
#             action = controller(obs)
#             action[action>self._action_max] = self._action_max
#             action[action<self._action_min] = self._action_min
#             obs_traj.append(obs)
#             act_traj.append(action)
#             obs, reward, done, info = self._env.step(action)
#             if done:
#                 break
#         # if self._is_goal_env:
#             # obs = obs['observation']
#         # obs_traj.append(obs)

#         desc = self.compute_bd(obs_traj)
#         fitness = self.compute_fitness(obs_traj, act_traj)

#         if render:
#             print("Desc from simulation", desc)

#         return fitness, desc, obs_traj, act_traj, 0 # 0 is disagr

#     def evaluate_solution_model(self, ctrl, mean=False, det=True, render=False):
#         """
#         Input: ctrl (array of floats) the genome of the individual
#         Output: Trajectory and actions taken
#         """
#         ## Create a copy of the controller
#         controller = self.controller.copy()
#         ## Verify that x and controller parameterization have same size
#         # assert len(x) == len(self.controller.get_parameters())
#         ## Set controller parameters
#         controller.set_parameters(ctrl)
#         env = copy.copy(self._env) ## need to verify this works
#         obs = self._init_obs
#         act_traj = []
#         obs_traj = []
#         ## WARNING: need to get previous obs
#         for t in range(self._env_max_h):
#             action = controller(obs)
#             action[action>self._action_max] = self._action_max
#             action[action<self._action_min] = self._action_min
#             obs_traj.append(obs)
#             act_traj.append(action)

#             s = ptu.from_numpy(np.array(obs))
#             a = ptu.from_numpy(np.array(action))
#             s = s.view(1,-1)
#             a = a.view(1,-1)

#             if det:
#                 # if deterministic dynamics model
#                 pred_delta_ns = self.dynamics_model.output_pred(torch.cat((s, a), dim=-1))
#             else:
#                 # if probalistic dynamics model - choose output mean or sample
#                 pred_delta_ns = self.dynamics_model.output_pred(torch.cat((s, a), dim=-1), mean=mean)
#             obs = pred_delta_ns[0] + obs # the [0] just seelect the row [1,state_dim]
            
#         obs_traj.append(obs)

#         desc = self.compute_bd(obs_traj)
#         fitness = self.compute_fitness(obs_traj, act_traj)

#         if render:
#             print("Desc from model", desc)

#         return fitness, desc, obs_traj, act_traj

#     def evaluate_solution_model_ensemble(self, ctrl, mean=True, disagr=True, render=False):
#         """
#         Input: ctrl (array of floats) the genome of the individual
#         Output: Trajectory and actions taken
#         """
#         ## Create a copy of the controller
#         controller = self.controller.copy()
#         ## Verify that x and controller parameterization have same size
#         # assert len(x) == len(self.controller.get_parameters())
#         ## Set controller parameters
#         controller.set_parameters(ctrl)
#         env = copy.copy(self._env) ## need to verify this works
#         obs = self._init_obs
#         act_traj = []
#         obs_traj = []
#         disagr_traj = []
#         obs = np.tile(obs,(self.dynamics_model.ensemble_size, 1))
#         ## WARNING: need to get previous obs
#         for t in range(self._env_max_h):
#             ## Get mean obs to determine next action
#             # mean_obs = [np.mean(obs[:,i]) for i in range(len(obs[0]))]
#             action = controller(obs)
#             action[action>self._action_max] = self._action_max
#             action[action<self._action_min] = self._action_min
#             # if t == 0:
#                 # obs = np.tile(obs,(self.dynamics_model.ensemble_size, 1))
#             obs_traj.append(obs)
#             act_traj.append(action)

#             s = ptu.from_numpy(np.array(obs))
#             a = ptu.from_numpy(np.array(action))

#             # if t ==0:
#                 # a = a.repeat(self.dynamics_model.ensemble_size,1)
            
#             # if probalistic dynamics model - choose output mean or sample
#             if disagr:
#                 pred_delta_ns, _ = self.dynamics_model.sample_with_disagreement(torch.cat((
#                     self.dynamics_model._expand_to_ts_form(s),
#                     self.dynamics_model._expand_to_ts_form(a)), dim=-1))#,
#                     # disagreement_type="mean" if mean else "var")
#                 pred_delta_ns = ptu.get_numpy(pred_delta_ns)
#                 disagreement = self.compute_abs_disagreement(obs, pred_delta_ns)
#                 # print("Disagreement: ", disagreement.shape)
#                 # print("Disagreement: ", disagreement)
#                 disagreement = ptu.get_numpy(disagreement) 
#                 #disagreement = ptu.get_numpy(disagreement[0,3]) 
#                 #disagreement = ptu.get_numpy(torch.mean(disagreement)) 
#                 disagr_traj.append(disagreement)
                
#             else:
#                 pred_delta_ns = self.dynamics_model.output_pred_ts_ensemble(s,a, mean=mean)

#             # mean_pred = [np.mean(pred_delta_ns[:,i]) for i in range(len(pred_delta_ns[0]))]
#             obs = pred_delta_ns + obs # This keeps all model predictions separated
#             # obs = mean_pred + obs # This uses mean prediction
            
#         # obs_traj.append(obs)

#         obs_traj = np.array(obs_traj)
#         act_traj = np.array(act_traj)
        
#         desc = self.compute_bd(obs_traj, ensemble=True)
#         fitness = self.compute_fitness(obs_traj, act_traj, disagr_traj=disagr_traj, ensemble=True)

#         if render:
#             print("Desc from model", desc)

#         return fitness, desc, obs_traj, act_traj, disagr_traj

#     def evaluate_solution_model_ensemble_all(self, ctrls, mean=True, disagr=True,
#                                                  render=False, use_particules=True):
#         """
#         Input: ctrl (array of floats) the genome of the individual
#         Output: Trajectory and actions taken
#         """
#         controller_list = []
#         traj_list = []
#         actions_list = []
#         disagreements_list = []
#         obs_list = []

#         env = copy.copy(self._env) ## need to verify this works
#         obs = self._init_obs

#         for ctrl in ctrls:
#             ## Create a copy of the controller
#             controller_list.append(self.controller.copy())
#             ## Set controller parameters
#             controller_list[-1].set_parameters(ctrl)
#             traj_list.append([])
#             actions_list.append([])
#             disagreements_list.append([])
#             obs_list.append(obs.copy())

#         ens_size = self.dynamics_model.ensemble_size

#         if use_particules:
#             S = np.tile(obs, (ens_size*len(ctrls), 1))
#             A = np.empty((ens_size*len(ctrls),
#                           self.controller.output_dim))
#         else:
#             ## WARNING: need to get previous obs
#             # S = np.tile(prev_element.trajectory[-1].copy(), (len(X)))
#             S = np.tile(obs, (len(ctrls), 1))
#             A = np.empty((len(ctrls), self.controller.output_dim))

#         for _ in tqdm.tqdm(range(self._env_max_h), total=self._env_max_h):
#             for i in range(len(ctrls)):
#                 # A[i, :] = controller_list[i](S[i,:])
#                 if use_particules:
#                     A[i*ens_size:i*ens_size+ens_size] = \
#                         controller_list[i](S[i*ens_size:i*ens_size+ens_size])
#                 else:
#                     A[i] = controller_list[i](S[i])
                
#             start = time.time()
#             if use_particules:
#                 batch_pred_delta_ns, batch_disagreement = self.forward(A, S, mean=mean,
#                                                                        disagr=disagr,
#                                                                        multiple=True)
#             else:
#                 batch_pred_delta_ns, batch_disagreement = self.forward_multiple(A, S,
#                                                                                 mean=True,
#                                                                                 disagr=True)
#             # print(f"Time for inference {time.time()-start}")
#             for i in range(len(ctrls)):
#                 if use_particules:
#                     ## Don't use mean predictions and keep each particule trajectory
#                     # Be careful, in that case there is no need to repeat each state in
#                     # forward multiple function
#                     disagreement = self.compute_abs_disagreement(S[i*ens_size:i*ens_size+ens_size]
#                                                                  , batch_pred_delta_ns[i])
#                     # print("Disagreement: ", disagreement.shape)
#                     # print("Disagreement: ", disagreement)
#                     disagreement = ptu.get_numpy(disagreement)
                    
#                     disagreements_list[i].append(disagreement.copy())
                    
#                     S[i*ens_size:i*ens_size+ens_size] += batch_pred_delta_ns[i]
#                     traj_list[i].append(S[i*ens_size:i*ens_size+ens_size].copy())

#                     # disagreements_list[i].append(batch_disagreement[i])
#                     actions_list[i].append(A[i*ens_size:i*ens_size+ens_size])

#                 else:
#                     ## Compute mean prediction from model samples
#                     next_step_pred = batch_pred_delta_ns[i]
#                     mean_pred = [np.mean(next_step_pred[:,i]) for i
#                                  in range(len(next_step_pred[0]))]
#                     S[i,:] += mean_pred.copy()
#                     traj_list[i].append(S[i,:].copy())
#                     disagreements_list[i].append(batch_disagreement[i])
#                     actions_list[i].append(A[i,:])

#         bd_list = []
#         fit_list = []

#         obs_trajs = np.array(traj_list)
#         act_trajs = np.array(actions_list)
#         disagr_trajs = np.array(disagreements_list)

#         for i in range(len(ctrls)):
#             obs_traj = obs_trajs[i]
#             act_traj = act_trajs[i]
#             disagr_traj = disagr_trajs[i]

#             desc = self.compute_bd(obs_traj, ensemble=True)
#             fitness = self.compute_fitness(obs_traj, act_traj,
#                                            disagr_traj=disagr_traj,
#                                            ensemble=True)

#             fit_list.append(fitness)
#             bd_list.append(desc)
            
#         return fit_list, bd_list, obs_trajs, act_trajs, disagr_trajs

#     def forward_multiple(self, A, S, mean=True, disagr=True):
#         ## Takes a list of actions A and a list of states S we want to query the model from
#         ## Returns a list of the return of a forward call for each couple (action, state)
#         assert len(A) == len(S)
#         batch_len = len(A)
#         ens_size = self.dynamics_model.ensemble_size

#         S_0 = np.empty((batch_len*ens_size, S.shape[1]))
#         A_0 = np.empty((batch_len*ens_size, A.shape[1]))

#         batch_cpt = 0
#         for a, s in zip(A, S):
#             S_0[batch_cpt*ens_size:batch_cpt*ens_size+ens_size,:] = \
#             np.tile(s,(self.dynamics_model.ensemble_size, 1))
#             # np.tile(copy.deepcopy(s),(self._dynamics_model.ensemble_size, 1))

#             A_0[batch_cpt*ens_size:batch_cpt*ens_size+ens_size,:] = \
#             np.tile(a,(self.dynamics_model.ensemble_size, 1))
#             # np.tile(copy.deepcopy(a),(self._dynamics_model.ensemble_size, 1))
#             batch_cpt += 1
#         # import pdb; pdb.set_trace()
#         return self.forward(A_0, S_0, mean=mean, disagr=disagr, multiple=True)

#         # return batch_pred_delta_ns, batch_disagreement

#     def forward(self, a, s, mean=True, disagr=True, multiple=False):
#         s_0 = copy.deepcopy(s)
#         a_0 = copy.deepcopy(a)

#         if not multiple:
#             s_0 = np.tile(s_0,(self.dynamics_model.ensemble_size, 1))
#             a_0 = np.tile(a_0,(self.dynamics_model.ensemble_size, 1))

#         s_0 = ptu.from_numpy(s_0)
#         a_0 = ptu.from_numpy(a_0)

#         # a_0 = a_0.repeat(self._dynamics_model.ensemble_size,1)

#         # if probalistic dynamics model - choose output mean or sample
#         if disagr:
#             if not multiple:
#                 pred_delta_ns, disagreement = self.dynamics_model.sample_with_disagreement(
#                     torch.cat((
#                         self.dynamics_model._expand_to_ts_form(s_0),
#                         self.dynamics_model._expand_to_ts_form(a_0)), dim=-1
#                     ))#, disagreement_type="mean" if mean else "var")
#                 pred_delta_ns = ptu.get_numpy(pred_delta_ns)
#                 return pred_delta_ns, disagreement
#             else:
#                 pred_delta_ns_list, disagreement_list = \
#                 self.dynamics_model.sample_with_disagreement_multiple(
#                     torch.cat((
#                         self.dynamics_model._expand_to_ts_form(s_0),
#                         self.dynamics_model._expand_to_ts_form(a_0)), dim=-1
#                     ))#, disagreement_type="mean" if mean else "var")
#                 for i in range(len(pred_delta_ns_list)):
#                     pred_delta_ns_list[i] = ptu.get_numpy(pred_delta_ns_list[i])
#                 return pred_delta_ns_list, disagreement_list
#         else:
#             pred_delta_ns = self.dynamics_model.output_pred_ts_ensemble(s_0, a_0, mean=mean)
#         return pred_delta_ns, 0

    
#     def compute_abs_disagreement(self, cur_state, pred_delta_ns):
#         '''
#         Computes absolute state dsiagreement between models in the ensemble
#         cur state is [4,48]
#         pred delta ns [4,48]
#         '''
#         next_state = pred_delta_ns + cur_state
#         next_state = ptu.from_numpy(next_state)
#         mean = next_state

#         sample=False
#         if sample: 
#             inds = torch.randint(0, mean.shape[0], next_state.shape[:1]) #[4]
#             inds_b = torch.randint(0, mean.shape[0], next_state.shape[:1]) #[4]
#             inds_b[inds == inds_b] = torch.fmod(inds_b[inds == inds_b] + 1, mean.shape[0]) 
#         else:
#             inds = torch.tensor(np.array([0,0,0,1,1,2]))
#             inds_b = torch.tensor(np.array([1,2,3,2,3,3]))

#         # Repeat for multiplication
#         inds = inds.unsqueeze(dim=-1).to(device=ptu.device)
#         inds = inds.repeat(1, mean.shape[1])
#         inds_b = inds_b.unsqueeze(dim=-1).to(device=ptu.device)
#         inds_b = inds_b.repeat(1, mean.shape[1])

#         means_a = (inds == 0).float() * mean[0]
#         means_b = (inds_b == 0).float() * mean[0]
#         for i in range(1, mean.shape[0]):
#             means_a += (inds == i).float() * mean[i]
#             means_b += (inds_b == i).float() * mean[i]
            
#         disagreements = torch.mean(torch.sqrt((means_a - means_b)**2), dim=-2, keepdim=True)
#         #disagreements = torch.mean((means_a - means_b) ** 2, dim=-1, keepdim=True)

#         return disagreements

#     def compute_bd(self, obs_traj, ensemble=False, mean=True):
#         bd = None
#         last_obs = obs_traj[-1]
#         if ensemble:
#             if mean:
#                 last_obs = np.mean(last_obs, axis=0)
#             else:
#                 last_obs = last_obs[np.random.randint(self.dynamics_model.ensemble_size)]
                
#         if self._env_name == 'ball_in_cup':
#             bd = last_obs[:3]
#         if self._env_name == 'fastsim_maze':
#             bd = last_obs[:2]
#         if self._env_name == 'fastsim_maze_traps':
#             bd = last_obs[:2]
#         if self._env_name == 'redundant_arm_no_walls_limited_angles':
#             bd = last_obs[-2:]
#         return bd

#     def energy_minimization_fit(self, actions, disagrs):
#         return -np.sum(np.abs(actions))

#     def disagr_minimization_fit(self, actions, disagrs):
#         if disagrs is None: # was a real world eval
#             return 0
#         return -np.sum(disagrs)

#     def compute_fitness(self, obs_traj, act_traj, disagr_traj=None, ensemble=False):
#         fit = 0
#         ## Energy minimization fitness
#         if self.fitness_func == 'energy_minimization':
#             fit_func = self.energy_minimization_fit
#         elif self.fitness_func == 'disagr_minimization':
#             fit_func = self.disagr_minimization_fit
#         if self._env_name == 'ball_in_cup':
#             fit = fit_func(act_traj, disagr_traj)
#         if self._env_name == 'fastsim_maze':
#             fit = fit_func(act_traj, disagr_traj)
#         if self._env_name == 'fastsim_maze_traps':
#             fit = fit_func(act_traj, disagr_traj)
#         if self._env_name == 'redundant_arm_no_walls_limited_angles':
#             fit = fit_func(act_traj, disagr_traj)
#         return fit

# def get_env_params(args):
#     env_params = {}

#     env_params['env_register_id'] = 'fastsim_maze_laser'
#     env_params['is_local_env'] = False
#     env_params['init_obs'] = None
#     env_params['gym_args'] = {}
#     env_params['a_min'] = None
#     env_params['a_max'] = None
#     env_params['ss_min'] = None
#     env_params['ss_max'] = None
#     env_params['dim_map'] = None
    
#     if args.environment == 'ball_in_cup':
#         import mb_ge ## Contains ball in cup
#         env_params['env_register_id'] = 'BallInCup3d-v0'
#         env_params['a_min'] = np.array([-1, -1, -1])
#         env_params['a_max'] = np.array([1, 1, 1])
#         env_params['ss_min'] = np.array([-0.4]*6)
#         env_params['ss_max'] = np.array([0.4]*6)
#         env_params['obs_min'] = np.array([-0.4]*6)
#         env_params['obs_max'] = np.array([0.4]*6)
#         env_params['dim_map'] = 3
#         env_params['init_obs'] = np.array([300., 300., 0., 0., 0. , 0.])
#         env_params['state_dim'] = 6
#         env_params['bd_inds'] = [0, 1, 2]
        
#     elif args.environment == 'redundant_arm':
#         import redundant_arm ## contains classic redundant arm
#         env_params['gym_args']['dof'] = 20
        
#         env_register_id = 'RedundantArmPos-v0'
#         env_params['a_min'] = np.array([-1]*env_params['gym_args']['dof'])
#         env_params['a_max'] = np.array([1]*env_params['gym_args']['dof'])
#         env_params['ss_min'] = np.array([-0.5]*env_params['gym_args']['dof']+[0,0])
#         env_params['ss_max'] = np.array([0.8]*env_params['gym_args']['dof']+[1,1])
#         # ss_min = np.array([-2.91586418e-01, -2.91059290e-01, -4.05994661e-01,
#         #                    -3.43161155e-01, -4.48797687e-01, -3.42430607e-01,
#         #                    -4.64587165e-01, -4.57486040e-01, -4.40965296e-01,
#         #                    -3.74359165e-01, -4.73628034e-01, -3.64009843e-01,
#         #                    -4.78609985e-01, -4.22113313e-01, -5.27555361e-01,
#         #                    -5.18617559e-01, -4.36935815e-01, -5.31945509e-01,
#         #                    -4.44923835e-01, -5.36581457e-01, 2.33058244e-05,
#         #                    7.98103927e-05])
#         # ss_max = np.array([0.8002732,  0.74879046, 0.68724849, 0.76289724,
#         #                    0.66943127, 0.77772601, 0.67210694, 0.56392794,
#         #                    0.65394265, 0.74616584, 0.61193007, 0.73037668,
#         #                    0.59987872, 0.71458412, 0.58088037, 0.60106068,
#         #                    0.66026566, 0.58433874, 0.64901992, 0.44800244,
#         #                    0.99999368, 0.99999659])
#         env_params['obs_min'] = env_params['ss_min']
#         env_params['obs_max'] = env_params['ss_max']
#         env_params['state_dim'] = env_params['gym_args']['dof'] + 2
#         env_params['dim_map'] = 2
#         env_params['bd_inds'] = [-2, -1]
#     elif args.environment == 'redundant_arm_no_walls':
#         env_params['gym_args']['dof'] = 20
#         env_params['env_register_id'] = 'RedundantArmPosNoWalls-v0'
#         env_params['a_min'] = np.array([-1]*env_params['gym_args']['dof'])
#         env_params['a_max'] = np.array([1]*env_params['gym_args']['dof'])
#         env_params['ss_min'] = -1
#         env_params['ss_max'] = 1
#         env_params['ss_min'] = np.array([-0.5]*env_params['gym_args']['dof']+[0,0])
#         env_params['ss_max'] = np.array([0.8]*env_params['gym_args']['dof']+[1,1])
#         env_params['obs_min'] = env_params['ss_min']
#         env_params['obs_max'] = env_params['ss_max']
#         env_params['state_dim'] = env_params['gym_args']['dof'] + 2
#         env_params['dim_map'] = 2
#         env_params['bd_inds'] = [-2, -1]
#     elif args.environment == 'redundant_arm_no_walls_no_collision':
#         env_params['gym_args']['dof'] = 20
#         env_params['env_register_id'] = 'RedundantArmPosNoWallsNoCollision-v0'
#         env_params['a_min'] = np.array([-1]*env_params['gym_args']['dof'])
#         env_params['a_max'] = np.array([1]*env_params['gym_args']['dof'])
#         env_params['ss_min'] = -1
#         env_params['ss_max'] = 1
#         env_params['ss_min'] = np.array([-0.5]*env_params['gym_args']['dof']+[0,0])
#         env_params['ss_max'] = np.array([0.8]*env_params['gym_args']['dof']+[1,1])
#         env_params['obs_min'] = env_params['ss_min']
#         env_params['obs_max'] = env_params['ss_max']
#         env_params['state_dim'] = env_params['gym_args']['dof'] + 2
#         env_params['dim_map'] = 2
#         env_params['bd_inds'] = [-2, -1]
#     elif args.environment == 'redundant_arm_no_walls_limited_angles':
#         env_params['gym_args']['dof'] = 100
#         env_params['env_register_id'] = 'RedundantArmPosNoWallsLimitedAngles-v0'
#         env_params['a_min'] = np.array([-1]*env_params['gym_args']['dof'])
#         env_params['a_max'] = np.array([1]*env_params['gym_args']['dof'])
#         env_params['ss_min'] = -1
#         env_params['ss_max'] = 1
#         env_params['ss_min'] = np.array([-0.5]*env_params['gym_args']['dof']+[0,0])
#         env_params['ss_max'] = np.array([0.8]*env_params['gym_args']['dof']+[1,1])
#         env_params['obs_min'] = env_params['ss_min']
#         env_params['obs_max'] = env_params['ss_max']
#         env_params['state_dim'] = env_params['gym_args']['dof'] + 2
#         env_params['dim_map'] = 2
#         env_params['gym_args']['dof'] = 100
#         env_params['bd_inds'] = [-2, -1]
#     elif args.environment == 'fastsim_maze_laser':
#         env_params['env_register_id'] = 'FastsimSimpleNavigation-v0'
#         env_params['a_min'] = np.array([-1, -1])
#         env_params['a_max'] = np.array([1, 1])
#         env_params['ss_min'] = np.array([0, 0, -1, -1, -1, -1])
#         env_params['ss_max'] = np.array([600, 600, 1, 1, 1, 1])
#         env_params['init_obs'] = np.array([60., 450., 0., 0., 0. , 0.])
#         env_params['state_dim'] = 6
#         env_params['obs_min'] = np.array([0, 0, 0, 0, 0])
#         env_params['obs_max'] = np.array([100, 100, 100, 1, 1])
#         env_params['dim_map'] = 2
#         env_params['bd_inds'] = [0, 1]
#         args.use_obs_model = True
#     elif args.environment == 'empty_maze_laser':
#         env_params['env_register_id'] = 'FastsimEmptyMapNavigation-v0'
#         env_params['a_min'] = np.array([-1, -1])
#         env_params['a_max'] = np.array([1, 1])
#         env_params['ss_min'] = np.array([0, 0, -1, -1, -1, -1])
#         env_params['ss_max'] = np.array([600, 600, 1, 1, 1, 1])
#         env_params['init_obs'] = np.array([300., 300., 0., 0., 0. , 0.])
#         env_params['state_dim'] = 6
#         env_params['obs_min'] = np.array([0, 0, 0, 0, 0])
#         env_params['obs_max'] = np.array([100, 100, 100, 1, 1])
#         env_params['dim_map'] = 2
#         env_params['bd_inds'] = [0, 1]
#         args.use_obs_model = True
#     elif args.environment == 'fastsim_maze':
#         env_params['env_register_id'] = 'FastsimSimpleNavigationPos-v0'
#         env_params['a_min'] = np.array([-1, -1])
#         env_params['a_max'] = np.array([1, 1])
#         env_params['obs_min'] = env_params['ss_min'] = np.array([0, 0, -1, -1, -1, -1])
#         env_params['obs_max'] = env_params['ss_max'] = np.array([600, 600, 1, 1, 1, 1])
#         env_params['state_dim'] = 6
#         #env_params['init_obs'] = np.array([60., 450., 0., 0., 0. , 0.])
#         env_params['dim_map'] = 2
#         env_params['bd_inds'] = [0, 1]
#     elif args.environment == 'empty_maze':
#         env_params['env_register_id'] = 'FastsimEmptyMapNavigationPos-v0'
#         env_params['a_min'] = np.array([-1, -1])
#         env_params['a_max'] = np.array([1, 1])
#         env_params['obs_min'] = env_params['ss_min'] = np.array([0, 0, -1, -1, -1, -1])
#         env_params['obs_max'] = env_params['ss_max'] = np.array([600, 600, 1, 1, 1, 1])
#         env_params['state_dim'] = 6
#         #env_params['init_obs = np.array([300., 300., 0., 0., 0. , 0.])
#         env_params['dim_map'] = 2
#         env_params['bd_inds'] = [0, 1]
#     elif args.environment == 'fastsim_maze_traps':
#         env_params['env_register_id'] = 'FastsimSimpleNavigationPos-v0'
#         env_params['a_min'] = np.array([-1, -1])
#         env_params['a_max'] = np.array([1, 1])
#         env_params['obs_min'] = env_params['ss_min'] = np.array([0, 0, -1, -1, -1, -1])
#         env_params['obs_max'] = env_params['ss_max'] = np.array([600, 600, 1, 1, 1, 1])
#         env_params['state_dim'] = 6
#         env_params['dim_map'] = 2
#         env_params['gym_args']['physical_traps'] = True
#         env_params['bd_inds'] = [0, 1]
#     elif args.environment == 'half_cheetah':
#         env_params['env_register_id'] = 'HalfCheetah-v3'
#         env_params['a_min'] = np.array([-1, -1, -1, -1, -1, -1])
#         env_params['a_max'] = np.array([1, 1, 1, 1, 1, 1])
#         ## Got these with NS 100 000 eval budget
#         env_params['ss_min'] = np.array([-49.02189923, -0.61095456, -16.64607454, -0.70108701,
#                            -1.00943152, -0.65815842, -1.19701832, -1.28944137,
#                            -0.76604915, -5.2375874, -5.51574707, -10.4422284,
#                            -26.43682609, -31.22491269, -31.96452725,
#                            -26.68346276, -32.95576583, -32.70174356])
#         env_params['ss_max'] = np.array([32.47642872, 0.83392967, 38.93965081, 1.14752425,
#                            0.93195033, 0.95062493, 0.88961483, 1.11808423,
#                            0.76134696, 4.81465142, 4.9208565, 10.81297147,
#                            25.82911106, 28.41785798, 24.95866255, 31.30177305,
#                            34.88956652, 30.07857634])

#         env_params['init_obs'] = np.array([0.]*18)
#         env_params['dim_map'] = 1
#         env_params['gym_args']['exclude_current_positions_from_observation'] = False
#         env_params['gym_args']['reset_noise_scale'] = 0
#     elif args.environment == 'walker2d':
#         env_params['env_register_id'] = 'Walker2d-v3'
#         env_params['a_min'] = np.array([-1, -1, -1, -1, -1, -1])
#         env_params['a_max'] = np.array([1, 1, 1, 1, 1, 1])
#         ## Got these with NS 100 000 eval budget
#         env_params['ss_min'] = np.array([-4.26249395, 0.75083099, -1.40787207, -2.81284653,
#                            -2.93150238, -1.5855295, -3.04205169, -2.91603065,
#                            -1.62175821, -7.11379591, -10., -10., -10., -10.,
#                            -10., -10., -10., -10.])
#         env_params['ss_max'] = np.array([1.66323372, 1.92256493, 1.15429141, 0.43140988,
#                            0.49341738, 1.50477799, 0.47811355, 0.63702984,
#                            1.50380045, 4.98763458, 4.00820283, 10., 10., 10.,
#                            10., 10., 10., 10.])

#         ## Got these with NS 100 000 eval budget
#         env_params['ss_min'] = np.array([-5.62244541, 0.7439814, -1.41163676, -3.1294922,
#                            -2.97025984, -1.67482138, -3.1644274, -3.01373681,
#                            -1.78557467, -8.55243269, -10., -10., -10., -10.,
#                            -10., -10., -10., -10.])
#         env_params['ss_max'] = np.array([1.45419434, 1.98069464, 1.1196152, 0.5480219,
#                            0.65664259, 1.54582436, 0.53905455, 0.61275703,
#                            1.5541609, 6.12093722, 5.9363082, 10., 10., 10.,
#                            10., 10., 10., 10.])
#         env_params['init_obs'] = np.array([0.]*18)
#         env_params['dim_map'] = 1
#         env_params['gym_args']['exclude_current_positions_from_observation'] = False
#         env_params['gym_args']['reset_noise_scale'] = 0
#     elif args.environment == 'hexapod_omni':
#         from src.envs.hexapod_dart.hexapod_env import HexapodEnv ## Contains hexapod 
#         env_params['is_local_env'] = True
#         max_step = 300 # ctrl_freq = 100Hz, sim_time = 3.0 seconds 
#         env_params['obs_dim'] = 48
#         env_params['act_dim'] = 18
#         env_params['dim_x'] = 36
#         ## Need to check the dims for hexapod
#         env_params['ss_min']= -1
#         env_params['ss_max'] = 1
#         env_params['dim_map'] = 2
#     else:
#         raise ValueError(f"{args.environment} is not a defined environment")

#     return env_params

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
        elif args.model_type == "prob" and args.environment == 'hexapod_omni':
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

    print()
    print(f'Finished performing mbqd search successfully.')
    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    args = process_args()

    main(args)
