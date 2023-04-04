#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
import os, sys
import time
import math, random
import numpy as np
import multiprocessing

# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree

from src.map_elites import common as cm
from src.map_elites import unstructured_container, cvt
from src.map_elites import model_condition_utils

import torch
import src.torch.pytorch_util as ptu

import cma

from multiprocessing import get_context

def evaluate_(t):
    # evaluate a single vector (x) with a function f and return a species
    # evaluate z with function f - z is the genotype and f is the evalution function
    # t is the tuple from the to_evaluate list
    z, f = t
    fit, desc, obs_traj, act_traj, disagr = f(z) 
    if hasattr(desc[0], '__len__'):
        desc = desc[0]    
    # becasue it somehow returns a list in a list (have to keep checking sometimes)
    # desc = desc[0] # important - if not it fails the KDtree for cvt and grid map elites
    # disagr = 0 # no disagreement for real evalaution - but need to put to save archive
    # return a species object (containing genotype, descriptor and fitness)
    return cm.Species(z, desc, fit, obs_traj=obs_traj, act_traj=act_traj, model_dis=disagr)

def model_evaluate_(t):
    # same as the above evaluate but this takes in the disagreement also
    # - useful if you want to make use disargeement value
    # needs two types because no such thing as disagreemnt for real eval
    z, f = t
    fit, desc, obs_traj, act_traj, disagr = f(z) 
    if hasattr(desc[0], '__len__'):
        desc = desc[0]
    # becasue it somehow returns a list in a list (have to keep checking sometimes)
    # desc = desc[0] # important - if not it fails the KDtree for cvt and grid map elites
    
    # return a species object (containing genotype, descriptor and fitness)
    return cm.Species(z, desc, fit, obs_traj=obs_traj, act_traj=act_traj, model_dis=disagr)

def model_evaluate_all_(T):
    # same as the above evaluate but this takes in the disagreement also
    # - useful if you want to make use disargeement value
    # needs two types because no such thing as disagreemnt for real eval
    Z = [T[i][0] for i in range(len(T))]
    f = T[0][1]
    fit_list, desc_list, obs_traj_list, act_traj_list, disagr_list = f(Z) 
    
    # becasue it somehow returns a list in a list (have to keep checking sometimes)
    # desc = desc[0] # important - if not it fails the KDtree for cvt and grid map elites
    
    # return a species object (containing genotype, descriptor and fitness)
    model_inds = []
    for i in range(len(T)):
        model_inds.append(cm.Species(Z[i], desc_list[i], fit_list[i], obs_traj=obs_traj_list[i],
                                     act_traj=act_traj_list[i], model_dis=disagr_list[i]))
    return model_inds

class ModelBasedQD:
    def __init__(self,
                 dim_map, dim_x,
                 f_real, f_model,
                 model, model_trainer,
                 dynamics_model, dynamics_model_trainer,
                 replay_buffer,
                 n_niches=1000,
                 params=cm.default_params,
                 bins=None,
                 log_dir='./',):

        torch.set_num_threads(24)
        
        self.qd_type = params["type"]    # QD type - grid, cvt, unstructured
        self.dim_map = dim_map           # number of BD dimensions  
        self.dim_x = dim_x               # gemotype size (number of genotype dim)
        self.n_niches = n_niches         # estimated total population in archive
        self.bins = bins                 # grid shape - only for grid map elites
        self.params = params

        # 2 eval functions
        # 1 for real eval, 1 for model eval (imagination)
        self.f_real = f_real
        
        if params["model_variant"]=="dynamics":
            self.f_model = f_model
            print("Dynamics Model Variant")
        if params["model_variant"]=="all_dynamics":
            self.f_model = f_model
            print("All Dynamics Model Variant")
        elif params["model_variant"]=="direct":
            self.f_model = self.evaluate_solution_surrogate_model 
            print("Direct Model Variant")
            
        # Model and Model trainer init -
        # initialize the classes outside this class and pass in
        self.model = model # refers to the direct qd surrogate model
        self.model_trainer = model_trainer # direct qd surrogate trainer

        self.dynamics_model = dynamics_model
        self.dynamics_model_trainer = dynamics_model_trainer

        self.replay_buffer = replay_buffer
        self.all_real_evals = []
        
        # Init logging directory and log file
        self.log_dir = log_dir
        log_filename = self.log_dir + '/log_file.dat'
        self.log_file = open(log_filename, 'w')
    
        # path and filename to save model
        self.save_model_path = self.log_dir + '/trained_model.pth'
        
        if params['log_time_stats']:
            time_stats_filename = self.log_dir + '/time_log_file.dat'
            self.time_log_file = open(time_stats_filename, 'w')
            self.gen_time = 0
            self.model_eval_time = 0
            self.eval_time = 0
            self.model_train_time = 0 

        if 'dab_params' in params:
            o_params = params['dab_params']
            self.o_params = params['dab_params']
                
        # Init cvt and grid - only if cvt and grid map elites used
        if (self.qd_type=="cvt") or (self.qd_type=="grid"):
            c = []
            if (self.qd_type=="cvt"):
                c = cm.cvt(self.n_niches,
                           self.dim_map,params['cvt_samples'],
                           params['cvt_use_cache'])
            else:
                if self.bins is None:
                    self.bins = [50]*params['dim_map']
                    print(f"WARNING: Using {self.bins} as bins (default)")
                bd_limits = None
                if 'dab_params' in params:
                    self.o_params['bins'] = self.bins
                    self.bd_inds = self.o_params['bd_inds']
                    bd_max = self.o_params['state_max'][self.bd_inds]
                    bd_min = self.o_params['state_min'][self.bd_inds]
                    bd_limits = [[a, b] for (a,b) in zip(bd_min, bd_max)]
                
                c = cm.grid_centroids(self.bins, bd_limits=bd_limits)

            self.kdt = KDTree(c, leaf_size=30, metric='euclidean')
            cm._write_centroids(c)
            
        if (self.qd_type == "cvt") or (self.qd_type=="grid"):
            self.archive = {} # init archive as dic (empty)
            self.model_archive = {}
        elif self.qd_type == "unstructured":
            self.archive = [] # init archive as list
            self.model_archive = []        


    def random_archive_init(self, to_evaluate):
        for i in range(0, self.params['random_init_batch']):
            x = np.random.uniform(low=self.params['min'], high=self.params['max'],size=self.dim_x)
            to_evaluate += [(x, self.f_real)]
        
        return to_evaluate

    def random_archive_init_model(self, to_evaluate):
        for i in range(0, self.params['random_init_batch']):
            x = np.random.uniform(low=self.params['min'], high=self.params['max'],size=self.dim_x)
            to_evaluate += [(x, self.f_model)]
        
        return to_evaluate


    def select_and_mutate(self, to_evaluate, archive, f, params, variation_operator=cm.variation, batch=False):

        if (self.qd_type=="cvt") or (self.qd_type=="grid"):
            keys = list(archive.keys())
        elif (self.qd_type=="unstructured"):
            keys = archive
                    
        # we select all the parents at the same time because randint is slow
        rand1 = np.random.randint(len(keys), size=self.params['batch_size'])
        rand2 = np.random.randint(len(keys), size=self.params['batch_size'])
            
        for n in range(0, params['batch_size']):
            # parent selection - mutation operators like iso_dd/sbx require 2 gen parents
            if (self.qd_type == "cvt") or (self.qd_type=="grid"):
                x = archive[keys[rand1[n]]]
                y = archive[keys[rand2[n]]]
            elif (self.qd_type == "unstructured"):                    
                x = archive[rand1[n]]
                y = archive[rand2[n]]
                
            # copy & add variation
            z = variation_operator(x.x, y.x, params)

            if batch:
                to_evaluate += [z]
            else: 
                to_evaluate += [(z, f)]

        return to_evaluate
    
    def addition_condition(self, s_list, archive, params):
        add_list = [] # list of solutions that were added
        discard_list = []
        for s in s_list:
            if self.qd_type == "unstructured":
                success = unstructured_container.add_to_archive(s, archive, params)
            else:
                success = cvt.add_to_archive(s, s.desc, archive, self.kdt)
            if success:
                add_list.append(s)
            else:
                discard_list.append(s) #not important for alogrithm but to collect stats
                
        return archive, add_list, discard_list

    def model_condition(self, s_list, archive, params):
        add_list = [] # list of solutions that are worth evaluating in real life
        discard_list = []
        for s in s_list:
            success = model_condition_utils.add_to_archive(s, archive, params)    
            if success:
                add_list.append(s)
            else:
                discard_list.append(s) # not important for algorithm but to collect stats
                
        return archive, add_list, discard_list

    
    # model based map-elites algorithm
    def compute(self,
                num_cores_set,
                max_evals=1e5,
                params=None,):

        if params is None:
            params = self.params

        # setup the parallel processing pool
        if num_cores_set == 0:
            num_cores = multiprocessing.cpu_count() - 1 # use all cores
        else:
            num_cores = num_cores_set
            
        # pool = multiprocessing.Pool(num_cores)
        pool = get_context("spawn").Pool(num_cores)
        #pool = ThreadPool(num_cores)
        
        gen = 0 # generation
        n_evals = 0 # number of evaluations since the beginning
        b_evals = 0 # number evaluation since the last dump
        n_model_evals = 0 # number of evals done by model

        evals_since_last_train = 0 
        print("################# Starting QD algorithm #################")

        all_errors_medians = []; all_errors_1q = []; all_errors_3q = []
        add_errors_medians = []; add_errors_1q = []; add_errors_3q = []
        discard_errors_medians = []; discard_errors_1q = []; discard_errors_3q = []

        bds_per_gen = []
        bds_per_gen = {}
        # main loop
        while (n_evals < max_evals):
            # lists of individuals we want to evaluate (list of tuples) for this gen
            # each entry in the list is a tuple of the genotype and the evaluation function
            to_evaluate = []
            to_model_evaluate = []

            ## initialize counter for model stats ##
            true_pos = 0
            false_pos = 0
            false_neg = 0
            true_neg = 0

            ## intialize for time related stats ##
            gen_start_time = time.time()
            self.model_train_time = 0
            
            # random initialization of archive - start up
            if (len(self.archive) <= params['random_init']*self.n_niches) \
               and params['init_method'] == 'vanilla':
                print("Evaluation on real environment for initialization")
                to_evaluate = self.random_archive_init(to_evaluate) # init real archive
                #to_evaluate = self.random_archive_init_model(to_evaluate) #init synthetic archive 

                start = time.time()
                ## real sys eval, we force parallel
                params['parallel'] =  True
                s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, params) #init real archive
                params['parallel'] =  False
                #s_list = cm.parallel_eval(model_evaluate_, to_evaluate, pool, params) #init model
                self.eval_time = time.time() - start 
                self.archive, add_list, _ = self.addition_condition(s_list, self.archive, params)

            else:
                print("Evaluation on model")
                # variation/selection loop - select ind from archive to evolve                
                self.model_archive = self.archive.copy()
                tmp_archive = self.archive.copy() # tmp archive for stats of negatives
                #print("Real archive size start: ",len(self.archive))
                #print("Model archive size start: ",len(self.model_archive))

                '''
                # For serial or parallel evaluations
                to_model_evaluate = self.select_and_mutate(to_model_evaluate, self.model_archive, self.f_model, params)
                start = time.time()
                s_list_model = cm.parallel_eval(evaluate_, to_model_evaluate, pool, params)
                self.model_eval_time = time.time() - start
                #s_list_model = self.serial_eval(evaluate_, to_model_evaluate, params)
                
                ### MODEL CONDITIONS ###
                self.model_archive, add_list_model, discard_list_model = self.model_condition(s_list_model, self.model_archive, params)
                #self.model_archive, add_list_model = self.addition_condition(s_list_model, self.model_archive, params)
                #print("Model novel list: ", len(add_list_model))
                #print("Model discard list: ", len(discard_list_model))
                '''

                # uniform selection of emitter - other options is UCB
                emitter = params["emitter_selection"] #np.random.randint(3)
                if emitter == 0: 
                    add_list_model, to_model_evaluate = self.random_model_emitter(to_model_evaluate, pool, params)
                elif emitter == 1:
                    add_list_model, to_model_evaluate = self.optimizing_emitter(to_model_evaluate, pool, params, gen)
                elif emitter == 2: 
                    add_list_model, to_model_evaluate = self.random_walk_emitter(to_model_evaluate, pool, params, gen)
                elif emitter == 3: 
                    add_list_model, to_model_evaluate = self.model_disagr_emitter(to_model_evaluate, pool, params, gen)

                    
                ### REAL EVALUATIONS ###
                if params['transfer_selection'] == 'disagr':
                    ## Sort by mean disagr on all states
                    sorted_by_disagr = sorted(add_list_model,
                                              key=lambda x: np.mean(np.array(x.model_dis)))

                    ## Select params['nb_transfer'] indiviuals for transfer
                    add_list_model = sorted_by_disagr[:params['nb_transfer']]
                elif params['transfer_selection'] == 'disagr_bd':
                    ## Sort by mean disagr on bd states only
                    if params['env_name'] == 'ball_in_cup':
                        sorted_by_disagr_bd = sorted(add_list_model,
                                                     key=lambda x:
                                                     np.mean(np.array(x.model_dis)[:,0,:3]))
                    if params['env_name'] == 'fastsim_maze':
                        sorted_by_disagr_bd = sorted(add_list_model,
                                                     key=lambda x:
                                                     np.mean(np.array(x.model_dis)[:,0,:2]))
                    if params['env_name'] == 'fastsim_maze_traps':
                        sorted_by_disagr_bd = sorted(add_list_model,
                                                     key=lambda x:
                                                     np.mean(np.array(x.model_dis)[:,0,:2]))
                    if params['env_name'] == 'redundant_arm_no_walls_limited_angles':
                        sorted_by_disagr_bd = sorted(add_list_model,
                                                     key=lambda x:
                                                     np.mean(np.array(x.model_dis)[:,0,-2:]))

                    ## Select params['nb_transfer'] indiviuals for transfer
                    add_list_model = sorted_by_disagr_bd[:params['nb_transfer']]
                elif params['transfer_selection'] == 'random':
                    ## Randomly shuffle the list
                    shuffled_list = random.sample(add_list_model, len(add_list_model))
                    ## Select params['nb_transfer'] indiviuals for transfer
                    add_list_model = shuffled_list[:params['nb_transfer']]
                    
                # if model finds novel solutions - evalute in real setting
                if len(add_list_model) > 0:
                    start = time.time()
                    to_evaluate = []
                    for z in add_list_model: 
                        to_evaluate += [(z.x, self.f_real)]
                    ## real sys eval, we force parallel
                    params['parallel'] =  True
                    s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, params)
                    params['parallel'] =  False
                    self.archive, add_list, discard_list = self.addition_condition(s_list, self.archive, params)
                    ## Create Species pairs for model eval and real eval
                    # nb: useful only if order is not preserved by parallel eval
                    s_pairs = []
                    for s in s_list: # s_list contains real evaluations
                        for s_m in add_list_model: # s_m contains model evals
                            if (s.x == s_m.x).all():
                                s_pairs.append((s, s_m))
                    ## Compute descriptor estimation errors
                    all_errors = []
                    add_errors = []
                    discard_errors = []
                    for s_pair in s_pairs:
                        s = s_pair[0]
                        s_m = s_pair[1]
                        error = np.linalg.norm(s.desc-s_m.desc)
                        all_errors.append(error)
                        if s in add_list: ## Handle Species that were added
                            add_errors.append(error)
                        elif s in discard_list: ## Handle discarded Species
                            discard_errors.append(error)
                        else:
                            print('WARNING: Specy neither in added or discarded list')

                    all_errors_medians.append(np.median(all_errors))
                    if np.isnan(np.median(all_errors)):
                        all_errors_1q.append(np.nan)
                        all_errors_3q.append(np.nan)
                    else:
                        all_errors_1q.append(np.quantile(all_errors, 1/4))
                        all_errors_3q.append(np.quantile(all_errors, 3/4))
                    add_errors_medians.append(np.median(add_errors))
                    if np.isnan(np.median(add_errors)):
                        add_errors_1q.append(np.nan)
                        add_errors_3q.append(np.nan)
                    else:
                        add_errors_1q.append(np.quantile(add_errors, 1/4))
                        add_errors_3q.append(np.quantile(add_errors, 3/4))
                    discard_errors_medians.append(np.median(discard_errors))
                    if np.isnan(np.median(discard_errors)):
                        discard_errors_1q.append(np.nan)
                        discard_errors_3q.append(np.nan)
                    else:
                        discard_errors_1q.append(np.quantile(discard_errors, 1/4))
                        discard_errors_3q.append(np.quantile(discard_errors, 3/4))
                    true_pos = len(add_list)
                    false_pos = len(discard_list)
                    self.eval_time = time.time()-start
                    print("True positives - solutions added into real archive: ", true_pos)
                    print("False positives: ", false_pos)

                ## FOR STATISTICS - EVALUATE MODEL DISCARD PILE ##
                if params["log_model_stats"]:
                    if len(discard_list_model) > 0:
                        to_evaluate_stats = []
                        for z in discard_list_model: 
                            to_evaluate_stats += [(z.x, self.f_real)]
                        s_list_stats = cm.parallel_eval(evaluate_, to_evaluate_stats, pool,params)
                        tmp_archive, add_list_stats, discard_list_stats = self.addition_condition(s_list_stats, tmp_archive, params)
                        false_neg = len(add_list_stats)
                        true_neg = len(discard_list_stats)
                        #print("False negative: ", false_neg)
                        #print("True negative: ", true_neg)
                    
                #print("Real archive size end: ",len(self.archive))
                #print("Model archive size end: ",len(self.model_archive))
                
            # print("Gen: ", gen)
            ####### UPDATE MODEL - MODEL LEARNING ############
            evals_since_last_train += len(to_evaluate)
            self.add_sa_to_buffer(s_list, self.replay_buffer)
            #print("Replay buffer size: ", self.replay_buffer._size)
            
            if (((gen%params["train_freq"]) == 0)or(evals_since_last_train>params["evals_per_train"])) and params["train_model_on"]: 
                # s_list are solutions that have been evaluated in the real setting
                print("Training model")
                start = time.time()
                if params["model_variant"]=="dynamics" or params["model_variant"]=="all_dynamics":
                    # FOR DYNAMICS MODEL
                    # torch.set_num_threads(24)
                    self.dynamics_model_trainer.train_from_buffer(self.replay_buffer, 
                                                                  holdout_pct=0.1,
                                                                  max_grad_steps=100000,
                                                                  verbose=True)
                    
                elif params["model_variant"]=="direct":
                    # FOR DIRECT QD SURROGATE MODEL 
                    # track all real evals - keep only always 50k of latest data
                    # RAM problems if too many is kept
                    #self.all_real_evals = s_list + self.all_real_evals 
                    #if len(self.all_real_evals) > 50000:
                    #    self.all_real_evals = self.all_real_evals[:50000]
                    #dataset = self.get_gen_bd_fit_dataset(self.all_real_evals)
                    dataset = self.get_gen_bd_fit_dataset(self.archive)
                    self.model_trainer.train_from_dataset(dataset,
                                                          holdout_pct=0.2,
                                                          max_grad_steps=10000)

                self.model_train_time = time.time() - start
                print("Model train time: ", self.model_train_time)
                evals_since_last_train = 0

            # count evals
            gen += 1 # generations
            n_evals += len(to_evaluate) # total number of  real evals
            b_evals += len(to_evaluate) # number of evals since last dump
            n_model_evals += len(to_model_evaluate) # total number of model evals

            bds_at_gen = []
            for ind in self.archive:
                # bds_at_gen.append(ind.desc)
                bds_at_gen.append(ind.desc)
            # bds_per_gen.append(bds_at_gen)
            bds_per_gen[f'bd_{gen}'] = bds_at_gen
            # write archive during dump period
            if b_evals >= params['dump_period'] and params['dump_period'] != -1 \
               and params['dump_mode'] == 'budget':
                save_start = time.time()
            
                # write archive
                #print("[{}/{}]".format(n_evals, int(max_evals)), end=" ", flush=True)
                print("[{}/{}]".format(n_evals, int(max_evals)))
                cm.save_archive(self.archive, n_evals, params, self.log_dir)
                ## Also save model archive for more visualizations
                cm.save_archive(self.model_archive, f"{n_evals}_model", params, self.log_dir)
                b_evals = 0
                
                # Save models
                #ptu.save_model(self.model, self.save_model_path)
                print("Saving torch model")
                ptu.save_model(self.dynamics_model, self.save_model_path)
                print("Done saving torch model")

                # print("Saving median,1q,3q of descriptor estimation errors")
                # dump_path = os.path.join(self.log_dir, 'desc_estimation_errors.npz')
                # np.savez(dump_path,
                #          all_errors_medians, all_errors_1q, all_errors_3q,
                #          add_errors_medians, add_errors_1q, add_errors_3q,
                #          discard_errors_medians, discard_errors_1q, discard_errors_3q)
                # print("Done saving descriptor estimation errors")
                
                save_end = time.time() - save_start
                print("Save archive and model time: ", save_end)
            elif params['dump_mode'] == 'gen':
                save_start = time.time()
            
                # write archive
                #print("[{}/{}]".format(n_evals, int(max_evals)), end=" ", flush=True)
                print("[{}/{}]".format(n_evals, int(max_evals)))
                cm.save_archive(self.archive, n_evals, params, self.log_dir)
                ## Also save model archive for more visualizations
                cm.save_archive(self.model_archive, f"{n_evals}_model", params, self.log_dir)
                b_evals = 0
                # plot cov
                ## Extract real sys BD data from s_list
                has_model_data = False
                if isinstance(self.archive, dict):
                    real_bd_traj_data = [s.obs_traj for s in self.archive.values()]
                    if len(self.model_archive.values()) > 0:
                        model_bd_traj_data = [s.obs_traj for s in self.model_archive.values()]
                        has_model_data = True
                else:
                    real_bd_traj_data = [s.obs_traj for s in self.archive]
                    if len(self.model_archive) > 0:
                        model_bd_traj_data = [s.obs_traj for s in self.model_archive]
                        has_model_data = True
                ## Format the bd data to plot with labels
                all_bd_traj_data = []
                all_bd_traj_data.append((real_bd_traj_data, 'real system'))
                if has_model_data:
                    all_bd_traj_data.append((model_bd_traj_data, 'model'))

                params['plot_functor'](all_bd_traj_data, params['args'], self.o_params)
                
                # Save models
                #ptu.save_model(self.model, self.save_model_path)
                print("Saving torch model")
                ptu.save_model(self.dynamics_model, self.save_model_path)
                print("Done saving torch model")

                save_end = time.time() - save_start
                print("Save archive and model time: ", save_end)
                
            # write log -  write log every generation 
            if (self.qd_type=="cvt") or (self.qd_type=="grid"):
                fit_list = np.array([x.fitness for x in self.archive.values()])
                self.log_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(gen,
                                         n_evals,
                                         n_model_evals, 
                                         len(self.archive.keys()),
                                         fit_list.max(),
                                         np.sum(fit_list),
                                         np.mean(fit_list),
                                         np.median(fit_list),
                                         np.percentile(fit_list, 5),
                                         np.percentile(fit_list, 95)))

            elif (self.qd_type=="unstructured"):
                fit_list = np.array([x.fitness for x in self.archive])
                self.log_file.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                    gen,
                    n_evals,
                    n_model_evals,
                    len(self.archive),
                    fit_list.max(),
                    np.sum(fit_list),
                    np.mean(fit_list),
                    np.median(fit_list),
                    np.percentile(fit_list, 5),
                    np.percentile(fit_list, 95),
                    true_pos,
                    false_pos,
                    false_neg,
                    true_neg))
                
            self.log_file.flush() # writes to file but does not close stream

            self.gen_time = time.time() - gen_start_time 
            if params['log_time_stats']:
                self.time_log_file.write("{} {} {} {} {} {} {}\n".format(gen,
                                                                   self.gen_time,
                                                                   self.model_eval_time,
                                                                   self.eval_time,
                                                                   self.model_train_time,
                                                                   len(to_evaluate),
                                                                   len(to_model_evaluate),))
                self.time_log_file.flush()

            print(f"n_evals: {n_evals}, archive_size: {len(self.archive)}, eval time: {self.gen_time}")
                
        print("==========================================")
        print("End of QD algorithm - saving final archive")        
        print("Saving final real archive")
        cm.save_archive(self.archive, n_evals, params, self.log_dir)
        print("Done saving final real archive")
        # Save models
        print("Saving torch model")
        ptu.save_model(self.dynamics_model, self.save_model_path)
        print("Done saving torch model")
        print("Saving median,1q,3q of descriptor estimation errors")

        dump_path = os.path.join(self.log_dir, 'desc_estimation_errors.npz')
        np.savez(dump_path,
                 all_errors_medians=all_errors_medians,
                 all_errors_1q=all_errors_1q,
                 all_errors_3q=all_errors_3q,
                 add_errors_medians=add_errors_medians,
                 add_errors_1q=add_errors_1q,
                 add_errors_3q=add_errors_3q,
                 discard_errors_medians=discard_errors_medians,
                 discard_errors_1q=discard_errors_1q,
                 discard_errors_3q=discard_errors_3q)
        print("Done saving descriptor estimation errors")

        print("Saving behavior descriptors per generation")
        dump_path = os.path.join(self.log_dir, 'bds_per_gen.npz')
        # np.savez(dump_path,
        #          bds_per_gen=bds_per_gen)
        np.savez(dump_path,
                 **bds_per_gen)
        print("Done saving behavior descriptors per generation")
        
        return self.archive, n_evals

    

    ##################### Emitters ##############################
    def random_model_emitter(self, to_model_evaluate, pool, params):
        start = time.time()
        add_list_model_final = []
        all_model_eval = []
        gen = 0
        while len(add_list_model_final) < params['min_found_model']:
        #for i in range(5000): # 600 generations (500 gens = 100,000 evals)
            to_model_evaluate=[]

            if (len(self.model_archive) <= params['random_init']*self.n_niches) \
               and params['init_method'] != 'vanilla':
                to_model_evaluate = self.random_archive_init_model(to_model_evaluate)
                if len(self.model_archive) > 0:
                    to_model_evaluate = random.sample(to_model_evaluate,
                                                      int(params['random_init']*self.n_niches -
                                                          len(self.model_archive)))


                    ## Create fake archive with new species created to have an archive size of 100
                    fake_archive = self.model_archive.copy()
                    for n in range(int(params['random_init']*self.n_niches -
                                                          len(self.model_archive))):
                        s = cm.Species(to_model_evaluate[n][0], [], [])
                        fake_archive.append(s)

                    to_model_evaluate = self.select_and_mutate(to_model_evaluate,
                                                               fake_archive,
                                                               self.f_model, params)
            else:
                to_model_evaluate = self.select_and_mutate(to_model_evaluate, self.model_archive,
                                                           self.f_model, params)
            if params["model_variant"]=="dynamics" or params["perfect_model_on"]:
                #s_list_model = cm.parallel_eval(evaluate_, to_model_evaluate, pool, params)
                print("Starting parallel evaluation of individuals")
                s_list_model = cm.parallel_eval(model_evaluate_, to_model_evaluate, pool, params)
                print("Finished parallel evaluation of individuals")
            elif params["model_variant"]=="direct":
                s_list_model = self.serial_eval(evaluate_, to_model_evaluate, params)
            elif params["model_variant"]=="all_dynamics":
                s_list_model = model_evaluate_all_(to_model_evaluate)
            #self.model_archive, add_list_model, discard_list_model = self.model_condition(s_list_model, self.model_archive, params)
            self.model_archive, add_list_model, discard_list_model = self.addition_condition(s_list_model, self.model_archive, params)

            add_list_model_final += add_list_model
            all_model_eval += to_model_evaluate # count all inds evaluated by model
            #print("to model eval length: ",len(to_model_evaluate)) 
            #print("s list length: ",len(s_list_model)) 
            #print("model list length: ",len(add_list_model_final)) 
            #print("all model evals length: ", len(all_model_eval))

            #if i%20 ==0: 
            #    cm.save_archive(self.model_archive, "model_gen_"+str(i), params, self.log_dir)
            #    print("Model gen: ", i)
            #    print("Model archive size: ", len(self.model_archive))
            print(f'Individuals evaluated on model: {len(s_list_model)}\nCurrent valid population at gen {gen}: {len(add_list_model_final)}')
            gen += 1
        self.model_eval_time = time.time() - start
        print(f"Random model emitter ended in {self.model_eval_time} after {gen} gen")
        return add_list_model_final, all_model_eval

    def optimizing_emitter(self, to_model_evaluate, pool, params, gen):
        '''
        uses CMA - no mutations
        '''
        start = time.time()
        add_list_model_final = []
        all_model_eval = []

        rand1 = np.random.randint(len(self.model_archive))
        mean_init = (self.model_archive[rand1]).x
        sigma_init = 0.01
        popsize = 50
        max_iterations = 100
        es = cma.CMAEvolutionStrategy(mean_init,
                                      sigma_init,
                                      {'popsize': popsize,
                                       'bounds': [0,1]})
        gen = 0
        for i in range(max_iterations):
        #while not es.stop():    
            to_model_evaluate = []
            solutions = es.ask()
            for sol in solutions:
                to_model_evaluate += [(sol, self.f_model)]
            s_list_model = cm.parallel_eval(model_evaluate_, to_model_evaluate, pool, params)

            self.model_archive, add_list_model, discard_list_model = self.model_condition(s_list_model, self.model_archive, params)
            add_list_model_final += add_list_model
            all_model_eval += to_model_evaluate # count all inds evaluated by model
            #print("model list length: ",len(add_list_model_final)) 
            #print("all model evals length: ", len(all_model_eval))

            # convert maximize to minimize
            # for optimizing emitter fitness of CMAES is fitness of the ind
            reward_list = []
            for s in s_list_model:
                reward_list.append(s.fitness)

            cost_arr = -np.array(reward_list)
            es.tell(solutions, list(cost_arr))
            #es.disp()
            
            #if i%10==0:
            #    cm.save_archive(self.model_archive, str(gen)+"_"+str(i), params, self.log_dir)
            #i +=1
            print(f'Current valid population at gen {gen}: {len(add_list_model_final)}')
            gen += 1
                        
        self.model_eval_time = time.time() - start
        print(f"Optimizing model emitter ended in {self.model_eval_time} after {gen} gen")        
        return add_list_model_final, all_model_eval

    def random_walk_emitter(self, to_model_evaluate, pool, params, gen):
        start = time.time()
        add_list_model_final = []
        all_model_eval = []

        # sample an inidivudal from the archive to init cmaes
        rand1 = np.random.randint(len(self.model_archive))
        ind_init = self.model_archive[rand1]
        mean_init = ind_init.x
        sigma_init = 0.01
        popsize = 50
        max_iterations = 100
        es = cma.CMAEvolutionStrategy(mean_init,
                                      sigma_init,
                                      {'popsize': popsize,
                                       'bounds': [0,1]})

        # sample random vector/direction in the BD space to compute CMAES fitness on
        # BD space is 2 dim
        desc_init = ind_init.desc
        target_dir = np.random.uniform(-1,1,size=2)

        gen = 0
        for i in range(max_iterations):
        #i = 0 
        #while not es.stop():
            to_model_evaluate = []
            solutions = es.ask()
            for sol in solutions:
                to_model_evaluate += [(sol, self.f_model)]
            print("Starting parallel evaluation of individuals")
            s_list_model = cm.parallel_eval(model_evaluate_, to_model_evaluate, pool, params)
            print("Finished parallel evaluation of individuals")
            #s_list_model = self.serial_eval(model_evaluate_, to_model_evaluate, params)

            #self.model_archive, add_list_model, discard_list_model = self.model_condition(s_list_model, self.model_archive, params)
            self.model_archive, add_list_model, discard_list_model = self.addition_condition(s_list_model, self.model_archive, params)
            add_list_model_final += add_list_model
            all_model_eval += to_model_evaluate # count all inds evaluated by model
            #print("model list length: ",len(add_list_model_final)) 
            #print("all model evals length: ", len(all_model_eval))

            # convert maximize to minimize
            # for random walk emitter, fitnes of CMAES is the magnitude of vector in the target_direction
            reward_list = []
            for s in s_list_model:
                s_dir = s.desc - desc_init
                comp_proj = (np.dot(s_dir, target_dir))/np.linalg.norm(target_dir)
                reward_list.append(comp_proj)

            cost_arr = -np.array(reward_list)
            es.tell(solutions, list(cost_arr))
            #es.disp()

            if i%10==0:
                cm.save_archive(self.model_archive, str(gen)+"_"+str(i), params, self.log_dir)
            #i +=1
            
            self.model_eval_time = time.time() - start
            #print("model_eval_time", self.model_eval_time)
            print(f'Current valid population at gen {gen}: {len(add_list_model_final)}')
            gen += 1

        self.model_eval_time = time.time() - start
        print(f"Random walk model emitter ended in {self.model_eval_time} after {gen} gen") 
        return add_list_model_final, all_model_eval

    def improvement_emitter():
    
        return 1
    
    def model_disagr_emitter(self, to_model_evaluate, pool, params, gen):
        '''
        emitter which maximises model dissagreement
        CMAES fitness function is disagreement
        '''
        start = time.time()
        add_list_model_final = []
        all_model_eval = []

        # sample an inidivudal from the archive to init cmaes
        rand1 = np.random.randint(len(self.model_archive))
        ind_init = self.model_archive[rand1]
        mean_init = ind_init.x
        sigma_init = 0.01
        popsize = 50
        max_iterations = 100
        es = cma.CMAEvolutionStrategy(mean_init,
                                      sigma_init,
                                      {'popsize': popsize,
                                       'bounds': [0,1]})
        gen = 0
        for i in range(max_iterations):
        #i = 0 
        #while not es.stop():
            to_model_evaluate = []
            solutions = es.ask()
            for sol in solutions:
                to_model_evaluate += [(sol, self.f_model)]
            s_list_model = cm.parallel_eval(model_evaluate_, to_model_evaluate, pool, params)
            #s_list_model = self.serial_eval(model_evaluate_, to_model_evaluate, params)

            #self.model_archive, add_list_model, discard_list_model = self.model_condition(s_list_model, self.model_archive, params)
            self.model_archive, add_list_model, discard_list_model = self.addition_condition(s_list_model, self.model_archive, params)
            add_list_model_final += add_list_model
            all_model_eval += to_model_evaluate # count all inds evaluated by model
            #print("model list length: ",len(add_list_model_final)) 
            #print("all model evals length: ", len(all_model_eval))

            # convert maximize to minimize
            # for disagr_emitter, fitness of CMA-ES is model disagreement
            reward_list = []
            for s in s_list_model:
                s_dis = s.model_dis
                reward_list.append(s_dis)

            cost_arr = -np.array(reward_list)
            es.tell(solutions, list(cost_arr))
            es.disp()

            if i%10==0:
                cm.save_archive(self.model_archive, str(gen)+"_"+str(i), params, self.log_dir)
            #i +=1
            print(f'Current valid population at gen {gen}: {len(add_list_model_final)}')
            gen += 1

        self.model_eval_time = time.time() - start    
        print(f"Model disagr emitter ended in {self.model_eval_time} after {gen} gen")        
        return add_list_model_final, all_model_eval

    
    ################## Custom functions for Model Based QD ####################
    def serial_eval(self, evaluate_function, to_evaluate, params):
        s_list = map(evaluate_function, to_evaluate)
        return list(s_list)
    
    def evaluate_solution_surrogate_model(self, gen):
        #torch.set_num_threads(1)
        with torch.no_grad():
            x = ptu.from_numpy(gen)
            x = x.view(1, -1)
        
            pred = self.model.output_pred(x)            
            fitness = pred[0,0]
            desc = [pred[0,1:1+self.dim_map]]
            
        obs_traj = None
        act_traj = None
        
        return fitness, desc, obs_traj, act_traj
    
    def evaluate_batch(self, z_batch):
        '''
        # For batch evaluations of NN models
        #to_model_evaluate = self.select_and_mutate(to_model_evaluate, self.model_archive, self.f_model, params, batch=True)
        #s_list_model = self.evaluate_batch(to_model_evaluate)
        '''
        f = self.f_model
        fit_batch, desc_batch, obs_traj_b, act_traj_b = f(z_batch) 
        # this is a batch eval functions so returns everything as [batch_size, dim]
        s_list = []
        for i in range(len(z_batch)):
            z = z_batch[i]
            desc = desc_batch[i] 
            desc_ground = desc
            fit = fit_batch[i]
            obs_traj = obs_traj_b[i]
            act_traj = act_traj_b[i]
            
            s = cm.Species(z, desc, desc_ground, fit, obs_traj, act_traj)
            s_list.append(s) 
        
        return s_list
    
    def get_gen_bd_fit_dataset(self, s_list):
        x = []
        y = []
        for s in s_list:
            x.append(s.x)
            y.append(np.concatenate((s.fitness, s.desc), axis=None))

        x = np.array(x)
        y = np.array(y)
        dataset = [x,y]

        return dataset
    
    def add_sa_to_buffer(self, s_list, replay_buffer):
        for sol in s_list:
            s = sol.obs_traj[:-1]  
            a = sol.act_traj[:]
            ns = sol.obs_traj[1:]

            reward = 0
            done = 0
            info = {}
            for i in range(len(s)):
                replay_buffer.add_sample(s[i], a[i], reward, done, ns[i], info)
        return 1
