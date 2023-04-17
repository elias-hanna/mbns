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
from sklearn.neighbors import NearestNeighbors

from src.map_elites import common as cm
from src.map_elites import unstructured_container, cvt
from src.map_elites import model_condition_utils
from src.map_elites.ns import NS

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

class ModelBasedNS():
    def __init__(self,
                 dim_map, dim_x,
                 f_real, f_model,
                 dynamics_model, dynamics_model_trainer,
                 replay_buffer,
                 params=cm.default_params,
                 log_dir='./',):

        self.qd_type = params["type"]    # QD type - grid, cvt, unstructured
        self.dim_map = dim_map           # number of BD dimensions  
        self.dim_x = dim_x               # gemotype size (number of genotype dim)
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
            
        # Model and Model trainer init -
        # initialize the classes outside this class and pass in
        
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
                
        self.archive = [] # init archive as list
        self.model_archive = []        

    def random_archive_init_model(self, to_evaluate):
        for i in range(0, self.params['random_init_batch']):
            x = np.random.uniform(low=self.params['min'], high=self.params['max'],size=self.dim_x)
            to_evaluate += [(x, self.f_model)]
        
        return to_evaluate

    def random_archive_init(self, to_evaluate):
        for i in range(0, self.params['pop_size']):
            x = np.random.uniform(low=self.params['min'], high=self.params['max'], size=self.dim_x)
            to_evaluate += [(x, self.f_real)]
        
        return to_evaluate


    def select_and_mutate(self, to_evaluate, archive, f, params, variation_operator=cm.variation, batch=False):

        if (self.qd_type=="cvt") or (self.qd_type=="grid"):
            keys = list(archive.keys())
        elif (self.qd_type=="unstructured" or self.qd_type=="fixed"):
            keys = archive
                    
        # we select all the parents at the same time because randint is slow
        rand1 = np.random.randint(len(keys), size=self.params['batch_size'])
        rand2 = np.random.randint(len(keys), size=self.params['batch_size'])
            
        for n in range(0, params['batch_size']):
            # parent selection - mutation operators like iso_dd/sbx require 2 gen parents
            if (self.qd_type == "cvt") or (self.qd_type=="grid"):
                x = archive[keys[rand1[n]]]
                y = archive[keys[rand2[n]]]
            elif (self.qd_type == "unstructured" or self.qd_type == "fixed"):                    
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
        if self.qd_type == "fixed":
            if params['arch_sel'] ==  'random':
                ## Randomly add lambda inds to archive
                sel_s_list = np.random.choice(s_list, size=params['lambda'], replace=False)
            elif params['arch_sel'] == 'nov' or params['arch_sel'] == 'novelty':
                ## Add lambda inds to archive based on novelty
                sorted_s_list = sorted(s_list, key=lambda x:x.nov, reverse=True)
                sel_s_list = sorted_s_list[:params['lambda']]
            discard_list = s_list.copy()
            for s in sel_s_list:
                archive.append(s)
                add_list.append(s)
                discard_list.remove(s)
        else:
            for s in s_list:
                if self.qd_type == "unstructured":
                    success = unstructured_container.add_to_archive(s, archive, params)
                else:
                    success = cvt.add_to_archive(s, s.desc, self.archive, self.kdt)
                if success:
                    add_list.append(s)
                else:
                    discard_list.append(s) #not important for alogrithm but to collect stats

        return archive, add_list, discard_list
    
    def model_condition(self, s_list, archive, params):
        return self.addition_condition(s_list, archive, params)
    
    def update_novelty_scores(self, pop, archive, k=15):
        # Convert the dataset to a numpy array
        all_bds = []
        all_bds += [ind.desc for ind in pop] # pop is usually pop + offspring
        all_bds += [ind.desc for ind in archive]
        all_bds = np.array(all_bds)
        novelty_scores = np.empty((len(all_bds)))
        # Compute the k-NN of the data point
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(all_bds)

        ## New way
        neigh_dists, neigh_inds = neighbors.kneighbors()
        for ind, dists in zip(pop, neigh_dists):
            ind.nov = np.mean(dists)

    # nov: min takes minimum of novelty from all models, mean takes the mean
    def update_novelty_scores_ensemble(self, pop, archive, k=15, nov='sum', norm=False):
        # Get novelty scores on all models of ensemble individually
        ind_novs = []
        ens_size = self.params['ensemble_size']
        if self.params["include_target"]:
            ens_size += 1
        for i in range(ens_size):
            ind_novs.append([])
            # Convert the dataset to a numpy array
            all_bds = []
            all_bds += [ind.desc[i*self.dim_map:i*self.dim_map+self.dim_map]
                        for ind in pop] # pop is usually pop + offspring 

            all_bds += [ind.desc[i*self.dim_map:i*self.dim_map+self.dim_map]
                        for ind in archive]
            all_bds = np.array(all_bds)

            novelty_scores = np.empty((len(all_bds)))
            # Compute the k-NN of the data point
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors.fit(all_bds)

            ## New way
            neigh_dists, neigh_inds = neighbors.kneighbors()
            for idx, dists in zip(range(len(pop)), neigh_dists):
                novelty_scores[idx] = np.mean(dists)
            max_nov = np.max(novelty_scores)
            min_nov = np.min(novelty_scores)
            novelty_scores = (novelty_scores - min_nov)/(max_nov - min_nov)

            for idx in range(len(pop)):
                ind_novs[i].append(novelty_scores[idx])
                
            # if norm:
            #     max_bd = np.max(all_bds, axis=0)
            #     min_bd = np.min(all_bds, axis=0)
            #     all_bds = (all_bds - min_bd)/(max_bd - min_bd)
            # # Compute the k-NN of the data point
            # neighbors = NearestNeighbors(n_neighbors=k)
            # neighbors.fit(all_bds)

            # ## New way
            # neigh_dists, neigh_inds = neighbors.kneighbors()
            # for ind, dists in zip(pop, neigh_dists):
            #     ind_novs[i].append(np.mean(dists))

        ind_novs = np.array(ind_novs)
        # update all individuals nov by minimum of novelty on all environments
        for i in range(len(pop)):
            if nov == 'min':
                pop[i].nov = np.min(ind_novs[:,i])
            elif nov == 'mean':
                pop[i].nov = np.mean(ind_novs[:,i])
            elif nov == 'sum':
                pop[i].nov = sum(ind_novs[:,i])

    def update_population_novelty(self, population, offspring, archive, params):
        ## Update population nov (pop + offsprings)
        ensembling = False
        if 'model_type' in params:
            if 'perfect_model_on' in params:
                if params['perfect_model_on']:
                    ensembling = False
            elif params['model_type'] == 'det':
                ensembling = False
            else:
                ensembling = True
        if ensembling:
            self.update_novelty_scores_ensemble(population + offspring,
                                                archive,
                                                nov=params['nov_ens'],
                                                norm=params['norm_bd'])
        else:
            self.update_novelty_scores(population + offspring, archive)

    def dynamics_model_gpu_mode(self, mode):
        ptu.set_gpu_mode(mode)
        ## Send model params to current device
        ptu.to_current_device(self.dynamics_model)
        ## And layers params to gpu or cpu
        if mode:
            self.dynamics_model.fc0.cuda()
            self.dynamics_model.fc1.cuda()
            self.dynamics_model.last_fc.cuda()
        else:
            self.dynamics_model.fc0.cpu()
            self.dynamics_model.fc1.cpu()
            self.dynamics_model.last_fc.cpu()
    
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
            if len(self.archive) == 0 and params['init_method'] == 'vanilla': 
                print("Evaluation on real environment for initialization")
                to_evaluate = self.random_archive_init(to_evaluate) # init real archive
                start = time.time()
                ## real sys eval, we force parallel, then revert to real params value
                parallel = params['parallel']
                params['parallel'] =  True
                s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, params) #init real archive
                params['parallel'] =  parallel

                population = s_list
                
                self.eval_time = time.time() - start 

                self.archive, add_list, _ = self.addition_condition(population,
                                                                    self.archive,
                                                                    params)

            else:
                print("Evaluation on model")
                # variation/selection loop - select ind from archive to evolve                
                self.model_archive = self.archive.copy()
                model_population = population.copy()
                
                tmp_archive = self.archive.copy() # tmp archive for stats of negatives

                if ptu._use_gpu:
                    ## Switch dynamics model to CPU
                    print("Switched dynamics model to CPU")
                    self.dynamics_model_gpu_mode(False)
                
                # uniform selection of emitter - other options is UCB
                emitter = params["emitter_selection"] #np.random.randint(3)

                if emitter == 0: 
                    model_population, to_model_evaluate = self.novelty_search_emitter(
                        to_model_evaluate,
                        model_population,
                        pool,
                        params)

                    ## Filter out the individuals that remained in population
                    add_list_model = [ind for ind in model_population if ind not in population]
                    
                    
                ### REAL EVALUATIONS ###    
                # if model finds novel solutions - evaluate in real setting
                if len(add_list_model) > 0:
                    start = time.time()
                    to_evaluate = []
                    for z in add_list_model: 
                        to_evaluate += [(z.x, self.f_real)]
                    ## real sys eval, we force parallel, then revert to real params value
                    parallel = params['parallel']
                    params['parallel'] =  True
                    s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, params)
                    params['parallel'] =  parallel

                    offspring = s_list
                    self.update_population_novelty(population,
                                                   offspring,
                                                   self.archive,
                                                   params)
                    
                    self.archive, add_list, discard_list = self.addition_condition(
                        offspring, self.archive, params)

                    ## Update population
                    sorted_pop = sorted(population + offspring,
                                        key=lambda x:x.nov, reverse=True)
                    filtered_s_pop = [ind for ind in sorted_pop if ind not in self.archive]
                    population = filtered_s_pop[:params['pop_size']]
                    self.population = population
                
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
                    
            # print("Gen: ", gen)
            ####### UPDATE MODEL - MODEL LEARNING ############
            evals_since_last_train += len(to_evaluate)
            self.add_sa_to_buffer(s_list, self.replay_buffer)
            #print("Replay buffer size: ", self.replay_buffer._size)
            
            if (((gen%params["train_freq"]) == 0)or(evals_since_last_train>params["evals_per_train"])) and params["train_model_on"]:
                
                if torch.cuda.is_available():
                    if not ptu._use_gpu:
                        ## Switch dynamics model to GPU
                        print("Switched dynamics model to GPU")
                        self.dynamics_model_gpu_mode(True)
                    print("Training model on GPU")
                else:
                    print("Training model on CPU")
                # s_list are solutions that have been evaluated in the real setting
                start = time.time()
                if params["model_variant"]=="dynamics" or params["model_variant"]=="all_dynamics":
                    # FOR DYNAMICS MODEL
                    # torch.set_num_threads(24)
                    self.dynamics_model_trainer.train_from_buffer(self.replay_buffer, 
                                                                  holdout_pct=0.1,
                                                                  max_grad_steps=100000,
                                                                  verbose=True)

                self.model_train_time = time.time() - start
                print("Model train time: ", self.model_train_time)
                evals_since_last_train = 0

            # count evals
            gen += 1 # generations
            n_evals += len(to_evaluate) # total number of  real evals
            b_evals += len(to_evaluate) # number of evals since last dump
            n_model_evals += len(to_model_evaluate) # total number of model evals

            bds_per_gen[f'bd_{gen}'] = [ind.desc for ind in self.archive]

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
                        # model_bd_traj_data = [s.obs_traj for s in self.model_archive.values()]
                        model_bd_traj_data = [s.obs_traj for s in add_list_model]
                        has_model_data = True
                else:
                    real_bd_traj_data = [s.obs_traj for s in self.archive]
                    if len(self.model_archive) > 0:
                        # model_bd_traj_data = [s.obs_traj for s in self.model_archive]
                        if self.params['env_name'] == 'hexapod_omni':
                            model_bd_traj_data = [s.obs_traj[:,0,:] for s in add_list_model]
                        else:
                            model_bd_traj_data = [s.obs_traj for s in add_list_model]
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
        np.savez(dump_path, **bds_per_gen)
        print("Done saving behavior descriptors per generation")
        
        return self.archive, n_evals

    

    ##################### Emitters ##############################
    def novelty_search_emitter(self, to_model_evaluate, model_population,
                               pool, params):
        start = time.time()
        add_list_model_final = []
        all_model_eval = []
        gen = 0
        # while len(add_list_model_final) < params['min_found_model']:
        #for i in range(5000): # 600 generations (500 gens = 100,000 evals)
        for model_gen in range(params['model_budget_gen']):
            to_model_evaluate=[]

            to_model_evaluate = self.select_and_mutate(to_model_evaluate,
                                                       model_population,
                                                       self.f_model,
                                                       params)
            if params["model_variant"]=="dynamics" or params["perfect_model_on"]:
                print("Starting parallel evaluation of individuals")
                s_list_model = cm.parallel_eval(model_evaluate_, to_model_evaluate, pool, params)
            elif params["model_variant"]=="all_dynamics":
                print("Starting batch evaluation of individuals")
                s_list_model = model_evaluate_all_(to_model_evaluate)

            model_offspring = s_list_model

            ## Update population nov (pop + offsprings)
            self.update_population_novelty(model_population,
                                           model_offspring,
                                           self.model_archive,
                                           params)

            self.model_archive, add_list_model, discard_list_model = self.addition_condition(model_offspring, self.model_archive, params)

            ## Update population
            model_sorted_pop = sorted(model_population + model_offspring,
                                      key=lambda x:x.nov, reverse=True)
            # filtered_s_model_pop = [ind for ind in model_sorted_pop if ind not in self.model_archive]
            filtered_s_model_pop = model_sorted_pop
            
            model_population = filtered_s_model_pop[:params['pop_size']]

            ### Need to think of what individuals to select for transfer
            add_list_model_final += add_list_model
            all_model_eval += to_model_evaluate # count all inds evaluated by model
            print(f'Individuals evaluated on model: {len(s_list_model)}\nCurrent valid population at gen {gen}: {len(add_list_model_final)}')
            gen += 1

        self.model_eval_time = time.time() - start
        print(f"Novelty Search emitter ended in {self.model_eval_time} after {gen} gen")

        if params['model_ns_return'] == 'archive':
            return self.model_archive, all_model_eval
        elif params['model_ns_return'] == 'population':
            return model_population, all_model_eval
        # return add_list_model_final, all_model_eval
        # return model_population, all_model_eval

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
