#! /usr/bin/env python
import os, sys
import time
import math, random
import numpy as np
import multiprocessing

from sklearn.neighbors import KDTree

from src.map_elites import common as cm
from src.map_elites import unstructured_container, cvt

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
    ## Check if list of list (happens only with hexapod)
    if hasattr(desc[0], '__len__'):
        desc = desc[0]    
    # return a species object (containing genotype, descriptor and fitness)
    return cm.Species(z, desc, fit, obs_traj=obs_traj, act_traj=act_traj, model_dis=disagr)

def model_evaluate_(t):
    # same as the above evaluate but this takes in the disagreement also
    # - useful if you want to make use disargeement value
    # needs two types because no such thing as disagreemnt for real eval
    z, f = t
    fit, desc, obs_traj, act_traj, disagr = f(z) 
    ## Check if list of list (happens only with hexapod)
    if hasattr(desc[0], '__len__'):
        desc = desc[0]
    # return a species object (containing genotype, descriptor and fitness)
    return cm.Species(z, desc, fit, obs_traj=obs_traj, act_traj=act_traj, model_dis=disagr)

def model_evaluate_all_(T):
    # same as the above evaluate but this takes in the disagreement also
    # - useful if you want to make use disargeement value
    # needs two types because no such thing as disagreemnt for real eval
    Z = [T[i][0] for i in range(len(T))]
    f = T[0][1]
    fit_list, desc_list, obs_traj_list, act_traj_list, disagr_list = f(Z)     
    # return a species object (containing genotype, descriptor and fitness)
    model_inds = []
    for i in range(len(T)):
        model_inds.append(cm.Species(Z[i], desc_list[i], fit_list[i], obs_traj=obs_traj_list[i],
                                     act_traj=act_traj_list[i], model_dis=disagr_list[i]))
    return model_inds

class MultiDynamicsModelQD:
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
                    o_params = params['dab_params']
                    bd_inds = o_params['bd_inds']
                    bd_max = o_params['state_max'][bd_inds]
                    bd_min = o_params['state_min'][bd_inds]
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
                success = cvt.add_to_archive(s, s.desc, self.archive, self.kdt)
            if success:
                add_list.append(s)
            else:
                discard_list.append(s) #not important for alogrithm but to collect stats
                
        return archive, add_list, discard_list

    # Multi Dynamics Model map-elites algorithm
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
                s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, params) #init real archive
                #s_list = cm.parallel_eval(model_evaluate_, to_evaluate, pool, params) #init model
                self.eval_time = time.time() - start 
                self.archive, add_list, _ = self.addition_condition(s_list, self.archive, params)

            else:
                print("Evaluation on model")
                # variation/selection loop - select ind from archive to evolve                
                self.model_archive = self.archive.copy()
                tmp_archive = self.archive.copy() # tmp archive for stats of negatives
                add_list_model, to_model_evaluate = self.random_model_emitter(to_model_evaluate, pool, params)
                    
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
                    s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, params)
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
                    all_errors_1q.append(np.quantile(all_errors, 1/4))
                    all_errors_3q.append(np.quantile(all_errors, 3/4))
                    add_errors_medians.append(np.median(add_errors))
                    add_errors_1q.append(np.quantile(add_errors, 1/4))
                    add_errors_3q.append(np.quantile(add_errors, 3/4))
                    discard_errors_medians.append(np.median(discard_errors))
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
                    
            ####### UPDATE MODELS - MODELS LEARNING ############
            evals_since_last_train += len(to_evaluate)
            self.add_sa_to_buffer(s_list, self.replay_buffer)
            
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
                    
                self.model_train_time = time.time() - start
                print("Model train time: ", self.model_train_time)
                evals_since_last_train = 0

            # count evals
            gen += 1 # generations
            n_evals += len(to_evaluate) # total number of  real evals
            b_evals += len(to_evaluate) # number of evals since last dump
            n_model_evals += len(to_model_evaluate) # total number of model evals
            
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

                print("Saving median,1q,3q of descriptor estimation errors")
                dump_path = os.path.join(self.log_dir, 'desc_estimation_errors.npz')
                np.savez(dump_path,
                         all_errors_medians, all_errors_1q, all_errors_3q,
                         add_errors_medians, add_errors_1q, add_errors_3q,
                         discard_errors_medians, discard_errors_1q, discard_errors_3q)
                print("Done saving descriptor estimation errors")
                
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
            elif params["model_variant"]=="all_dynamics":
                s_list_model = model_evaluate_all_(to_model_evaluate)
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

    ################## Custom functions for Model Based QD ####################
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
