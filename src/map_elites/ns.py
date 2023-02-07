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
import math
import numpy as np
import multiprocessing
from multiprocessing import get_context
import tqdm


# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

from src.map_elites import common as cm
from src.map_elites import unstructured_container, cvt
from src.map_elites import model_condition_utils

import torch
import src.torch.pytorch_util as ptu

import cma

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

def training_and_target_evaluate_all_(to_evaluate, params):
    Z = [to_evaluate[i][0] for i in range(len(to_evaluate))]
    ## Evaluate on training system(s)
    f = params['f_training']
    fit_list, desc_list, obs_traj_list, act_traj_list, disagr_list = f(Z) 

    ## Evaluate on target system
    f = params['f_target']

    all_obs_traj_list = []
    all_act_traj_list = []
    all_disagr_list = []
    for i in range(len(Z)):
        fit, desc, obs_traj, act_traj, disagr = f(Z[i]) 
        # fit_list[i] += fit
        desc_list[i] = np.concatenate((desc_list[i], desc))
        if params['log_ind_trajs'] == True:
            m_h = params['dab_params']['dynamics_model_params']['model_horizon']
            m_h += 1
            state_dim = params['dab_params']['state_dim']
            action_dim = params['dab_params']['action_dim']
            obs_traj_loc = np.empty((m_h, 1, state_dim))
            obs_traj_loc[:,0,:2] = obs_traj[-m_h:]
            act_traj_loc = np.empty((m_h-1, 1, action_dim))
            act_traj_loc[:,0,:2] = act_traj[-m_h+1:]
            all_obs_traj_list.append(np.concatenate((obs_traj_list[i], obs_traj_loc), axis=1))
            all_act_traj_list.append(np.concatenate((act_traj_list[i], act_traj_loc), axis=1))
            all_disagr_list.append([])
    all_obs_traj_list = np.array(all_obs_traj_list)
    all_act_traj_list = np.array(all_obs_traj_list)
    all_disagr_list = np.array(all_disagr_list)
    # return a list of species object (containing genotype, descriptor and fitness)
    inds = []
    for i in range(len(Z)):
        inds.append(cm.Species(Z[i], desc_list[i], fit_list[i], obs_traj=all_obs_traj_list[i],
                               act_traj=all_act_traj_list[i], model_dis=all_disagr_list[i]))
    return inds


class NS:
    def __init__(self,
                 dim_map, dim_x,
                 f_real,
                 params=cm.default_params,
                 log_dir='./',):

        #torch.set_num_threads(24)
        
        self.qd_type = params["type"]    # QD type - grid, cvt, unstructured
        self.dim_map = dim_map           # number of BD dimensions  
        self.dim_x = dim_x               # gemotype size (number of genotype dim)
        self.params = params
        
        # eval functions
        self.f_real = f_real
        
        # Init logging directory and log file
        self.log_dir = log_dir
        log_filename = self.log_dir + '/log_file.dat'
        self.log_file = open(log_filename, 'w')
        
        if params['log_time_stats']:
            time_stats_filename = self.log_dir + '/time_log_file.dat'
            self.time_log_file = open(time_stats_filename, 'w')
            self.gen_time = 0
            self.model_eval_time = 0
            self.eval_time = 0
            self.model_train_time = 0 
        
        self.archive = [] # init archive as list
        self.model_archive = []        
        self.population = None
        
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
            for s in sel_s_list:
                    archive.append(s)
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

    def update_novelty_scores(self, pop, archive, k=15, slow=False):
        # Convert the dataset to a numpy array
        all_bds = []
        all_bds += [ind.desc for ind in pop] # pop is usually pop + offspring
        all_bds += [ind.desc for ind in archive]
        all_bds = np.array(all_bds)
        novelty_scores = np.empty((len(all_bds)))
        # Compute the k-NN of the data point
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(all_bds)

        ## Slow way lol
        if slow:
            for ind in pop:
                k_nearest_neighbors = neighbors.kneighbors([ind.desc],
                                                           return_distance=False,
                                                           n_neighbors=k+1)[0]

                # # Compute the average distance between the data point and its k-NN
                ind.nov = np.mean(np.linalg.norm(all_bds[k_nearest_neighbors] - ind.desc,
                                                 axis=1))
        else:
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

            if norm:
                max_bd = np.max(all_bds, axis=0)
                min_bd = np.min(all_bds, axis=0)
                all_bds = (all_bds - min_bd)/(max_bd - min_bd)
            novelty_scores = np.empty((len(all_bds)))
            # Compute the k-NN of the data point
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors.fit(all_bds)

            ## New way
            neigh_dists, neigh_inds = neighbors.kneighbors()
            for ind, dists in zip(pop, neigh_dists):
                ind_novs[i].append(np.mean(dists))
        ind_novs = np.array(ind_novs)
        # update all individuals nov by minimum of novelty on all environments
        for i in range(len(pop)):
            if nov == 'min':
                pop[i].nov = np.min(ind_novs[:,i])
            elif nov == 'mean':
                pop[i].nov = np.mean(ind_novs[:,i])
            elif nov == 'sum':
                pop[i].nov = sum(ind_novs[:,i])
                
    def compute(self,
                num_cores_set,
                max_evals=1e6,
                params=None,):

        if params is None:
            params = self.params

        # setup the parallel processing pool
        if num_cores_set == 0:
            num_cores = multiprocessing.cpu_count() - 1 # use all cores
        else:
            num_cores = num_cores_set
            
        pool = multiprocessing.Pool(num_cores)

        # if params['bootstrap_archive'] != '':
        #     arch_data = pd.read_csv(params['bootstrap_archive'])
        #     arch_data = arch_data.iloc[:,:-1]
        #     s_list = []
        #     gen_cols = [col for col in arch_data.columns if 'x' in col]
        #     bd_cols = [col for col in arch_data.columns if 'bd' in col]
        #     for index, row in arch_data.iterrows():
        #         genotype = row[gen_cols]
        #         bd = row[bd_cols]
        #         s = cm.Species(genotype, bd, row['fit'])
        #         s_list.append(s)
        #     population = s_list
        #     archive = s_list
        # else:
        #     population = []

        population = []
        
        gen = 0 # generation
        n_evals = 0 # number of evaluations since the beginning
        b_evals = 0 # number evaluation since the last dump

        print("################# Starting NS algorithm #################")

        # main loop
        while (n_evals < max_evals):
            # lists of individuals we want to evaluate (list of tuples) for this gen
            # each entry in the list is a tuple of the genotype and the evaluation function
            to_evaluate = []

            ## intialize for time related stats ##
            gen_start_time = time.time()
            self.model_train_time = 0
            # initialization of archive - start up
            if len(self.archive) == 0:
                if params['bootstrap_archive'] is not None:
                    for ind in params['bootstrap_archive']:
                        to_evaluate += [(ind.x, self.f_real)]
                else:
                    to_evaluate = self.random_archive_init(to_evaluate)
                start = time.time()
                if params["include_target"] == True:
                    s_list = training_and_target_evaluate_all_(to_evaluate, params)  
                elif params["model_variant"]=="all_dynamics":
                    s_list = evaluate_all_(to_evaluate)
                else:
                    s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, params)

                population = s_list
                
                self.eval_time = time.time() - start

                self.archive, add_list, _ = self.addition_condition(population,
                                                                    self.archive,
                                                                    params)
                
            else:
                # variation/selection loop - select ind from population to evolve

                to_evaluate = self.select_and_mutate(to_evaluate, population,
                                                     self.f_real, params)
                if params["include_target"] == True:
                    training_and_target_evaluate_all_(to_evaluate, params)
                elif params["model_variant"]=="all_dynamics":
                    s_list = evaluate_all_(to_evaluate)
                else:
                    s_list = cm.parallel_eval(evaluate_, to_evaluate,
                                              pool, params)

                offspring = s_list

                ## Update population nov (pop + offsprings)
                # if params['model_type'] == 'det_ens':
                if params['model_type'] == 'det_ens' and not params['perfect_model_on']:
                    self.update_novelty_scores_ensemble(population + offspring,
                                                        self.archive,
                                                        nov=params['nov_ens'],
                                                        norm=params['norm_bd'])
                else:
                    self.update_novelty_scores(population + offspring, self.archive)
                ## Add offsprings to archive
                self.archive, add_list, _ = self.addition_condition(offspring,
                                                                    self.archive,
                                                                    params)

                ## Update population
                sorted_pop = sorted(population + offspring,
                                    key=lambda x:x.nov, reverse=True)
                population = sorted_pop[:params['pop_size']]
                self.population = population

            # count evals
            gen += 1 # generations
            n_evals += len(to_evaluate) # total number of  real evals
            b_evals += len(to_evaluate) # number of evals since last dump
            
            # write archive during dump period
            
            if b_evals >= params['dump_period'] and params['dump_period'] != -1:
                # write archive
                print("[{}/{}]".format(n_evals, int(max_evals)), end=" ", flush=True)
                cm.save_archive(self.archive, n_evals, params, self.log_dir)
                b_evals = 0

            # write log -  write log every generation 
            fit_list = np.array([x.fitness for x in self.archive])
            self.log_file.write("{} {} {} {} {} {} {} {} {}\n".format(
                gen,
                n_evals,
                len(self.archive),
                fit_list.max(),
                np.sum(fit_list),
                np.mean(fit_list),
                np.median(fit_list),
                np.percentile(fit_list, 5),
                np.percentile(fit_list, 95),))
                
            self.log_file.flush() # writes to file but does not close stream

            self.gen_time = time.time() - gen_start_time 

            print(f"n_evals: {n_evals}/{max_evals}, archive_size: {len(self.archive)}, eval time: {self.gen_time}")
                
        print("==========================================")
        print("End of NS algorithm - saving final archive")        
        cm.save_archive(self.archive, n_evals, params, self.log_dir)
        pool.close()
        self.log_file.close()
        
        return self.archive, n_evals
