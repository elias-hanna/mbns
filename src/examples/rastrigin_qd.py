'''
Quick validation tests for QD scripts on rastrigin task
'''
import os, sys
import argparse

import numpy as np
import math

# map elites and qd imports
from src.map_elites.qd import QD



parser = argparse.ArgumentParser()
#-----------------Type of QD---------------------#
# options are 'cvt', 'grid' and 'unstructured'
parser.add_argument("--qd_type", type=str, default="unstructured")

#---------------CPU usage-------------------#
parser.add_argument("--num_cores", type=int, default=8)

#-----------Store results + analysis-----------#
parser.add_argument("--log_dir", type=str)

args = parser.parse_args()


px = \
{
    # type of qd 'unstructured, grid, cvt'
    "type": args.qd_type,
    
    # more of this -> higher-quality CVT
    "cvt_samples": 25000,
    # we evaluate in batches to parallelize
    "batch_size": 200,
    # proportion of total number of niches to be filled before starting
    "random_init": 0.005,  
    # batch for random initialization
    "random_init_batch": 100,
    # when to write results (one generation = one batch)
    "dump_period": 1000,
    
    # do we use several cores?
    "parallel": True,
    # do we cache the result of CVT and reuse?
    "cvt_use_cache": False,
    # min/max of genotype parameters - check mutation operators too
    "min": 0.0,
    "max": 1.0,
    
    #------------MUTATION PARAMS---------#
    # selector ["uniform", "random_search"]
    "selector" : "uniform",
    # mutation operator ["iso_dd", "polynomial", "sbx"]
    "mutation" : "iso_dd",
    
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
    "nov_l": 0.01,
    "eps": 0.1, # usually 10%
    "k": 15,  # from novelty search


    "log_time_stats": False, 
}


# define rastrigin evaluation task
def rastrigin(xx):
    x = xx * 10 - 5 # scaling to [-5, 5]
    f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
    desc = np.array([xx[0], xx[1]])
    return -f, [desc], None, None


dim_map = 2
dim_x = 10
n_niches = 10000
qd = QD(dim_map, dim_x, rastrigin, n_niches, params=px, log_dir=args.log_dir)
qd.compute(num_cores_set=args.num_cores, max_evals=1e6)
