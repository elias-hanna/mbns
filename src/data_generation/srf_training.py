import gstools as gs
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from itertools import repeat

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def normalize_inputs_sa_minmax(data, sa_min, sa_max):
    data_norm = (data - sa_min)/(sa_max - sa_min)
    rescaled_data_norm = data_norm * (1 + 1) - 1
    return rescaled_data_norm

def denormalize_outputs_sa_minmax(data, sa_min, sa_max, obs_dim):
    data_denorm = ((data + 1)*(sa_max[:obs_dim] - sa_min[:obs_dim]) / \
                   (1 + 1)) + sa_min[:obs_dim]
    return data_denorm

def query_srfs(params, input_data, srfs):
    ret_data = []
    for srf in srfs:
        ret_data.append(srf(input_data))
    ret_data = np.array(ret_data)
    ret_data = np.transpose(ret_data)
    # print(ret_data.shape)
    # print(input_data.shape)
    # print(input_data[:,:params['obs_dim']].shape)
    print(np.max(ret_data), np.min(ret_data))
    ret_data += input_data[:,:params['obs_dim']]
    ret_data = np.clip(ret_data, -1, 1)
    # return np.transpose(ret_data)
    return ret_data

def get_training_samples(params, n_training_samples):
    dim_in = params['obs_dim'] + params['action_dim']
    dim_out = params['obs_dim']
    sa_min = np.concatenate((params['state_min'], params['action_min']))
    sa_max = np.concatenate((params['state_max'], params['action_max']))
    
    ## Create a srfs model
    model = gs.Gaussian(dim=dim_in, var=params['srf_var'],
                        len_scale=params['srf_cor'])
    srfs = []
    for _ in range(dim_out):
        srfs.append(gs.SRF(model))
    
    ## Create fake input data
    fake_input_data = np.random.uniform(low=sa_min, high=sa_max,
                                        size=(n_training_samples, dim_in))
    ## Normalize data
    norm_input = normalize_inputs_sa_minmax(fake_input_data, sa_min, sa_max)
    ## Query the model using fake input data
    norm_output = query_srfs(params, norm_input, srfs)
    ## Denormalize data
    fake_output_data = denormalize_outputs_sa_minmax(norm_output, sa_min,
                                                     sa_max, params['obs_dim'])    
    # ForkedPdb().set_trace()
    ## Return the fake input and output training data
    return fake_input_data, fake_output_data

def get_ensemble_training_samples(params,
                                  n_training_samples=10000, ensemble_size=10):
    
    with Pool(processes=params['num_cores']) as pool:
        to_evaluate = zip(repeat(params,ensemble_size),
                          repeat(n_training_samples))
        input_data, output_data = zip(*pool.starmap(
            get_training_samples,
            to_evaluate
        ))
    return np.array(input_data), np.array(output_data)


if __name__ == '__main__':

    act_dim = 2
    obs_dim = 6
    dim_in = act_dim + obs_dim
    dim_out = obs_dim
    
    a_min = np.array([-1, -1])
    a_max = np.array([1, 1])
    ss_min = np.array([0, 0, -1, -1, -1, -1])
    ss_max = np.array([600, 600, 1, 1, 1, 1])

    sa_min = np.concatenate((ss_min, a_min))
    sa_max = np.concatenate((ss_max, a_max))

    params = \
    {
        'obs_dim': obs_dim,
        'action_dim': act_dim,

        'action_min': a_min,
        'action_max': a_max,

        'state_min': ss_min,
        'state_max': ss_max,

        'srf_var': 1,
        'srf_cor': 10,
        'num_cores': 20,
    }
    
    # fake_in, fake_out = get_training_samples(dim_in, dim_out, sa_min, sa_max, 10000)
    # fake_in_ens, fake_out_ens = get_ensemble_training_samples(dim_in, dim_out, sa_min, sa_max)
    fake_in_ens, fake_out_ens = get_ensemble_training_samples(params)
    print(f'Shape of generated data:\ninput -> {fake_in_ens.shape}\n'\
          f'output -> {fake_out_ens.shape}')
    import pdb; pdb.set_trace()
    exit()
    # structured field with a size 100x100 and a grid-size of 1x1
    x = y = range(100)
    model = gs.Gaussian(dim=2, var=1, len_scale=10)
    srf = gs.SRF(model)
    srf((x, y), mesh_type='structured')
    srf.plot()
    
    plt.show()
