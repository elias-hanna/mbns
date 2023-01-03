import gstools as gs
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from itertools import repeat

def query_srfs(input_data, srfs):
    ret_data = []
    for srf in srfs:
        ret_data.append(srf(input_data))
    ret_data = np.array(ret_data)
    return np.transpose(ret_data)

def get_training_samples(dim_in, dim_out, sa_min, sa_max,
                         n_training_samples):
    ## Create a srfs model
    model = gs.Gaussian(dim=dim_in, var=1, len_scale=10)
    srfs = []
    for _ in range(dim_out):
        srfs.append(gs.SRF(model))
    
    ## Create fake input data
    fake_input_data = np.random.uniform(low=sa_min, high=sa_max,
                                        size=(n_training_samples, dim_in))
    ## Query the model using fake input data
    fake_output_data = query_srfs(fake_input_data, srfs)
    ## Return the fake input and output training data
    return fake_input_data, fake_output_data

def get_ensemble_training_samples(dim_in, dim_out, sa_min, sa_max,
                                  n_training_samples=10000, ensemble_size=10):

    ensemble_fake_training_samples_input = []
    ensemble_fake_training_samples_output = []
    # for _ in range(ensemble_size):
    #     input_data, output_data = get_training_samples(
    #         dim_in, dim_out, sa_min, sa_max, n_training_samples
    #     )
        
    #     ensemble_fake_training_samples_input.append(input_data)
    #     ensemble_fake_training_samples_output.append(output_data)
    # return np.array(ensemble_fake_training_samples_input), \
        # np.array(ensemble_fake_training_samples_output)
        
    with Pool(processes=20) as pool:
        to_evaluate = zip(repeat(dim_in,ensemble_size), repeat(dim_out),
                          repeat(sa_min), repeat(sa_max), repeat(n_training_samples))
        input_data, output_data = zip(*pool.starmap(
        # input_data, output_data = pool.starmap(
            get_training_samples,
            # zip(dim_in, dim_out, sa_min, sa_max, n_training_samples)
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

    # fake_in, fake_out = get_training_samples(dim_in, dim_out, sa_min, sa_max, 10000)
    fake_in_ens, fake_out_ens = get_ensemble_training_samples(dim_in, dim_out, sa_min, sa_max)
    import pdb; pdb.set_trace()
    # structured field with a size 100x100 and a grid-size of 1x1
    x = y = range(100)
    model = gs.Gaussian(dim=2, var=1, len_scale=10)
    srf = gs.SRF(model)
    srf((x, y), mesh_type='structured')
    srf.plot()
    
    plt.show()
