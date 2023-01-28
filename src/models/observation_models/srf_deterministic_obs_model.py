import numpy as np
import src.torch.pytorch_util as ptu
import gstools as gs


class SrfDeterministicObsModel():
    """
    Predicts residual of next state given current state and action
    """
    def __init__(
            self,
            state_dim,              # Observation dim of environment
            obs_dim,           # Action dim of environment
            s_min,          # State space min values
            s_max,          # State space max values
            o_min,          # Obs space min values
            o_max,          # Obs space max values
            var,          # Hidden size for model, either int or array
            len_scale,
            kernels=[gs.Gaussian],
            use_minmax_norm=False,
    ):
                ## Dynamics model params
        self.obs_dim, self.state_dim = obs_dim, state_dim

        self.input_dim = self.state_dim
        self.output_dim = self.obs_dim 
        
        ## State and obs space params
        self.s_min = s_min
        self.s_max = s_max
        self.o_min = o_min
        self.o_max = o_max
        self.use_minmax_norm = use_minmax_norm

        ## SRF params
        self.var = var
        self.len_scale = len_scale

        print('Creating observation model...')
        kernel = kernels[np.random.randint(len(kernels))]
        model = kernel(dim=self.input_dim,
                       var=self.var,
                       len_scale=self.len_scale)
        srfs = []
        for _ in range(self.output_dim):
            srfs.append(gs.SRF(model))
        self.model = srfs
        print('Finished creating observation model !')

    def query_srfs(self, input_data, srfs):
        ret_data = []
        for srf in srfs:
            ret_data.append(srf(input_data))
        ret_data = np.array(ret_data)
        ret_data = np.transpose(ret_data)
        ret_data += input_data[:,:self.obs_dim]
        ## don't do below since we added --clip-state or --clip-obs option
        # ret_data = np.clip(ret_data, -1, 1) # clip in normalized space?
        return ret_data

    def output_pred(self, x_input):
        x_input = ptu.get_numpy(x_input) # numpify it
        ## Normalize data
        norm_input = x_input
        # norm_input = self.normalize_inputs_s_minmax(x_input)
        batch_pred = self.query_srfs(norm_input, self.model)
        output = self.denormalize_outputs_o_minmax(batch_pred)
        return np.array(output)

    def normalize_inputs_s_minmax(self, data):
        data_norm = (data - self.s_min)/(self.s_max - self.s_min)
        rescaled_data_norm = data_norm * (1 + 1) - 1
        return rescaled_data_norm

    def denormalize_outputs_o_minmax(self, data):
        data_denorm = ((data + 1)*(self.o_max - self.o_min) / \
                       (1 + 1)) + self.o_min
        return data_denorm
