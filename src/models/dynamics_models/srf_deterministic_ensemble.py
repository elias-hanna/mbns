import numpy as np
import src.torch.pytorch_util as ptu
import gstools as gs

class SrfDeterministicEnsemble():
    """
    Predicts residual of next state given current state and action
    """
    def __init__(
            self,
            obs_dim,              # Observation dim of environment
            action_dim,           # Action dim of environment
            sa_min,          # State-Action space min values
            sa_max,          # State-Action space max values
            var,             # variance of the srf model 
            len_scale,
            ensemble_size=4, # ensemble size
            kernels=[gs.Gaussian],
            use_minmax_norm=False,
    ):
        ## Dynamics model params
        self.obs_dim, self.action_dim = obs_dim, action_dim

        self.input_dim = self.obs_dim + self.action_dim
        self.output_dim = self.obs_dim # fitness always a scalar
        self.ensemble_size = ensemble_size
        
        ## State-Action space params
        self.sa_min = sa_min
        self.sa_max = sa_max
        self.use_minmax_norm = use_minmax_norm

        ## SRF params
        self.var = var
        self.len_scale = len_scale
        # create ensemble_size models drawn from 'kernels'
        self.models = []

        self.output_min = np.array([-5,-5,-0.25,-0.25,-0.05,-0.05])
        self.output_max = np.array([5,5,0.25,0.25,0.05,0.05])

        print('Creating dynamics model ensemble...')
        for ens_idx in range(self.ensemble_size):
            kernel = kernels[np.random.randint(len(kernels))]
            model = kernel(dim=self.input_dim,
                           var=self.var,
                           len_scale=self.len_scale)
            srfs = []
            for _ in range(self.output_dim):
                srfs.append(gs.SRF(model))
            self.models.append(srfs)
        print('Finished creating dynamics model ensemble !')

    def query_srfs(self, input_data, srfs):
        ret_data = []
        for srf in srfs:
            ret_data.append(srf(input_data))
        ret_data = np.array(ret_data)
        ret_data = np.transpose(ret_data)
        ## Don't do below since its done in the evaluation loop
        # ret_data -= input_data[:,:self.obs_dim]
        ## don't do below since we added --clip-state or --clip-obs option
        # ret_data = np.clip(ret_data, -1, 1) # clip in normalized space?
        
        return ret_data

    def output_pred_with_ts(self, x_input, mean=False):
        x_input = ptu.get_numpy(x_input) # numpify it
        ## Normalize data
        norm_input = self.normalize_inputs_sa_minmax(x_input)
        batch_preds = []
        for model, i in zip(self.models, range(self.ensemble_size)):        
            ## Query the model using norm input data
            # norm_output = self.query_srfs(norm_input, model)
            batch_pred = self.query_srfs(norm_input[i::self.ensemble_size], model)
            ## Denormalize data
            ## Denormalize output
            # output = batch_pred
            # output = self.denormalize_outputs_sa_minmax(batch_pred)
            output = self.denormalize_outputs(batch_pred)
            ## Remove previous state
            # output -= x_input[i::self.ensemble_size, :self.obs_dim]
            batch_preds.append(output)

        return np.array(batch_preds)
    
    def normalize_inputs_sa_minmax(self, data):
        data_norm = (data - self.sa_min)/(self.sa_max - self.sa_min)
        rescaled_data_norm = data_norm * (1 + 1) - 1
        return rescaled_data_norm

    def denormalize_outputs_sa_minmax(self, data):
        data_denorm = ((data + 1)*(self.sa_max[:self.obs_dim] - self.sa_min[:self.obs_dim]) / \
                       (1 + 1)) + self.sa_min[:self.obs_dim]
        return data_denorm

    def denormalize_outputs(self, data):
        data_denorm = ((data + 1)*(self.output_max - self.output_min) / \
                       (1 + 1)) + self.output_min
        return data_denorm
