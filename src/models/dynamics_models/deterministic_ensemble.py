import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

import src.torch.pytorch_util as ptu

from src.models.dynamics_models.deterministic_model import DeterministicDynModel

class DeterministicEnsemble():
    """
    Predicts residual of next state given current state and action
    """
    def __init__(
            self,
            obs_dim,              # Observation dim of environment
            action_dim,           # Action dim of environment
            hidden_size,          # Hidden size for model
            ensemble_size=4,      # Ensemble size
            init_method='uniform',# weight init method
            sa_min=None,          # State-Action space min values
            sa_max=None,          # State-Action space max values
            use_minmax_norm=False,
            hidden_activation=torch.tanh,
    ):
        # super(DeterministicEnsemble, self).__init__()

        # torch.set_num_threads(16)
        
        self.obs_dim, self.action_dim = obs_dim, action_dim

        self.input_dim = self.obs_dim + self.action_dim
        self.output_dim = self.obs_dim # fitness always a scalar
        self.hidden_size = hidden_size
        
        self.MSEcriterion = nn.MSELoss()
        self.L1criterion = nn.L1Loss()

        self.hidden_activation = hidden_activation
        
        self.sa_min = ptu.from_numpy(sa_min)
        self.sa_max = ptu.from_numpy(sa_max)
        self.use_minmax_norm = use_minmax_norm

        self.ensemble_size = ensemble_size
        
        ## Init the n ensembles ##
        self.models = [DeterministicDynModel(obs_dim,
                                             action_dim,
                                             hidden_size,
                                             init_method=init_method,
                                             sa_min=sa_min,
                                             sa_max=sa_max,
                                             use_minmax_norm=use_minmax_norm,
                                             hidden_activation=hidden_activation,)
                       for _ in range(self.ensemble_size)]
        
    def forward(self, x_input, mean=False):

        # normalize the inputs
        if self.use_minmax_norm:
            h = self.normalize_inputs_sa_minmax(x_input)
        else:
            h = self.normalize_inputs(x_input)

        xs = []
        for model in self.models:
            xs.append(model.forward(h))

        if mean:
            return np.mean(xs, axis=0)
        
        return np.array(xs)
    
    def get_loss(self, x, y, mean=False, return_l2_error=False):

        pred_ys = []
        for model in self.models:
            # predicted normalized outputs given inputs
            pred_ys.append(model.forward(x))

        # normalize the output/label as well
        if self.use_minmax_norm:
            norm_y = self.normalize_outputs_sa_minmax(y)
        else:
            norm_y = self.normalize_outputs(y)

        losses = []
        for pred_y in pred_ys:
            # calculate loss - we want to predict the normalized residual of next state
            losses.append(self.MSEcriterion(pred_y, norm_y))
            # losses.append(model.MSEcriterion(pred_y, norm_y))
            #losses.append(self.L1criterion(pred_y, y))

        if mean:
            return np.mean(loss, axis=0)
        
        return np.array(loss)

    #output predictions after unnormalized
    def output_pred(self, x_input, mean=False):
        batch_preds = []
        for model in self.models:
            batch_pred = model.forward(x_input)
            
            if self.use_minmax_norm:
                y = self.denormalize_output_sa_minmax(batch_pred)
            else:
                y = self.denormalize_output(batch_pred)
            batch_preds.append(ptu.get_numpy(y))
            
        if mean:
            return np.mean(batch_preds, axis=0)
        
        return np.array(batch_preds)

    def output_pred_with_ts(self, x_input, mean=False):
        ## x_input needs to be in the form
        ## x_input = [x0 model 1]
        ##           [x0 model 2]
        ##           [x1 model 1]
        ##           [x1 model 2] etc...
        ## cba to think of a stronger assert sorry
        assert len(x_input) % self.ensemble_size == 0 
        batch_preds = []
        ## run the particules through each model
        for model, i in zip(self.models, range(self.ensemble_size)):
            batch_pred = model.forward(x_input[i::self.ensemble_size])
            if self.use_minmax_norm:
                y = self.denormalize_output_sa_minmax(batch_pred)
            else:
                y = self.denormalize_output(batch_pred)
            batch_preds.append(ptu.get_numpy(y))

        return np.array(batch_preds)
    
    def normalize_inputs(self, data):
        data_norm = (data - self.input_mu)/(self.input_std)
        return data_norm

    def normalize_outputs(self, data):
        data_norm = (data - self.output_mu)/(self.output_std)
        return data_norm

    def denormalize_output(self, data):
        data_denorm = data*self.output_std + self.output_mu
        return data_denorm

    def normalize_inputs_sa_minmax(self, data):
        data_norm = (data - self.sa_min)/(self.sa_max - self.sa_min)
        rescaled_data_norm = data_norm * (1 + 1) - 1
        return rescaled_data_norm
        # return data_norm

    def normalize_outputs_sa_minmax(self, data):
        # data_norm = (data - self.sa_min[:self.obs_dim])/ \
        #             (self.sa_max[:self.obs_dim] - self.sa_min[:self.obs_dim])
        # rescaled_data_norm = data_norm * (1 + 1) - 1
        # return rescaled_data_norm
        return data_norm

    def denormalize_output_sa_minmax(self, data):
        # data_denorm = data*(self.sa_max[:self.obs_dim] - self.sa_min[:self.obs_dim]) + \
                      # self.sa_min[:self.obs_dim]
        # data_denorm = ((data + 1)*(self.sa_max[:self.obs_dim] - self.sa_min[:self.obs_dim]) / \
                       # (1 + 1)) + self.sa_min[:self.obs_dim]
        # return data_denorm
        return data
