import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

import src.torch.pytorch_util as ptu


class DeterministicObsModel(nn.Module):
    """
    Predicts residual of next state given current state and action
    """
    def __init__(
            self,
            state_dim,              # Observation dim of environment
            obs_dim,           # Action dim of environment
            hidden_size,          # Hidden size for model, either int or array
            init_method='uniform',# weight init method
            so_min=None,          # State-Action space min values
            so_max=None,          # State-Action space max values
            use_minmax_norm=False,
            hidden_activation=torch.relu,
    ):
        super(DeterministicObsModel, self).__init__()

        torch.set_num_threads(16)
        
        self.obs_dim, self.state_dim = obs_dim, state_dim

        self.input_dim = self.state_dim
        self.output_dim = self.obs_dim
        self.hidden_size = hidden_size

        self.fcs = []

        in_size = self.input_dim
        for i in range(len(hidden_size)):
            self.fcs.append(nn.Linear(in_size, self.hidden_size[i]))
            self.__setattr__("fc{}".format(i), self.fcs[i])
            in_size = self.hidden_size[i] 
        self.fcs.append(nn.Linear(self.hidden_size[-1], self.output_dim))
        self.__setattr__("fc{}".format(i+1), self.fcs[-1])

        self.MSEcriterion = nn.MSELoss()
        self.L1criterion = nn.L1Loss()

        # the trainer computes the mu and std of the train dataset
        self.input_mu = nn.Parameter(ptu.zeros(1,self.input_dim), requires_grad=False).float()
        self.input_std = nn.Parameter(ptu.ones(1,self.input_dim), requires_grad=False).float()

        
        self.output_mu = nn.Parameter(ptu.zeros(1,self.output_dim), requires_grad=False).float()
        self.output_std = nn.Parameter(ptu.ones(1,self.output_dim), requires_grad=False).float()

        kwargs={}
        if init_method == 'uniform':
            # uniform intialization of weights
            self.init_method = nn.init.uniform_
            a = -5; b = 5
            kwargs['a'] = a; kwargs['b'] = b
        elif init_method == 'xavier':
            # xavier uniform intialization of weights
            self.init_method = nn.init.xavier_uniform_
        elif init_method == 'kaiming':
            # kaiming uniform intialization of weights
            self.init_method = nn.init.kaiming_uniform_
        elif init_method == 'orthogonal':
            # orthogonal intialization of weights
            self.init_method = nn.init.orthogonal

        for fc in self.fcs:
            self.init_method(fc.weight, **kwargs)
            
        self.hidden_activation = hidden_activation
        
        self.so_min = ptu.from_numpy(so_min)
        self.so_max = ptu.from_numpy(so_max)
        self.use_minmax_norm = use_minmax_norm
        
    def forward(self, x_input):

        # normalize the inputs
        if self.use_minmax_norm:
            h = self.normalize_inputs_so_minmax(x_input)
        else:
            h = self.normalize_inputs(x_input)

        x = h
        
        for i in range(len(self.hidden_size)):
            x = self.hidden_activation(self.fcs[i](x))
        x = self.fcs[-1](x)

        return x


    def get_loss(self, x, y, return_l2_error=False):

        # predicted normalized outputs given inputs
        pred_y = self.forward(x)

        # normalize the output/label as well
        if self.use_minmax_norm:
            norm_y = self.normalize_outputs_so_minmax(y)
        else:
            norm_y = self.normalize_outputs(y)
        # calculate loss - we want to predict the normalized residual of next state
        loss = self.MSEcriterion(pred_y, norm_y)
        return loss

    # get input data mean and std for normalization
    def fit_input_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0

        if mask is not None:
            mean *= mask
            std *= mask

        self.input_mu.data = ptu.from_numpy(mean)
        self.input_std.data = ptu.from_numpy(std)

    # get output data mean and std for normalization
    def fit_output_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        
        if mask is not None:
            mean *= mask
            std *= mask

        self.output_mu.data = ptu.from_numpy(mean)
        self.output_std.data = ptu.from_numpy(std)

    #output predictions after unnormalized
    def output_pred(self, x_input, mean=False):
        # batch_preds is the normalized output from the network
        batch_preds = self.forward(x_input)
        if self.use_minmax_norm:
            y = self.denormalize_output_so_minmax(batch_preds)
        else:
            y = self.denormalize_output(batch_preds)
        output = ptu.get_numpy(y)
        return output
    
    def normalize_inputs(self, data):
        data_norm = (data - self.input_mu)/(self.input_std)
        return data_norm

    def normalize_outputs(self, data):
        data_norm = (data - self.output_mu)/(self.output_std)
        return data_norm

    def denormalize_output(self, data):
        data_denorm = data*self.output_std + self.output_mu
        return data_denorm

    def normalize_inputs_so_minmax(self, data):
        data_norm = (data - self.so_min)/(self.so_max - self.so_min)
        rescaled_data_norm = data_norm * (1 + 1) - 1
        return rescaled_data_norm

    def normalize_outputs_so_minmax(self, data):
        # data_norm = (data - self.so_min[self.obs_dim:])/ \
        #             (self.so_max[self.obs_dim:] - self.so_min[self.obs_dim:])
        # rescaled_data_norm = data_norm * (1 + 1) - 1
        # return rescaled_data_norm
        # return data_norm
        return data

    def denormalize_output_so_minmax(self, data):
        # data_denorm = data*(self.so_max[self.obs_dim:] - self.so_min[self.obs_dim:]) + \
                      # self.so_min[self.obs_dim:]
        # data_denorm = ((data + 1)*(self.so_max[self.obs_dim:] - self.so_min[self.obs_dim:]) / \
                       # (1 + 1)) + self.so_min[self.obs_dim:]
        # return data_denorm
        return data
