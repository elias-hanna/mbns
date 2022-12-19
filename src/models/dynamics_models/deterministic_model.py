import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

import src.torch.pytorch_util as ptu


class DeterministicDynModel(nn.Module):
    """
    Predicts residual of next state given current state and action
    """
    def __init__(
            self,
            obs_dim,              # Observation dim of environment
            action_dim,           # Action dim of environment
            hidden_size,          # Hidden size for model
            init_method='kaiming',# weight init method
            sa_min=None,          # State-Action space min values
            sa_max=None,          # State-Action space max values
            use_minmax_norm=False,
    ):
        super(DeterministicDynModel, self).__init__()

        torch.set_num_threads(16)
        
        self.obs_dim, self.action_dim = obs_dim, action_dim

        self.input_dim = self.obs_dim + self.action_dim
        self.output_dim = self.obs_dim # fitness always a scalar
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size)
        # self.fc2 = nn.Linear(self.hidden_size, self.output_dim)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.output_dim)

        self.MSEcriterion = nn.MSELoss()
        self.L1criterion = nn.L1Loss()

        # the trainer computes the mu and std of the train dataset
        self.input_mu = nn.Parameter(ptu.zeros(1,self.input_dim), requires_grad=False).float()
        self.input_std = nn.Parameter(ptu.ones(1,self.input_dim), requires_grad=False).float()

        
        self.output_mu = nn.Parameter(ptu.zeros(1,self.output_dim), requires_grad=False).float()
        self.output_std = nn.Parameter(ptu.ones(1,self.output_dim), requires_grad=False).float()

        if init_method == 'uniform':
            a = -0.05; b = 0.05
            # uniform intialization of weights
            nn.init.uniform_(self.fc1.weight, a=a, b=b)
            nn.init.uniform_(self.fc2.weight, a=a, b=b)
            # nn.init.uniform_(self.fc3.weight, a=a, b=b)
            # nn.init.uniform_(self.fc4.weight, a=a, b=b)
        elif init_method == 'xavier':
            # xavier uniform intialization of weights
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            # nn.init.xavier_uniform_(self.fc3.weight)
            # nn.init.xavier_uniform_(self.fc4.weight)
        elif init_method == 'kaiming':
            # kaiming uniform intialization of weights
            nn.init.kaiming_uniform_(self.fc1.weight)
            nn.init.kaiming_uniform_(self.fc2.weight)
            nn.init.kaiming_uniform_(self.fc3.weight)
            nn.init.kaiming_uniform_(self.fc4.weight)
        elif init_method == 'orthogonal':
            # orthogonal intialization of weights
            nn.init.orthogonal_(self.fc1.weight)
            nn.init.orthogonal_(self.fc2.weight)
            # nn.init.orthogonal_(self.fc3.weight)
            # nn.init.orthogonal_(self.fc4.weight)

        self.sa_min = ptu.from_numpy(sa_min)
        self.sa_max = ptu.from_numpy(sa_max)
        self.use_minmax_norm = use_minmax_norm
        
    def forward(self, x_input):

        # normalize the inputs
        if self.use_minmax_norm:
            h = self.normalize_inputs_sa_minmax(x_input)
        else:
            h = self.normalize_inputs(x_input)
        # print('xin norm:',h)
        # import pdb; pdb.set_trace()
        # x = torch.relu(self.fc1(h))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc1(h))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        # x = self.fc2(x)
        return x
    
    def get_loss(self, x, y, return_l2_error=False):

        # predicted normalized outputs given inputs
        pred_y = self.forward(x)

        # normalize the output/label as well
        if self.use_minmax_norm:
            norm_y = self.normalize_outputs_sa_minmax(y)
        else:
            norm_y = self.normalize_outputs(y)

        # calculate loss - we want to predict the normalized residual of next state
        loss = self.MSEcriterion(pred_y, norm_y)
        #loss = self.L1criterion(pred_y, y)

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
    def output_pred(self, x_input):
        # x_input = np.random.uniform(low=self.sa_min, high=self.sa_max, size=x_input.shape)
        # x_input = ptu.from_numpy(x_input)
        # batch_preds is the normalized output from the network
        # print('xin:', x_input)
        batch_preds = self.forward(x_input)
        # print('xout norm:', batch_preds)
        if self.use_minmax_norm:
            y = self.denormalize_output_sa_minmax(batch_preds)
            # print('xout:', y)
        else:
            y = self.denormalize_output(batch_preds)
        output = ptu.get_numpy(y)
        # import pdb; pdb.set_trace()
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
