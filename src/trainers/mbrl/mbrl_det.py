from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

import src.torch.pytorch_util as ptu
from src.trainers.trainer import TorchTrainer


class MBRLTrainer(TorchTrainer):
    def __init__(
            self,
            model,
            learning_rate=1e-3,
            batch_size=512,
            optimizer_class=optim.Adam,
            train_call_freq=1,
            **kwargs
    ):
        super().__init__()

        self.model = model
        
        self.obs_dim = model.obs_dim
        self.action_dim = model.action_dim
        self.batch_size = batch_size
        self.train_call_freq = train_call_freq
        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)

        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_from_buffer(self, replay_buffer, holdout_pct=0.2, max_grad_steps=1000, epochs_since_last_update=5, verbose=False):
        self._n_train_steps_total += 1
        if self._n_train_steps_total % self.train_call_freq > 0 and self._n_train_steps_total > 1:
            return

        data = replay_buffer.get_transitions()

        if len(data) < 1:
            return

        #x = data[:,:self.obs_dim + self.action_dim]  # inputs  s, a
        #y = data[:,self.obs_dim + self.action_dim:]  # predict r, d, ns

        x = data[:,:self.obs_dim + self.action_dim]  # inputs  s, a
        y = data[:,-self.obs_dim:]                   # predict  ns

        y[:,-self.obs_dim:] -= x[:,:self.obs_dim]    # predict delta in the state

        # normalize network inputs
        self.model.fit_input_stats(x)
        self.model.fit_output_stats(y)
        
        # generate holdout set
        inds = np.random.permutation(data.shape[0])
        x, y = x[inds], y[inds]

        n_train = max(int((1-holdout_pct) * data.shape[0]), data.shape[0] - 8092)
        n_test = data.shape[0] - n_train

        x_train, y_train = x[:n_train], y[:n_train]
        x_test, y_test = x[n_train:], y[n_train:]
        x_test, y_test = ptu.from_numpy(x_test), ptu.from_numpy(y_test)

        # train until holdout set convergence
        num_epochs, num_steps = 0, 0
        num_epochs_since_last_update = 0
        best_holdout_loss = float('inf')
        num_batches = int(np.ceil(n_train / self.batch_size))

        # if verbose:
        #     print('###########################################################')
        #     print('################# Model training stats ####################')
        #     print('###########################################################')

        while num_epochs_since_last_update < epochs_since_last_update and num_steps < max_grad_steps:
            train_loss = 0 
            # generate idx for each model to bootstrap
            self.model.train()
            for b in range(num_batches):
                # sample some idxs from train set
                b_idxs = np.random.randint(n_train, size=self.batch_size)
                x_batch, y_batch = x_train[b_idxs], y_train[b_idxs]

                x_batch, y_batch = ptu.from_numpy(x_batch), ptu.from_numpy(y_batch)
                x_batch = x_batch.view(self.batch_size, -1)
                y_batch = y_batch.view(self.batch_size, -1)
                loss = self.model.get_loss(x_batch, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss
                
            num_steps += num_batches

            # stop training based on holdout loss improvement
            self.model.eval()
            with torch.no_grad():
                holdout_losses = self.model.get_loss(x_test, y_test)
            holdout_loss = holdout_losses

            if num_epochs == 0 or \
               (best_holdout_loss - holdout_loss) / abs(best_holdout_loss) > 0.01:
                best_holdout_loss = holdout_loss
                num_epochs_since_last_update = 0
            else:
                num_epochs_since_last_update += 1
            # if verbose:
            #     print("Num epochs: ", num_epochs, "    Avg Train Loss: ",
            #           train_loss.item()/num_batches)
            #     print("Num epochs: ", num_epochs, "    Holdout Loss: ",
            #           holdout_loss.item())
            #     print()
            
            num_epochs += 1

        self.end_epoch(num_epochs)
        # if verbose:
        #     print("Num epochs: ", num_epochs, "    Avg Train Loss: ",
        #           train_loss.item()/num_batches)
        #     print("Num epochs: ", num_epochs, "    Holdout Loss: ",
        #           holdout_loss.item())
        #     print()
            
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            self.eval_statistics['Model Final Train Loss (MSE)'] = \
                np.mean(ptu.get_numpy(train_loss)/num_batches)
            self.eval_statistics['Model Holdout Loss (MSE)'] = \
                np.mean(ptu.get_numpy(holdout_loss))
            self.eval_statistics['Model Training Epochs'] = num_epochs
            self.eval_statistics['Model Training Steps'] = num_steps

            if verbose:
                print('###########################################################')
                print('################# Final training stats ####################')
                print('###########################################################')
                for key in self.eval_statistics.keys():
                    print(f'{key}: {self.eval_statistics[key]}')
                print('###########################################################')
                print('###########################################################')
                print('###########################################################')
                print()
            
    def train_from_torch(self, batch, idx=None):
        raise NotImplementedError

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.model
        ]

    def get_snapshot(self):
        return dict(
            model=self.model
        )
