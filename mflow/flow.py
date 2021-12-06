import math
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import time

import copy


class Flow(nn.Module):
    def __init__(self, *layers):
        super(self.__class__, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, context=None):
        self.num_inputs = x.size(-1)

        log_det = torch.zeros(x.size(0), 1, device=x.device)
        for layer in self.layers:
            x, _log_det = layer.forward(x, context)
            log_det += _log_det

        # Same ordering as input:
        for layer in self.layers[::-1]:
            if 'Perm' not in str(layer):
                continue
            #x = x[:, layer.reverse_perm]

        return x, log_det


    def inverse(self, u, context=None):
        self.num_inputs = u.size(-1)

        for layer in self.layers:
            if 'Perm' not in str(layer):
                continue
            #u = u[:, layer.perm]

        log_det = torch.zeros(u.size(0), 1, device=u.device)
        for layer in self.layers[::-1]:
            u, _log_det = layer.inverse(u, context)
            log_det += _log_det

        return u, log_det

    def sample(self, n, context=None):
        u = torch.Tensor(n, self.num_inputs).normal_()
        device = next(self.parameters()).device
        u = u.to(device)
        if context is not None:
            context = context.to(device)
        samples, log_det = self.inverse(u, context)
        return samples, log_det

    def log_prob(self, x, context=None):
        u, log_jacob = self.forward(x, context=context)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def base_distribution_log_prob(self, z, context=None):
        log_probs = (-0.5 * z.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        return log_probs.sum(-1, keepdim=True)

    def forward_and_log_prob(self, x, context=None):
        u, log_jacob = self.forward(x, context=context)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        return u, (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample_and_log_prob(self, n, context=None):
        pass

    def fit(self,
            data,
            context=None,
            validation_data=None,
            validation_context=None,
            validation_split=0.0,
            epochs=20,
            batch_size=100,
            patience=np.inf,
            monitor='val_loss',
            shuffle=True,
            lr=1e-3,
            device='cpu',
            verbose=2):
        """
            Method to fit the normalising flow.
        
        """
        

        optimizer = torch.optim.Adam(self.parameters(), lr)

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        if (validation_data is not None) and (not isinstance(validation_data, torch.Tensor)):
            validation_data = torch.tensor(validation_data, dtype=torch.float32)

        if context is not None:
            use_context = True
            if not isinstance(data, torch.Tensor):
                context = torch.tensor(context, dtype=torch.float32)
        else:
            use_context = False

        if (validation_context is not None) and (not isinstance(validation_context, torch.Tensor)):
            validation_context = torch.tensor(validation_context, dtype=torch.float32)
            

        if validation_data is not None:
        
            if use_context:
                train_dl = DataLoader(TensorDataset(data, context), batch_size, shuffle)
                val_dl = DataLoader(TensorDataset(validation_data, validation_context), batch_size, shuffle)
            else:
                train_dl = DataLoader(TensorDataset(data), batch_size, shuffle)
                val_dl = DataLoader(TensorDataset(validation_data), batch_size, shuffle)

            validation = True
        else:
            if validation_split > 0.0 and validation_split < 1.0:
                validation = True
                split = int(data.size()[0] * (1. - validation_split))
                if use_context:
                    data, validation_data = data[:split], data[split:]
                    context, validation_context = context[:split], context[split:]
                    train_dl = DataLoader(TensorDataset(data, context), batch_size, shuffle)
                    val_dl = DataLoader(TensorDataset(validation_data, validation_context), batch_size, shuffle)
                else:
                    data, validation_data = data[:split], data[split:]
                    train_dl = DataLoader(TensorDataset(data), batch_size, shuffle)
                    val_dl = DataLoader(TensorDataset(validation_data), batch_size, shuffle)
            else:
                validation = False
                if use_context:
                    train_dl = DataLoader(TensorDataset(data, context), batch_size, shuffle)
                else:
                    train_dl = DataLoader(TensorDataset(data), batch_size, shuffle)

        history = {} # Collects per-epoch loss
        history['loss'] = []
        history['val_loss'] = []

        if not validation:
            monitor = 'loss'
        best_epoch = 0
        best_loss = np.inf
        best_model = copy.deepcopy(self.state_dict())

        start_time_sec = time.time()

        for epoch in range(epochs):

            # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
            self.train()
            train_loss = 0.0

            for batch in train_dl:

                optimizer.zero_grad()

                if use_context:
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    loss = -self.log_prob(x, y).mean()
                else:
                    x = batch[0].to(device)
                    loss = -self.log_prob(x).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()

                train_loss += loss.data.item() * x.size(0)
            
            train_loss  = train_loss / len(train_dl.dataset)

            history['loss'].append(train_loss)


            # --- EVALUATE ON VALIDATION SET -------------------------------------
            self.eval()
            if validation:
                val_loss = 0.0

                for batch in val_dl:

                    if use_context:
                        x = batch[0].to(device)
                        y = batch[1].to(device)
                        loss = -self.log_prob(x, y).mean()
                    else:
                        x = batch[0].to(device)
                        loss = -self.log_prob(x).mean()

                    val_loss += loss.data.item() * x.size(0)

                val_loss = val_loss / len(val_dl.dataset)

                history['val_loss'].append(val_loss)

        

            if verbose > 1:
                try:
                    print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % \
                        (epoch+1, epochs, train_loss, val_loss))
                except:
                    print('Epoch %3d/%3d, train loss: %5.2f' % \
                        (epoch+1, epochs, train_loss))


            # Monitor loss
            if history[monitor][-1] < best_loss:
                best_loss = history[monitor][-1]
                best_epoch = epoch
                best_model = copy.deepcopy(self.state_dict())


            if epoch - best_epoch >= patience:
                self.load_state_dict(best_model)
                if verbose > 0:
                    print('Finished early after %3d epochs' % (best_epoch))
                    print('Best loss achieved %5.2f' % (best_loss))
                break
        
    
        # END OF TRAINING LOOP

        if verbose > 0:
            end_time_sec       = time.time()
            total_time_sec     = end_time_sec - start_time_sec
            time_per_epoch_sec = total_time_sec / epochs
            print()
            print('Time total:     %5.2f sec' % (total_time_sec))
            print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

        return history