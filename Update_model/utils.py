from typing import Any

import pandas as pd
import numpy as np
import sys

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from ukf import *

def model_parameter_list(model : nn.Module):
    learnable_params = list()
    for p, k in zip(model.parameters(), model.state_dict().keys()):
        if p.requires_grad == True:
            learnable_params.extend(model.state_dict()[k].flatten().tolist())
    return learnable_params

def get_f(name: str = "tanh") -> nn.Module:
    activations = nn.ModuleDict(
        [
            ["lrelu", nn.LeakyReLU(0.1)],
            ["relu", nn.ReLU()],
            ["tanh", nn.Tanh()],
        ]
    )
    return activations[name]

def dense_block(layers : list(), activation: str) -> nn.Sequential:
    out = list()
    for idx, (in_f, out_f) in enumerate(zip(layers, layers[1:])):
        if idx == len(layers) - 2:
            out.append(nn.Linear(in_f, out_f, bias=False))
        else:
            out.append(nn.Linear(in_f, out_f))
            if activation is not None:
                out.append(get_f(activation))
    return nn.Sequential(*out)

class BaseNN(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.apply(self._init_weights)

    def forward(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def _init_weights(self, module):
        """Define how to initialize weights and biases for each type of layer."""
        if isinstance(module, nn.Linear):
            fan_out, fan_in = module.weight.data.size()

            # [Option 1]
            # --- Xavier "truncated" normal + zero bias
            std = np.sqrt(2.0 / (fan_in + fan_out))

            # # [Option 2]
            # # # --- He "truncated" normal + zero bias
            # std = np.sqrt(2.0 / (fan_in))

            torch.nn.init.trunc_normal_(module.weight.data, std=std, mean=0, a=-2, b=2)
            if module.bias is not None:
                module.bias.data.zero_()

class ANN(BaseNN):
    def __init__(
            self,
            layers : list = [3, 4, 8, 16, 8, 4, 1],
            activation : str = 'relu'
    ):
        super().__init__()
        self.net = dense_block(layers, activation)
        self.apply(self._init_weights)

    def forward(self, x):
        return self.net(x)

class EV_dataset(Dataset):
    def __init__(
            self,
            overall_path,
            trip_path,
            v_nums : list = [0],
            routes : list = [0],
            normalize : bool = True
    ):
        super(EV_dataset, self).__init__()
        self.overall_data = pd.read_csv(overall_path)
        self.trip_data = pd.read_csv(trip_path)
        self.trip_data['trip_velocity'] = self.trip_data['trip_dist'] / self.trip_data['trip_duration']

        self.normalizing_factors = dict()
        if normalize:
            for key in ['trip_dist', 'trip_v']:
                self.normalizing(self.overall_data, key=key)

            for key in ['trip_velocity', 'trip_load']:
                self.normalizing(self.trip_data, key=key)

        self.v_nums = ['V' + str(i) for i in v_nums]
        self.routes = routes
        self.overall_data = self.overall_data.loc[(self.overall_data['vin'].isin(self.v_nums)) & (self.overall_data['route'].isin(self.routes))]
        self.trip_data = self.trip_data.loc[(self.trip_data['vin'].isin(self.v_nums)) & (self.trip_data['route'].isin(self.routes))]
        trip_velocity_list = []
        trip_load_list = []
        for i in self.v_nums:
            df = self.overall_data.loc[self.overall_data['vin'] == i]
            dd = df['route'].value_counts()
            for j in self.routes:
                if len(self.trip_data.loc[(self.trip_data['vin'] == i) & (self.trip_data['route'] == j)]['trip_velocity'].values) > 1:
                    print(self.trip_data.loc[(self.trip_data['vin'] == i) & (self.trip_data['route'] == j)]['trip_velocity'].values)
                trip_velocity_list.extend([self.trip_data.loc[(self.trip_data['vin'] == i) & (self.trip_data['route'] == j)]['trip_velocity'].values.item()] * dd[j])
                trip_load_list.extend([self.trip_data.loc[(self.trip_data['vin'] == i) & (self.trip_data['route'] == j)]['trip_load'].values.item()] * dd[j])
        self.overall_data['trip_velocity'] = trip_velocity_list
        self.overall_data['trip_load'] = trip_load_list

    def normalizing(self, df, key):
        minmax = (df[key].min(), df[key].max())
        df[key] = (df[key] - minmax[0]) / (minmax[1] - minmax[0])
        self.normalizing_factors[key] = minmax

    def __len__(self):
        return self.overall_data.shape[0]
    
    def __getitem__(self, idx):
        inp = self.overall_data.iloc[idx][['trip_velocity', 'trip_dist', 'trip_load']].values.astype(np.float32)
        tar = np.expand_dims(self.overall_data.iloc[idx]['trip_v'].astype(np.float32), axis=0)
        return  inp, tar

def train_ukf(
        model : torch.nn.Module,
        loader : DataLoader,
        log_dir : str = 'runs',
        epoch : int = 40,
        device : torch.device = 'cpu',
        ukf_params : list = [1, 2, 0]
        ):
    '''
    ukf_params : list of parameters. [alpha, beta, kappa]
    '''
    # if len(loader) > 1:
    #     print('There are several batchs in DataLoader')
    #     print('Use first batch to train')

    # device = next(iter(model.parameters())).device
    model.eval()
    writer = SummaryWriter(log_dir)
    
    learnable_params = list()
    for p, k in zip(model.parameters(), model.state_dict().keys()):
        if p.requires_grad == True:
            learnable_params.extend(model.state_dict()[k].flatten().tolist())
    state_dicts = model.state_dict()

    inp, tar = next(iter(loader))
    if len(inp) > 50 :
        idx = torch.linspace(0, len(inp)-1, 50, dtype=torch.int64)
        inp = inp[idx]
        tar = tar[idx]
    tar = tar.flatten().detach().numpy()
    
    ukf = UKF(dim_x = len(learnable_params), dim_y = len(tar), 
            fx = fx, hx = model, ukf_params=ukf_params, x_mean_fn=None, y_mean_fn=None, init_x=learnable_params)
    ukf.P = 1e-6
    ukf.Q = 1e-6
    ukf.R = 1e-6

    best_loss = 1e10
    learned_params = ukf.return_weight()

    for _ in range(epoch):
        ukf.predict()
        ukf.update(tar, inp)
        current_loss = ukf.return_loss()

        if current_loss <= best_loss:
            best_loss = current_loss
            learned_params = ukf.return_weight()

        if (_+1) % (epoch // 10) == 0:
            print('.', end='')

        writer.add_scalar('train_loss', current_loss, _)

    idx=0
    for p, k in zip(model.parameters(), model.state_dict().keys()):
        if p.requires_grad == True:
            l = len(p.flatten())
            state_dicts[k] = torch.nn.Parameter(torch.from_numpy(learned_params[idx : idx + l].reshape(p.shape)))
            idx += l
    model.load_state_dict(state_dicts)
    return model, ukf