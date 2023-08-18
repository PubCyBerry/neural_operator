from typing import Any
import pandas as pd
import numpy as np
import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import multiprocessing
from pathos.pools import ProcessPool

def output_with_each_particle(
        model : nn.Module,
        params,
        model_input,
        model_target
        ):

    state_dicts = model.state_dict()
    idx = 0
    for p, k in zip(model.parameters(), model.state_dict().keys()):
        if p.requires_grad == True:
            l = len(p.flatten())
            state_dicts[k] = nn.Parameter(torch.from_numpy(params[idx : idx + l].reshape(p.shape)))
            idx += l
    model.load_state_dict(state_dicts)

    output = model(model_input)
    obj_func = nn.functional.l1_loss(output, model_target)
    return obj_func.item()

class PSO():
    def __init__(
            self, 
            func, 
            n_dim=None, 
            pop=40, 
            max_iter=150, 
            lb=-1, 
            ub=1, 
            w=0.8,
            c1=0.5, 
            c2=0.5, 
            c3=0.5,
            constraint_eq=tuple(), 
            constraint_ueq=tuple(), 
            verbose=False, 
            dim=None, 
            data = tuple(), 
            p_interval = 0, 
            initial_values = list()
            ):

        n_dim = n_dim or dim  # support the earlier version

        self.func = func
        self.w = w # inertia
        self.init_w = w # initial inertia
        self.cp, self.cg, self.cc = c1, c2, c3 # parameters to control personal best, global best, geometric centre respectively
        self.pop = pop  # number of particles
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not

        # self.input = data[0]
        # self.target = np.tile(data[1], (self.pop, 1, 1))
        self.input = torch.tile(data[0].unsqueeze(dim=0), (self.pop, 1, 1))
        self.target = torch.tile(data[1].unsqueeze(dim=0), (self.pop, 1, 1))
        self.model_list = [deepcopy(self.func) for _ in range(self.pop)]

        self.p_interval = p_interval if p_interval else 1

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'
        self.has_constraint = bool(constraint_ueq)
        self.constraint_ueq = constraint_ueq
        self.is_feasible = np.array([True] * pop)

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        if initial_values:
            assert len(initial_values) == n_dim, 'must set initial values for all parameters'
            self.initial_values = initial_values
            self.X += np.tile(np.array(self.initial_values), (self.pop, 1))

        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))  # speed of particles
        self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = np.array([[np.inf]] * pop)  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = [self.gbest_y]  # gbest_y of every iteration
        self.update_gbest()

        self.centre = np.sum(self.pbest_x, axis=0) / self.pop

        # record verbose values
        self.record_mode = True
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated

    def initialize_V(self):
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))
        self.w = self.init_w 
        # print(self.w, self.init_w)

    def check_constraint(self, x):
        # gather all unequal constraint functions
        for constraint_func in self.constraint_ueq:
            if constraint_func(x) > 0:
                return False
        return True

    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        r3 = np.random.rand(self.pop, self.n_dim)

        self.V *= self.w
        self.V += self.cp * r1 * (self.pbest_x - self.X)
        self.V += self.cg * r2 * (self.gbest_x - self.X)
        self.V += self.cc * r3 * (self.centre - self.X)

        # self.V = self.w * self.V + \
        #          self.cp * r1 * (self.pbest_x - self.X) + \
        #          self.cg * r2 * (self.gbest_x - self.X) + \
        #          self.cc * r3 * (self.centre - self.X)

    def update_X(self):
        self.X = self.X + self.V
        # self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        p = ProcessPool(nodes=os.cpu_count())
        output = p.map(output_with_each_particle, self.model_list, self.X, self.input, self.target)
        self.Y = np.expand_dims(np.array(output), axis=1)

        # output = p.map(self.func, self.input, self.X)
        # self.Y = np.expand_dims(np.array([np.mean(np.abs(self.target - i)) for i in output]), axis=1)
        # return self.Y

    def update_centre(self):
        self.centre = np.sum(self.pbest_x, axis=0) / self.pop

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y
        for idx, x in enumerate(self.X):
            if self.need_update[idx]:
                self.need_update[idx] = self.check_constraint(x)

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None, precision=None, N=20, p_interval=0, terminate_threshold=0):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter or self.max_iter
        if p_interval:
            self.p_interval=p_interval
        else:
            self.p_interval=1

        c = 0
        temp = 0
        for iter_num in range(self.max_iter):
            self.update_V()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()
            if iter_num % self.p_interval == 0:
                self.update_centre()
                
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0

            if self.verbose and ((iter_num % (self.max_iter // 10) == 0) or (iter_num+1 == self.max_iter)):
                print('Iter: {}, Best fit: {}'.format(iter_num, self.gbest_y))
                # print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

            if self.gbest_y == self.gbest_y_hist[-1]:
                temp += 1
                if temp == 5:
                    self.initialize_V()
                    temp = 0
            else:
                temp = 0
            self.gbest_y_hist.append(self.gbest_y)
            
            self.recorder()

            if self.gbest_y < terminate_threshold:
                self.best_x, self.best_y = self.gbest_x, self.gbest_y
                print('achieve terminate threshold, ', end='')
                return self.best_x, self.best_y
            
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

def train_pso(
        model : torch.nn.Module,
        loader : DataLoader,
        log_dir : str = 'runs/pso/',
        pop : int = 40,
        epoch : int = 40,
        lb : float = -0.5,
        ub : float = 0.5,
        w : float = 0.85,
        c : list = [1,1,1],
        terminate_threshold : int = 0,
        device : torch.device = 'cpu',
        p_interval : int = 1
        ):
    c1, c2, c3 = c

    model.eval()

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
    # tar = tar.flatten().detach().numpy()

    pso = PSO(func=model, n_dim=len(learnable_params), pop=pop,
                max_iter=epoch, lb=lb, ub=ub, w=w, 
                c1=c1, c2=c2, c3=c3, data=(inp,tar), verbose=False, initial_values=learnable_params)
    
    learned_params, best_loss = pso.run(p_interval=p_interval, terminate_threshold=terminate_threshold)

    writer = SummaryWriter(log_dir)
    for idx, i in enumerate(pso.gbest_y_hist):
        writer.add_scalar('train_loss', i, idx)

    idx=0
    for p, k in zip(model.parameters(), model.state_dict().keys()):
        if p.requires_grad == True:
            l = len(p.flatten())
            state_dicts[k] = torch.nn.Parameter(torch.from_numpy(learned_params[idx : idx + l].reshape(p.shape)))
            idx += l
    model.load_state_dict(state_dicts)
    return model, pso