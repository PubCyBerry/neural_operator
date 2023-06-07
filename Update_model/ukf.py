import numpy as np
import torch
import os
from torch import nn
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import cholesky, sqrtm
from functools import partial
from copy import deepcopy

from pathos.pools import ProcessPool
import multiprocessing

class MerweSigmaPoints(object):

    """
    Generates sigma points and weights according to the alpha, beta, kappa formulation of Van der Merwe.

    Parameters
    ----------
    n : int, dimensionality of the state. 2n+1 weights will be generated.

    ukf_params: parameters for computing sigma points

    sigma_method: string, method for computing sigma points

    sqrt_method : string, determines how the square root of a matrix is calculated.

    subtract : callable (x, y), optional, function that computes the difference between x and y.

    Attributes
    ----------
    Wm : np.array, weight for each sigma point for the mean
    Wc : np.array, weight for each sigma point for the covariance
    """

    def __init__(self, n, ukf_params, sigma_method=None, sqrt_method=None, subtract=None):

        self.n = n

        if sigma_method is None or sigma_method == 'merwe':
            self.alpha = ukf_params[0]
            self.beta = ukf_params[1]
            self.kappa = ukf_params[2]
        elif sqrt_method == 'julier':
            self.kappa = ukf_params[0]
        else:
            raise RuntimeError("Invalid method for computing sigma points. Valid options are 'merwe' or 'julier'.")

        if sqrt_method is None or sqrt_method == 'chol':
            self.msqrt = cholesky
        elif sqrt_method == 'sqrtm':
            self.msqrt = sqrtm
        else:
            raise RuntimeError("Invalid method for computing matrix square root. Valid options are 'chol' or 'sqrtm'.")

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()

    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""

        return 2*self.n + 1

    def generate_sigmas(self, x, P):
        """
        Computes the sigma points for an unscented Kalman filter given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.

        Returns
        -------
        sigmas : np.array, of size (n, 2n+1), Two dimensional array of sigma points. Each column contains all of the
                sigmas for one dimension in the problem space. Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}.
        """

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.atleast_2d(P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        sqrt_P = self.msqrt((lambda_ + n)*P)

        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = x
        for k in range(n):
            sigmas[k+1] = self.subtract(x, -sqrt_P[k])
            sigmas[n+k+1] = self.subtract(x, sqrt_P[k])

        return sigmas

    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.
        """
        n = self.n
        lambda_ = self.alpha ** 2 * (n + self.kappa) - n
        c = .5 / (n + lambda_)
        self.Wc = np.full(2 * n + 1, c)
        self.Wm = np.full(2 * n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha ** 2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)


def unscented_transform(sigmas, w_mean, w_cov, noise_cov=None, mean_fn=None, residual_fn=None):
    """
    Computes unscented transform of a set of sigma points and weights. returns the mean and covariance in a tuple.
    This works in conjunction with the UnscentedKF class.

    Parameters
    ----------
    sigmas: ndarray, of size (n, 2n+1), 2D array of sigma points.

    w_mean : ndarray [# sigmas per dimension], Weights for the mean.

    w_cov : ndarray [# sigmas per dimension], Weights for the covariance.

    noise_cov : ndarray, optional, noise matrix added to the final computed covariance matrix.

    mean_fn : callable (sigma_points, weights), optional

    residual_fn : callable (x, y)

    Returns
    -------
    x : ndarray [dimension], Mean of the sigma points after passing through the transform.

    P : ndarray, covariance of the sigma points after passing throgh the transform.

    """

    num_sig, num_st = len(sigmas), len(sigmas[0])
    if mean_fn is None:
        # new mean is just the sum of the sigmas * weight
        x = np.dot(w_mean, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k]),
    else:
        x = mean_fn(sigmas, w_mean)

    # new covariance is the sum of the outer product of the residuals times the weights
    if residual_fn is np.subtract or residual_fn is None:
        y = sigmas - x[np.newaxis, :] # shape = sigma.shape, 
        P = np.dot(y.T, np.dot(np.diag(w_cov), y)) # shape = (num_st * num_st)
    else:
        P = np.zeros((num_st, num_st))
        for k in range(num_sig):
            y = residual_fn(sigmas[k], x)
            P += w_cov[k] * np.outer(y, y)

    if noise_cov is not None:
        if np.isscalar(noise_cov):
            P += noise_cov * np.eye(P.shape[0])
        else:
            P += noise_cov

    return x, P

def output_with_each_sigma_point(
        model : nn.Module,
        params,
        model_input
        ):

    state_dicts = model.state_dict()
    idx = 0
    for p, k in zip(model.parameters(), model.state_dict().keys()):
        if p.requires_grad == True:
            l = len(p.flatten())
            state_dicts[k] = torch.nn.Parameter(torch.from_numpy(params[idx : idx + l].reshape(p.shape)))
            idx += l
    model.load_state_dict(state_dicts)

    # for i, t in data_loader:
    #     output = model(i)
    #     break
    # i, t = next(iter(data_loader))
    output = model(model_input)

    return output.flatten().detach().numpy()

class UKF(object):
    def __init__(self, dim_x, dim_y, fx, hx, ukf_params, x_mean_fn, y_mean_fn, init_x):
        # self.x = np.zeros(dim_x)
        self.x = np.array(init_x)

        self.P = np.eye(dim_x)
        self.x_pred = np.copy(self.x)
        self.P_pred = np.copy(self.P)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_y)
        self._dim_x = dim_x
        self._dim_y = dim_y
        self._ukf_params = ukf_params

        self.x_mean = x_mean_fn
        self.y_mean = y_mean_fn
        self._fx = fx
        self._hx = hx

        sig_pts = MerweSigmaPoints(dim_x, ukf_params)
        # weights for the means and covariances.
        self.sigma_points = sig_pts
        self.Wm, self.Wc = sig_pts.Wm, sig_pts.Wc
        self._num_sigmas = sig_pts.num_sigmas() # 2n + 1

        # sigma points transformed through f(x) and h(x). variables for efficiency so we don't recreate every update
        self.sigmas_f = np.zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = np.zeros((self._num_sigmas, self._dim_y))
        self.residual_x = np.subtract
        self.residual_y = np.subtract

        self.K = np.zeros((dim_x, dim_y)) # Kalman Gain
        self.innov = np.zeros(dim_y)
        self.y = np.array([[None] * dim_y]).T

        self.S = np.zeros((dim_y, dim_y))  # system uncertainty
        self.SI = np.zeros((dim_y, dim_y))  # inverse system uncertainty
        self.inv = np.linalg.inv

        # these will always be a copy of x, P after predict() is called
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()

        # these will always be a copy of x, P after update() is called
        self.x_updt = self.x.copy()
        self.P_updt = self.P.copy()

        self.datas = None
        self._hx_list = [deepcopy(self._hx) for _ in range(len(self.sigmas_f))]
        
        
    def predict(self):
        self.compute_process_sigmas()
        self.x, self.P = unscented_transform(self.sigmas_f, self.Wm, self.Wc, self.Q, self.x_mean, self.residual_x)
        
        # save prior
        self.x_pred = np.copy(self.x)
        self.P_pred = np.copy(self.P)

    def update(self, y, model_input):
        if y is None:
            self.y = np.array([[None] * self._dim_y]).T
            self.x_updt = self.x.copy()
            self.P_updt = self.P.copy()
            return

        if np.isscalar(self.R):
            R = np.eye(self._dim_y) * self.R
        else:
            R = self.R

        sigmas_h = []
        """
        [추천 라이브러리]
        - multiprocessing
        - pathos <- multiprocessing 상위호환(래퍼)
        - ray
        - jax 
        Method 1
        1. multi-threading으로 복수의 thread 생성
        2. thread마다 output_with.... 작업 할당
        3. 각 thread 계산 종료 후에 결과 취합 -> sigma_h로
        
        주의: 모델 용량 * 모델 개수 + 데이터 개수 * 데이터 용량 <= 램 크기 
        """
        # self._hx_list = [deepcopy(self._hx) for _ in range(len(self.sigmas_f))]
        model_input = torch.tile(model_input.unsqueeze(dim=0), (len(self.sigmas_f), 1, 1))
        p = ProcessPool(nodes=os.cpu_count())
        sigmas_h = p.map(output_with_each_sigma_point, self._hx_list, self.sigmas_f, model_input)

        # for s in self.sigmas_f:
        #     # pass neural network with sigma_points parameters
        #     p = multiprocessing.Process(target=output_with_each_sigma_point, args=(self._hx, s, model_input))
        #     p.start()
        #     sigmas_h.append(p)
        #     # sigmas_h.append(output_with_each_sigma_point(self._hx, s, model_input))

        # for i in sigmas_h:
        #     i.join()

        yp, self.S = unscented_transform(sigmas_h, self.Wm, self.Wc, R, self.y_mean, self.residual_y)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxy = self.cross_variance(self.x, yp, self.sigmas_f, sigmas_h)
        self.K = np.dot(Pxy, self.SI)  # Kalman gain

        self.y = self.residual_y(y, yp)  # residual

        # update Gaussian state estimate (x, P)
        self.x = self.x + np.dot(self.K, self.y)
        self.P = self.P - np.dot(self.K, np.dot(self.S, self.K.T))

        # save measurement and posterior state
        # self.y = deepcopy(y)
        self.x_updt = self.x.copy()
        self.P_updt = self.P.copy()

    def cross_variance(self, x, y, sigmas_f, sigmas_h):
        """
        Compute cross variance of the state `x` and measurement `y`.
        """

        Pxy = np.zeros((sigmas_f.shape[1], len(sigmas_h[0])))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = self.residual_x(sigmas_f[i], x)
            dy = self.residual_y(sigmas_h[i], y)
            Pxy += self.Wc[i] * np.outer(dx, dy)
        return Pxy

    def compute_process_sigmas(self):
        fx = self._fx
        # calculate sigma points for given mean and covariance
        sigmas = self.sigma_points.generate_sigmas(self.x, self.P)

        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = fx(s) # num_sigmax * dim_x

    def return_weight(self):
        return self.x_updt
    
    def return_loss(self):
        return np.sum(np.abs(self.y)) / self._dim_y
    
def fx(x):
    # Nothing to do here. Process Model is random walk
    return x

# def model_update_ukf(data_loader,ukf_params,model : nn.Module,max_iter : int = 40):
#     learnable_params = list()
#     for p, k in zip(model.parameters(), model.state_dict().keys()):
#         if p.requires_grad == True:
#             learnable_params.extend(model.state_dict()[k].flatten().tolist())
#     state_dicts = model.state_dict()
    
#     loss_history = list()

#     inp, tar = next(iter(data_loader))
#     if len(inp) > 50 :
#         idx = torch.linspace(0, len(inp)-1, 50, dtype=torch.int64)
#         inp = inp[idx]
#         tar = tar[idx]
#     tar = tar.flatten().detach().numpy()

#     ukf = UKF(dim_x = len(learnable_params), dim_y = len(tar.flatten()), 
#             fx = fx, hx = model, ukf_params=ukf_params, x_mean_fn=None, y_mean_fn=None, init_x=learnable_params)
#     ukf.P = 0.00001
#     ukf.Q = 0.00001
#     ukf.R = 0.00001

#     for _ in range(max_iter):
#         # print('Iteration: ', iter + 1)
#         # print('Initial State at this iteration', ukf.x)
#         ukf.predict()
#         ukf.update(tar, inp)
#         print('.', end='')
#         # print('Updated State at this iteration', ukf.x)

#         loss_history.append(ukf.return_loss())


#     learned_params = ukf.return_weight()

#     idx = 0
#     for p, k in zip(model.parameters(), model.state_dict().keys()):
#         if p.requires_grad == True:
#             l = len(p.flatten())
#             state_dicts[k] = torch.nn.Parameter(torch.from_numpy(learned_params[idx : idx + l].reshape(p.shape)))
#             idx += l
#     model.load_state_dict(state_dicts)
#     return model, loss_history, ukf