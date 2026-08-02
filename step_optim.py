import torch
import os
import logging
import math
import numpy as np
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint



class StepOptim(object):
    def __init__(self, ns):
        super().__init__()
        self.ns = ns
        self.T = 1.0 # t_T of diffusion sampling, for VP models, T=1.0; for EDM models, T=80.0

    def alpha(self, t):
        t = torch.as_tensor(t, dtype = torch.float64)
        return self.ns.marginal_alpha(t).detach().cpu().numpy()
    def sigma(self, t):
        return np.sqrt(1 - self.alpha(t) * self.alpha(t))
    def lambda_func(self, t):
        return np.log(self.alpha(t)/self.sigma(t))
    def H0(self, h):
        return np.exp(h) - 1
    def H1(self, h):
        return np.exp(h) * h - self.H0(h)
    def H2(self, h):
        return np.exp(h) * h * h - 2 * self.H1(h)
    def H3(self, h):
        return np.exp(h) * h * h * h - 3 * self.H2(h)
    def inverse_lambda(self, lamb):
        lamb = torch.as_tensor(lamb, dtype = torch.float64)
        return self.ns.inverse_lambda(lamb)
    def edm_sigma(self, t):
        return np.sqrt(1./(self.alpha(t)*self.alpha(t)) - 1)
    def edm_inverse_sigma(self, edm_sigma):
        alpha = 1 / (edm_sigma*edm_sigma+1).sqrt()
        sigma = alpha*edm_sigma
        lambda_t = np.log(alpha/sigma)
        t = self.inverse_lambda(lambda_t)
        return t



    def sel_lambdas_lof_obj(self, lambda_vec, eps):

        lambda_eps, lambda_T = self.lambda_func(eps).item(), self.lambda_func(self.T).item()
        lambda_vec_ext = np.concatenate((np.array([lambda_T]), lambda_vec, np.array([lambda_eps])))
        N = len(lambda_vec_ext) - 1

        hv = np.zeros(N)
        for i in range(N):
            hv[i] = lambda_vec_ext[i+1] - lambda_vec_ext[i]
        elv = np.exp(lambda_vec_ext)
        emlv_sq = np.exp(-2*lambda_vec_ext)
        alpha_vec = 1./np.sqrt(1+emlv_sq)
        sigma_vec = 1./np.sqrt(1+np.exp(2*lambda_vec_ext))
        data_err_vec = (sigma_vec**2)/alpha_vec
        # for pixel-space diffusion models, we empirically find (sigma_vec**1)/alpha_vec will be better

        truncNum = 3 # For NFEs <= 7, set truncNum = 3 to avoid numerical instability; for NFEs > 7, truncNum = 0
        res = 0. 
        c_vec = np.zeros(N)
        for s in range(N):
            if s in [0, N-1]:
                n, kp = s, 1 
                J_n_kp_0 = elv[n+1] - elv[n]
                res += abs(J_n_kp_0 * data_err_vec[n])
            elif s in [1, N-2]:
                n, kp = s-1, 2
                J_n_kp_0 = -elv[n+1] * self.H1(hv[n+1]) / hv[n]
                J_n_kp_1 = elv[n+1] * (self.H1(hv[n+1])+hv[n]*self.H0(hv[n+1])) / hv[n]
                if s >= truncNum:
                    c_vec[n] += data_err_vec[n] * J_n_kp_0
                    c_vec[n+1] += data_err_vec[n+1] * J_n_kp_1
                else:
                    res += np.sqrt((data_err_vec[n] * J_n_kp_0)**2 + (data_err_vec[n+1] * J_n_kp_1)**2)
            else:
                n, kp = s-2, 3  
                J_n_kp_0 = elv[n+2] * (self.H2(hv[n+2])+hv[n+1]*self.H1(hv[n+2])) / (hv[n]*(hv[n]+hv[n+1]))
                J_n_kp_1 = -elv[n+2] * (self.H2(hv[n+2])+(hv[n]+hv[n+1])*self.H1(hv[n+2])) / (hv[n]*hv[n+1])
                J_n_kp_2 = elv[n+2] * (self.H2(hv[n+2])+(2*hv[n+1]+hv[n])*self.H1(hv[n+2])+hv[n+1]*(hv[n]+hv[n+1])*self.H0(hv[n+2])) / (hv[n+1]*(hv[n]+hv[n+1]))
                if s >= truncNum:
                    c_vec[n] += data_err_vec[n] * J_n_kp_0
                    c_vec[n+1] += data_err_vec[n+1] * J_n_kp_1
                    c_vec[n+2] += data_err_vec[n+2] * J_n_kp_2
                else:
                    res += np.sqrt((data_err_vec[n] * J_n_kp_0)**2 + (data_err_vec[n+1] * J_n_kp_1)**2 + (data_err_vec[n+2] * J_n_kp_2)**2)
        res += sum(abs(c_vec))
        return res

    def get_ts_lambdas(self, N, eps, initType):
        # eps is t_0 of diffusion sampling, e.g. 1e-3 for VP models
        # initType: initTypes with '_origin' are baseline time step discretizations (without optimization)
        # initTypes without '_origin' are optimized time step discretizations with corresponding baseline
        # time step discretizations as initializations. For latent-space diffusion models, 'unif_t' is recommended.
        # For pixel-space diffusion models, 'unif' is recommended (which is logSNR initialization)

        lambda_eps, lambda_T = self.lambda_func(eps).item(), self.lambda_func(self.T).item()

        # constraints
        constr_mat = np.zeros((N, N-1)) 
        for i in range(N-1):
            constr_mat[i][i] = 1.
            constr_mat[i+1][i] = -1
        lb_vec = np.zeros(N)
        lb_vec[0], lb_vec[-1] = lambda_T, -lambda_eps

        ub_vec = np.zeros(N)
        for i in range(N):
            ub_vec[i] = np.inf
        linear_constraint = LinearConstraint(constr_mat, lb_vec, ub_vec)

        # initial vector
        if initType in ['unif', 'unif_origin']:
            lambda_vec_ext = torch.linspace(lambda_T, lambda_eps, N+1)
        elif initType in ['unif_t', 'unif_t_origin']:
            t_vec = torch.linspace(self.T, eps, N+1)
            lambda_vec_ext = self.lambda_func(t_vec)
        elif initType in ['edm', 'edm_origin']:
            rho = 7
            edm_sigma_min, edm_sigma_max = self.edm_sigma(eps).item(), self.edm_sigma(self.T).item()
            edm_sigma_vec = torch.linspace(edm_sigma_max**(1. / rho), edm_sigma_min**(1. / rho), N + 1).pow(rho)
            t_vec = self.edm_inverse_sigma(edm_sigma_vec)
            lambda_vec_ext = self.lambda_func(t_vec)
        elif initType in ['quad', 'quad_origin']:
            t_order = 2
            t_vec = torch.linspace(self.T**(1./t_order), eps**(1./t_order), N+1).pow(t_order)
            lambda_vec_ext = self.lambda_func(t_vec)
        else:
            print('InitType not found!')
            return 

        if initType in ['unif_origin', 'unif_t_origin', 'edm_origin', 'quad_origin']:
                lambda_res = lambda_vec_ext
                t_res = self.inverse_lambda(lambda_res).detach().clone()
        else: 
            lambda_vec_init = np.array(lambda_vec_ext[1:-1])
            res = minimize(self.sel_lambdas_lof_obj, lambda_vec_init, method='trust-constr', args=(eps), constraints=[linear_constraint], options={'verbose': 1})
            lambda_res = torch.tensor(np.concatenate((np.array([lambda_T]), res.x, np.array([lambda_eps]))))
            t_res = self.inverse_lambda(lambda_res).detach().clone()
        return t_res, lambda_res
    