from dpm_solver_v3 import NoiseScheduleVP, model_wrapper
import torch
import torch.nn.functional as F
import math
import numpy as np
import os

class UniPC:
    def __init__(
        self,
        noise_schedule,
        steps=10,
        t_start=None,
        t_end=None,
        skip_type="customed_time_karras",
        degenerated=False,
        use_afs = False,
        denoise_to_zero=False,
        need_fp16_discrete_method = False,
        ultilize_vae_in_fp16 = False,
        is_high_resoulution = True,
        device="cuda",
    ):
        self.device = device
        self.model = None
        self.noise_schedule = noise_schedule
        self.steps = steps if not use_afs else steps + 1
        self.use_afs = use_afs
        self.ultilize_vae_in_fp16 = ultilize_vae_in_fp16
        self.need_fp16_discrete_method = need_fp16_discrete_method
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        self.is_high_resolution = is_high_resoulution
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"

        
        # precompute timesteps
        if skip_type == "logSNR" or skip_type == "time_uniform" or skip_type == "time_quadratic" or skip_type == "customed_time_karras":
            self.timesteps = self.get_time_steps(skip_type
                                                 , t_T=t_T
                                                 , t_0=t_0
                                                 , N=steps
                                                 , device=device,denoise_to_zero=denoise_to_zero
                                                 , is_high_resolution=self.is_high_resolution)
        else:
            raise ValueError(f"Unsupported timestep strategy {skip_type}")
        self.lambda_T = self.timesteps[0].cpu().item()
        self.lambda_0 = self.timesteps[-1].cpu().item()

        # print("Time steps", self.timesteps)
        # print("LogSNR steps", self.noise_schedule.marginal_lambda(self.timesteps))

        # store high-order exponential coefficients (lazy)
        self.exp_coeffs = {}

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def append_zero(self, x):
        return torch.cat([x, x.new_zeros([1])])

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7., device='cpu', need_append_zero=True):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return self.append_zero(sigmas).to(device) if need_append_zero else sigmas.to(device)
    
    def sigma_to_t(self, sigma, quantize=None):
        quantize = False
        log_sigma = sigma.log()
        dists = log_sigma - self.noise_schedule.log_sigmas[:, None]
        if quantize:
            return dists.abs().argmin(dim=0).view(sigma.shape)
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.noise_schedule.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.noise_schedule.log_sigmas[low_idx], self.noise_schedule.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def get_time_steps(self, skip_type, t_T, t_0, N, device, denoise_to_zero=False, is_high_resolution=True):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
            return t
        elif skip_type == "customed_time_karras" and is_high_resolution:
            sigma_T = self.noise_schedule.sigmas[-1].cpu().item()
            sigma_0 = self.noise_schedule.sigmas[0].cpu().item()
            if N == 8:
                sigmas = self.get_sigmas_karras(12, sigma_0, sigma_T,rho=12.0, device=device)
                if not self.need_fp16_discrete_method:
                    ct_start, ct_end = self.noise_schedule.sigma_to_t(sigmas[0]), self.sigma_to_t(sigmas[10])
                    ct = self.get_sigmas_karras(9, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                    sigmas_ct = self.noise_schedule.get_special_sigmas_with_timesteps(ct).to(device=device)
                    real_ct = [self.noise_schedule.sigma_to_t(sigma).to('cpu') / 999 for sigma in sigmas_ct]
                else:
                    sigmas = self.get_sigmas_karras(8, sigma_0, sigma_T, rho=5.0, device=device)
                    ct_start, ct_end = self.noise_schedule.sigma_to_t(sigmas[0]), self.sigma_to_t(sigmas[6])
                    ct = self.get_sigmas_karras(8, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                    sigmas_ct = self.noise_schedule.get_special_sigmas_with_timesteps(ct).to(device=device)
                    tmp_t = [self.noise_schedule.sigma_to_t(sigma).to('cpu') for sigma in sigmas_ct]
                    real_ct = [ t / 999 for t in tmp_t]
            elif N == 5:
                sigmas = self.get_sigmas_karras(8, sigma_0, sigma_T, rho=5.0, device=device)
                if not self.need_fp16_discrete_method:
                    sigmas = self.get_sigmas_karras(12, sigma_0, sigma_T,rho=12.0, device=device)
                    ct_start, ct_end = self.noise_schedule.sigma_to_t(sigmas[0]), self.sigma_to_t(sigmas[9])
                    ct = self.get_sigmas_karras(6, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                    sigmas_ct = self.noise_schedule.get_special_sigmas_with_timesteps(ct).to(device=device)
                    real_ct = [self.noise_schedule.sigma_to_t(sigma).to('cpu') / 999 for sigma in sigmas_ct]
                else:
                    ct_start, ct_end = self.noise_schedule.sigma_to_t(sigmas[0]), self.sigma_to_t(sigmas[6])
                    ct = self.get_sigmas_karras(5, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                    sigmas_ct = self.noise_schedule.get_special_sigmas_with_timesteps(ct).to(device=device)
                    real_ct = [self.noise_schedule.sigma_to_t(sigma).to('cpu') / 999 for sigma in sigmas_ct]
            elif N == 6:
                sigmas = self.get_sigmas_karras(8, sigma_0, sigma_T, rho=5.0, device=device)
                if not self.need_fp16_discrete_method:
                    sigmas = self.get_sigmas_karras(12, sigma_0, sigma_T,rho=12.0, device=device)
                    ct_start, ct_end = self.noise_schedule.sigma_to_t(sigmas[0]), self.sigma_to_t(sigmas[10])
                    ct = self.get_sigmas_karras(7, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                    sigmas_ct = self.noise_schedule.get_special_sigmas_with_timesteps(ct).to(device=device)
                    real_ct = [self.noise_schedule.sigma_to_t(sigma).to('cpu') / 999 for sigma in sigmas_ct]
                else:
                    if denoise_to_zero:
                        ct_start, ct_end = self.noise_schedule.sigma_to_t(sigmas[0]), self.sigma_to_t(sigmas[6])
                        ct = self.get_sigmas_karras(6, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                        sigmas_ct = self.noise_schedule.get_special_sigmas_with_timesteps(ct).to(device=device)
                        real_ct = [self.noise_schedule.sigma_to_t(sigma).to('cpu') / 999 for sigma in sigmas_ct]
                        real_ct.append(torch.tensor(t_0).to(dtype=real_ct[-1].dtype,device='cpu'))
                    else:
                        sigmas = self.get_sigmas_karras(12, sigma_0, sigma_T, rho=7.0, device=device)
                        ct_start, ct_end = self.noise_schedule.sigma_to_t(sigmas[0]), self.sigma_to_t(sigmas[7])
                        ct = self.get_sigmas_karras(7, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                        sigmas_ct = self.noise_schedule.get_special_sigmas_with_timesteps(ct).to(device=device)
                        real_ct = [self.noise_schedule.sigma_to_t(sigma).to('cpu') / 999 for sigma in sigmas_ct]
            elif N == 7:
                sigmas = self.get_sigmas_karras(8, sigma_0, sigma_T, rho=5.0, device=device)
                if not self.need_fp16_discrete_method:
                    ct_start, ct_end = self.noise_schedule.sigma_to_t(sigmas[0]), self.sigma_to_t(sigmas[6])
                    ct = self.get_sigmas_karras(8, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                    sigmas_ct = self.noise_schedule.get_special_sigmas_with_timesteps(ct).to(device=device)
                    real_ct = [self.noise_schedule.sigma_to_t(sigma).to('cpu') / 999 for sigma in sigmas_ct]
                else:
                    if denoise_to_zero:
                        ct_start, ct_end = self.noise_schedule.sigma_to_t(sigmas[0]), self.sigma_to_t(sigmas[6])
                        ct = self.get_sigmas_karras(7, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                        sigmas_ct = self.noise_schedule.get_special_sigmas_with_timesteps(ct).to(device=device)
                        real_ct = [self.noise_schedule.sigma_to_t(sigma).to('cpu') / 999 for sigma in sigmas_ct]
                        real_ct.append(torch.tensor(t_0).to(dtype=real_ct[-1].dtype,device='cpu'))
            # if denoise_to_zero:
            #     real_ct.append(torch.tensor(t_0).to(dtype=real_ct[-1].dtype,device='cpu'))
            
            if self.use_afs:
                tmp_t = (real_ct[0] + real_ct[1]) / 2
                real_ct.insert(1, tmp_t)
            none_k_ct = torch.from_numpy(np.array(real_ct)).to(device) 
            return none_k_ct#real_ct
        elif skip_type == "customed_time_karras" and not is_high_resolution:
            sigma_T = self.noise_schedule.sigmas[-1].cpu().item()
            sigma_0 = self.noise_schedule.sigmas[0].cpu().item()
            if N == 8:
                sigmas = self.get_sigmas_karras(12, sigma_0, sigma_T, rho=7.0, device=device)
                ct_start, ct_end = self.noise_schedule.sigma_to_t(sigmas[0]), self.sigma_to_t(sigmas[9])
                ct = self.get_sigmas_karras(9, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                sigmas_ct = self.noise_schedule.get_special_sigmas_with_timesteps(ct).to(device=device)
                real_ct = [self.noise_schedule.sigma_to_t(sigma).to('cpu') / 999 for sigma in sigmas_ct]
            elif N == 5:
                sigmas = self.get_sigmas_karras(8, sigma_0, sigma_T, rho=5.0, device=device)
                ct_start, ct_end = self.noise_schedule.sigma_to_t(sigmas[0]), self.sigma_to_t(sigmas[6])
                ct = self.get_sigmas_karras(6, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                sigmas_ct = self.noise_schedule.get_special_sigmas_with_timesteps(ct).to(device=device)
                real_ct = [self.noise_schedule.sigma_to_t(sigma).to('cpu') / 999 for sigma in sigmas_ct]
            elif N == 6:
                sigmas = self.sigmas = self.get_sigmas_karras(8, sigma_0, sigma_T, rho=5.0, device=device)
                ct_start, ct_end = self.noise_schedule.sigma_to_t(sigmas[0]), self.sigma_to_t(sigmas[6])
                ct = self.get_sigmas_karras(7, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                sigmas_ct = self.noise_schedule.get_special_sigmas_with_timesteps(ct).to(device=device)
                real_ct = [self.noise_schedule.sigma_to_t(sigma).to('cpu') / 999 for sigma in sigmas_ct]
            none_k_ct = torch.from_numpy(np.array(real_ct)).to(device) 
            return none_k_ct#real_ct
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type)
            )


    def multistep_uni_pc_update(self, x, model_prev_list:list, t_prev_list: list, t, order, **kwargs):
        if len(model_prev_list) == 0 or len(t_prev_list) == 0:
            return None, None
        if len(t.shape) == 0:
            t = t.view(-1)
        if True:#'bh' in self.variant:
            return self.multistep_uni_pc_bh_update(x, model_prev_list, t_prev_list, t, order, **kwargs)
        else:
            # assert self.variant == 'vary_coeff'
            return self.multistep_uni_pc_vary_update(x, model_prev_list, t_prev_list, t, order, **kwargs)

    def multistep_uni_pc_sde_update(self, x, model_prev_list:list, t_prev_list: list, t, order, level = 1.0, **kwargs):
        if len(model_prev_list) == 0 or len(t_prev_list) == 0:
            return None, None
        if len(t.shape) == 0:
            t = t.view(-1)
        if True:#'bh' in self.variant:
            return self.multistep_uni_pc_bh_sde_update(x, model_prev_list, t_prev_list, t, level=level, order= order, **kwargs)
        else:
            # assert self.variant == 'vary_coeff'
            return self.multistep_uni_pc_vary_update(x, model_prev_list, t_prev_list, t, order, **kwargs)

    def multistep_uni_pc_bh_update(self, x, model_prev_list, t_prev_list, t, order, x_t=None, use_corrector=True):
        # print(f'using unified predictor-corrector with order {order} (solver type: B(h))')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)
        dims = x.dim()

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = h[0]
        h_phi_1 = torch.expm1(hh) # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if True:
            B_h = hh
        else:
            B_h = torch.expm1(hh)
            
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i 

        R = torch.stack(R)
        b = torch.tensor(b, device=x.device)

        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1) # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if use_corrector:
            # print('using corrector')
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_t = None
        
        x_t_ = (
            expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims) * x
            - expand_dims(sigma_t * h_phi_1, dims) * model_prev_0
        )
        if x_t is None:
            if use_predictor:
                pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - expand_dims(sigma_t * B_h, dims) * pred_res

        if use_corrector:
            model_t = self.noise_prediction_fn(x_t, t)
            if D1s is not None:
                corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = (model_t - model_prev_0)
            x_t = x_t_ - expand_dims(sigma_t * B_h, dims) * (corr_res + rhos_c[-1] * D1_t)

        return x_t, model_t
    
    def multistep_uni_pc_bh_sde_update(self, x, model_prev_list, t_prev_list, t, order, level = 0, x_t=None, use_corrector=True):
        # print(f'using unified predictor-corrector with order {order} (solver type: B(h))')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)
        dims = x.dim()

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0
        z = torch.randn(x.shape, device=self.device)
        z = sigma_t * torch.sqrt(torch.expm1(2.0 * h[0])) * z

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = h[0]
        h_phi_1 = torch.expm1(hh) # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if True:
            B_h = hh
        else:
            B_h = torch.expm1(hh)
            
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i 

        R = torch.stack(R)
        b = torch.tensor(b, device=x.device)

        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1) # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if use_corrector:
            # print('using corrector')
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_t = None
        
        x_t_ = (
            expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims) * x
            - expand_dims(sigma_t * h_phi_1, dims) * (1 + level) * model_prev_0
        )
        if x_t is None:
            if use_predictor:
                pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
            else:
                pred_res = 0
            
            x_t_p = (
                expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                - expand_dims(sigma_t * h_phi_1, dims) * model_prev_0
            )
            x_t = x_t_p - expand_dims(sigma_t * B_h, dims) * pred_res

        if use_corrector:
            model_t = self.noise_prediction_fn(x_t, t)
            if D1s is not None:
                corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = (model_t - model_prev_0)
            x_t = x_t_ - (1 + level)  * expand_dims(sigma_t * B_h, dims) * (corr_res + rhos_c[-1] * D1_t) + z * level

        return x_t, model_t
    
    
    def multistep_uni_pc_vary_update(self, x, model_prev_list, t_prev_list, t, order, use_corrector=True):
        # print(f'using unified predictor-corrector with order {order} (solver type: vary coeff)')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)
        dims = x.dim()
        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        K = len(rks)
        # build C matrix
        C = []

        col = torch.ones_like(rks)
        for k in range(1, K + 1):
            C.append(col)
            col = col * rks / (k + 1) 
        C = torch.stack(C, dim=1)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1) # (B, K)
            C_inv_p = torch.linalg.inv(C[:-1, :-1])
            A_p = C_inv_p

        if use_corrector:
            # print('using corrector')
            C_inv = torch.linalg.inv(C)
            A_c = C_inv

        hh = h
        h_phi_1 = torch.expm1(hh)
        h_phi_ks = []
        factorial_k = 1
        h_phi_k = h_phi_1
        for k in range(1, K + 2):
            h_phi_ks.append(h_phi_k)
            h_phi_k = h_phi_k / hh - 1 / factorial_k
            factorial_k *= (k + 1)

        model_t = None
        if True:
            log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
            x_t_ = (
                expand_dims((torch.exp(log_alpha_t - log_alpha_prev_0)),dims) * x
                - expand_dims((sigma_t * h_phi_1),dims) * model_prev_0
            )
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - expand_dims(sigma_t * h_phi_ks[k + 1],dims) * torch.einsum('bkchw,k->bchw', D1s, A_p[k])
            # now corrector
            if use_corrector:
                model_t = self.noise_prediction_fn(x_t, t)
                D1_t = (model_t - model_prev_0)
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    x_t = x_t - expand_dims(sigma_t * h_phi_ks[k + 1],dims) * torch.einsum('bkchw,k->bchw', D1s, A_c[k][:-1])
                x_t = x_t - expand_dims(sigma_t * h_phi_ks[K],dims) * (D1_t * A_c[k][-1])
        return x_t, model_t

    def sample(
        self,
        x,
        model_fn,
        order,
        use_corrector,
        lower_order_final,
        start_free_u_step=None,
        free_u_apply_callback=None,
        free_u_stop_callback=None,
        npnet_x = None,
        npnet_scale = None,
        half=False,
        return_intermediate=False,
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        steps = self.steps
        vec_t = self.timesteps[0].expand((x.shape[0]))
        if free_u_stop_callback is not None:
            free_u_stop_callback()
        if start_free_u_step is not None and 0 == start_free_u_step and free_u_apply_callback is not None:
            free_u_apply_callback()
            has_called_free_u = True
        if not self.use_afs:
            fir_output = self.noise_prediction_fn(x, vec_t)
        else:
            fir_output = x  # ultilize npnet there in the future
            if npnet_x is not None and npnet_scale is not None:
                fir_output = npnet_x 
                # fir_output = fir_output - npnet_scale * (npnet_out - fir_output) #guidance_scale * (noise - noise_uncond)
                x = fir_output.clone().detach().to(fir_output.device)
            

        model_prev_list = [fir_output]
        full_cache = [fir_output]
        t_prev_list = [vec_t]
        has_called_free_u = False
        for init_order in range(1, order):
            if start_free_u_step is not None and init_order == start_free_u_step and free_u_apply_callback is not None and (not has_called_free_u):
                free_u_apply_callback()
                has_called_free_u = True
            vec_t = self.timesteps[init_order].expand(x.shape[0])
            x, model_x = self.multistep_uni_pc_update(x, model_prev_list, t_prev_list, vec_t, init_order, use_corrector=True)
            if model_x is None:
                model_x = self.noise_prediction_fn(x, vec_t)
                x = model_x.clone().detach().to(torch.float32).to(model_x.device)
            full_cache.append(x)
            model_prev_list.append(model_x)
            t_prev_list.append(vec_t)
        
        for step in range(order, steps + 1):
            if start_free_u_step is not None and step == start_free_u_step and free_u_apply_callback is not None and (not has_called_free_u):
                free_u_apply_callback()
            vec_t = self.timesteps[step].expand(x.shape[0])
            if lower_order_final:
                step_order = min(order, steps + 1 - step)
            else:
                step_order = order
            # print('this step order:', step_order)
            if step == steps:
                # print('do not run corrector at the last step')
                use_corrector = False
            else:
                use_corrector = True
            x, model_x =  self.multistep_uni_pc_update(x, model_prev_list, t_prev_list, vec_t, step_order, use_corrector=use_corrector)
            for i in range(order - 1):
                t_prev_list[i] = t_prev_list[i + 1]
                model_prev_list[i] = model_prev_list[i + 1]
            t_prev_list[-1] = vec_t
                    # We do not need to evaluate the final model value.
            full_cache.append(x)
            if step < steps:
                if model_x is None:
                    model_x = self.noise_prediction_fn(x, vec_t)
                model_prev_list[-1] = model_x
        return x, full_cache
    def sample_mix(
        self,
        x,
        model_fn,
        order,
        use_corrector,
        lower_order_final,
        start_free_u_step=None,
        free_u_apply_callback=None,
        free_u_stop_callback=None,
        noise_level = 0.1,
        half=False,
        return_intermediate=False,
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        steps = self.steps
        vec_t = self.timesteps[0].expand((x.shape[0]))
        fir_output = self.noise_prediction_fn(x, vec_t)
        model_prev_list = [fir_output]
        full_cache = [fir_output]
        t_prev_list = [vec_t]
        has_called_free_u = False
        if free_u_stop_callback is not None:
            free_u_stop_callback()
        for init_order in range(1, order):
            if start_free_u_step is not None and init_order == start_free_u_step and free_u_apply_callback is not None:
                free_u_apply_callback()
                has_called_free_u = True
            vec_t = self.timesteps[init_order].expand(x.shape[0])
            if start_free_u_step is not None and init_order >= start_free_u_step and free_u_apply_callback is not None:
                x, model_x = self.multistep_uni_pc_sde_update(x
                                                              , model_prev_list
                                                              , t_prev_list
                                                              , vec_t
                                                              , init_order
                                                              , use_corrector=True
                                                              ,level=noise_level)
            else:
                x, model_x = self.multistep_uni_pc_sde_update(x
                                                              , model_prev_list
                                                              , t_prev_list
                                                              , vec_t
                                                              , init_order
                                                              , use_corrector=True
                                                              ,level=0.0)
            if model_x is None:
                model_x = self.noise_prediction_fn(x, vec_t)
                x = model_x.clone().detach().to(torch.float32).to(model_x.device)
            full_cache.append(x)
            model_prev_list.append(model_x)
            t_prev_list.append(vec_t)
        
        if free_u_stop_callback is not None:
            free_u_stop_callback()
        for step in range(order, steps + 1):
            if start_free_u_step is not None and step == start_free_u_step and free_u_apply_callback is not None and (not has_called_free_u):
                free_u_apply_callback()
            vec_t = self.timesteps[step].expand(x.shape[0])
            if lower_order_final:
                step_order = min(order, steps + 1 - step)
            else:
                step_order = order
            # print('this step order:', step_order)
            if step == steps:
                # print('do not run corrector at the last step')
                use_corrector = False
            else:
                use_corrector = True
            if start_free_u_step is not None and step >= start_free_u_step and free_u_apply_callback is not None:
                x, model_x =  self.multistep_uni_pc_sde_update(x
                                                               , model_prev_list
                                                               , t_prev_list
                                                               , vec_t
                                                               , step_order
                                                               , use_corrector=use_corrector
                                                               , level=noise_level)
            else:
                x, model_x =  self.multistep_uni_pc_sde_update(x
                                                               , model_prev_list
                                                               , t_prev_list
                                                               , vec_t
                                                               , step_order
                                                               , use_corrector=use_corrector
                                                               , level=0.0)
            for i in range(order - 1):
                t_prev_list[i] = t_prev_list[i + 1]
                model_prev_list[i] = model_prev_list[i + 1]
            t_prev_list[-1] = vec_t
                    # We do not need to evaluate the final model value.
            full_cache.append(x)
            if step < steps:
                if model_x is None:
                    model_x = self.noise_prediction_fn(x, vec_t)
                model_prev_list[-1] = model_x
        return x, full_cache
        
        


#############################################################
# other utility functions
#############################################################

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]
