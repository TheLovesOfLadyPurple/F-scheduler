import argparse, os, sys, glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import accelerate
import torchsde

import diffusers
from pycocotools.coco import COCO
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)
from huggingface_hub import login, hf_hub_download
import shutil
from SVDNoiseUnet import NPNet128
import functools
import random
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import json
import subprocess
import os
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

# New helper to load a list-of-dicts preference JSON
# JSON schema: [ { 'human_preference': [int], 'prompt': str, 'file_path': [str] }, ... ]
def load_preference_json(json_path: str) -> list[dict]:
    """Load records from a JSON file formatted as a list of preference dicts."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# New helper to extract just the prompts from the preference JSON
# Returns a flat list of all 'prompt' values

def extract_prompts_from_pref_json(json_path: str) -> list[str]:
    """Load a JSON of preference records and return only the prompts."""
    records = load_preference_json(json_path)
    return [rec['prompt'] for rec in records]

# Example usage:
# prompts = extract_prompts_from_pref_json("path/to/preference.json")
# print(prompts)

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu',need_append_zero = True):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device) if need_append_zero else sigmas.to(device)

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    
    def prepare_sdxl_pipeline_step_parameter(self, pipe, prompts, need_cfg, device):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            device=device,
            do_classifier_free_guidance=need_cfg,
        )
    # timesteps = pipe.scheduler.timesteps
    
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = pooled_prompt_embeds.to(device)
        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)
        text_encoder_projection_dim = None
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        if pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
        passed_add_embed_dim = (
            pipe.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = pipe.unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype)
        add_time_ids = add_time_ids.to(device)
        negative_add_time_ids = add_time_ids

        if need_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        ret_dict = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids
        }
        return prompt_embeds, ret_dict

    
    def get_golden_noised(self, x, sigma,sigma_nxt, prompt, cond_scale,need_distill_uncond=False,tmp_list = [], uncond_list = [],noise_training_list={}):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        sigma_nxt = torch.cat([sigma_nxt] * 2)
        prompt_embeds, cond_kwargs = self.prepare_sdxl_pipeline_step_parameter(self.inner_model.pipe, prompt, need_cfg=True, device=self.inner_model.pipe.device)
        _, ret = self.inner_model.get_customed_golden_noise(x_in
                                                         , cond_scale
                                                         , sigma_in, sigma_nxt
                                                         , True
                                                         , noise_training_list=noise_training_list
                                                         , encoder_hidden_states=prompt_embeds.to(device=x.device, dtype=x.dtype)
                                                         , added_cond_kwargs=cond_kwargs).chunk(2)

        return ret

    def forward(self, x, sigma, prompt, cond_scale,need_distill_uncond=False,tmp_list = [], uncond_list = []):
        prompt_embeds, cond_kwargs = self.prepare_sdxl_pipeline_step_parameter(self.inner_model.pipe, prompt, need_cfg=True, device=self.inner_model.pipe.device)
        # w = cond_scale * x.new_ones([x.shape[0]])
        # w_embedding = guidance_scale_embedding(w, embedding_dim=self.inner_model.inner_model.config["time_cond_proj_dim"])
        # w_embedding = w_embedding.to(device=x.device, dtype=x.dtype)
        # # t = self.inner_model.sigma_to_t(sigma)
        # cond = self.inner_model(
        #     x,
        #     sigma,
        #     timestep_cond=w_embedding,
        #     encoder_hidden_states=cond.to(device=x.device, dtype=x.dtype),
        # )
        # return cond 
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        # cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in
                                        , sigma_in
                                        , tmp_list
                                        , encoder_hidden_states=prompt_embeds.to(device=x.device, dtype=x.dtype)
                                        , added_cond_kwargs=cond_kwargs).chunk(2)
        if need_distill_uncond:
            uncond_list.append(uncond)
        return prompt_embeds, uncond + (cond - uncond) * cond_scale
            
    
class DiscreteSchedule(nn.Module):
    """A mapping between continuous noise levels (sigmas) and a list of discrete noise
    levels."""

    def __init__(self, sigmas, quantize):
        super().__init__()
        self.register_buffer('sigmas', sigmas)
        self.register_buffer('log_sigmas', sigmas.log())
        self.quantize = quantize

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def get_sigmas(self, n=None):
        if n is None:
            return append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma, quantize=None):
        quantize = self.quantize if quantize is None else quantize
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        if quantize:
            return dists.abs().argmin(dim=0).view(sigma.shape)
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()

class DiscreteEpsDDPMDenoiser(DiscreteSchedule):
    """A wrapper for discrete schedule DDPM models that output eps (the predicted
    noise)."""

    def __init__(self, pipe, alphas_cumprod, quantize = False):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
        self.pipe = pipe
        self.inner_model = pipe.unet
        # self.alphas_cumprod = alphas_cumprod.flip(0)
        # Prepare a reversed version of alphas_cumprod for backward scheduling
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        # self.register_buffer('alphas_cumprod_prev', append_zero(alphas_cumprod[:-1]))
        self.sigma_data = 1.

    def get_scalings(self, sigma):
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_out, c_in

    def get_eps(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)

    def get_alphact_and_sigma(self, timesteps, x_0, noise):
        high_idx = torch.ceil(timesteps).int()
        low_idx = torch.floor(timesteps).int()
        
        nxt_ts = timesteps - timesteps.new_ones(timesteps.shape[0])
        
        w = (timesteps - low_idx) / (high_idx - low_idx)
        
        beta_1 = torch.tensor([1e-4],dtype=torch.float32) 
        beta_T = torch.tensor([0.02],dtype=torch.float32)
        ddpm_max_step = torch.tensor([1000.0],dtype=torch.float32)
        
        beta_t: torch.Tensor = (beta_T - beta_1) / ddpm_max_step  * timesteps + beta_1
        beta_t_prev: torch.Tensor = (beta_T - beta_1) / ddpm_max_step  * nxt_ts + beta_1
        
        alpha_t = beta_t.new_ones(beta_t.shape[0]) - beta_t
        alpha_t_prev = beta_t.new_ones(beta_t.shape[0]) - beta_t_prev
        
        dir_xt = (1. - alpha_t_prev).sqrt() * noise
        x_prev = alpha_t_prev.sqrt() * x_0 + dir_xt + noise
        
        alpha_cumprod_t_floor = self.alpha_cumprods[low_idx]
        alpha_cumprod_t = (alpha_cumprod_t_floor * alpha_t) #.unsqueeze(1)
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sigmas = torch.sqrt(alpha_cumprod_t.new_ones(alpha_cumprod_t.shape[0]) - alpha_cumprod_t)
        
        # Fix broadcasting
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t[:, None, None]
        sigmas = sigmas[:, None, None]
        return alpha_cumprod_t, sigmas

    def get_c_ins(self,sigmas): # use to adjust loss
        ret = list()
        for sigma in sigmas:
            _, c_in = self.get_scalings(sigma=sigma)
            ret.append(c_in)
        return ret
    
    # def predicted_origin(model_output, timesteps, sample, alphas, sigmas, prediction_type = "epsilon"):
    #     if prediction_type == "epsilon":
    #         sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    #         alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    #         pred_x_0 = (sample - sigmas * model_output) / alphas
    #     elif prediction_type == "v_prediction":
    #         sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    #         alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    #         pred_x_0 = alphas * sample - sigmas * model_output
    #     else:
    #         raise ValueError(f"Prediction type {prediction_type} currently not supported.")
    #     return pred_x_0
    
    def get_customed_golden_noise(self
                                  , input
                                  , unconditional_guidance_scale:float
                                  , sigma
                                  , sigma_nxt
                                  ,  need_cond = True
                                  , noise_training_list = {}
                                  , **kwargs):
        """User should ensure the input is a pure noise.  
        It's a customed golden noise, not the one purposed in the paper.
        Maybe the one purposed in the paper should be implemented in the future."""
        c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        if need_cond:
            _, tmp_img = (input * c_in).chunk(2)
        else :
            tmp_img = input * c_in
        # print(tmp_img.max())
        # tmp_list.append(tmp_img)
        eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs).sample
        x_0 = input + eps * c_out
        # normal_form_input = input * c_in
        x_0_uncond, x_0 = x_0.chunk(2)
        x_0 = x_0_uncond + unconditional_guidance_scale * (x_0 - x_0_uncond)
        x_0 = torch.cat([x_0] * 2)
        
        
        t, t_next = t_fn(sigma), t_fn(sigma_nxt)
        h = t_next - t
        
        x = (append_dims(sigma_fn(t_next) / sigma_fn(t),input.ndim)) * input - append_dims((-h).expm1(),input.ndim) * x_0
        
        c_out_2, c_in_2 = [append_dims(x, input.ndim) for x in self.get_scalings(sigma_nxt)]
        
        
        # e_t_uncond_ret, e_t_ret = self.get_eps(x * c_in_2, self.sigma_to_t(sigma_nxt), **kwargs).sample.chunk(2)
        eps_ret = self.get_eps(x * c_in_2, self.sigma_to_t(sigma_nxt), **kwargs).sample
        org_golden_noise = False
        x_1 = x + eps_ret * c_out_2
        if org_golden_noise:
            ret = (x + append_dims((-h).expm1(),input.ndim) * x_1) / (append_dims(sigma_fn(t_next) / sigma_fn(t),input.ndim))
        else :
            e_t_uncond_ret, e_t_ret = eps_ret.chunk(2)
            e_t_ret = e_t_uncond_ret + 1.0 * (e_t_ret - e_t_uncond_ret)
            e_t_ret = torch.cat([e_t_ret] * 2)
            ret = x_0 + e_t_ret * append_dims(sigma,input.ndim)
        
        noise_training_list['org_noise'] = input * c_in
        noise_training_list['golden_noise'] = ret * c_in
        # noise_training_list.append(tmp_dict)
        return ret

        # timesteps = self.sigma_to_t(sigma)

        # high_idx = torch.ceil(timesteps).int().to(input.device)
        # low_idx = torch.floor(timesteps).int().to(input.device)

        # nxt_ts = (timesteps - timesteps.new_ones(timesteps.shape[0])).to(input.device) 
        
        # w = (timesteps - low_idx) / (high_idx - low_idx)
        
        # beta_1 = torch.tensor([1e-4],dtype=torch.float32).to(input.device)  
        # beta_T = torch.tensor([0.02],dtype=torch.float32).to(input.device) 
        # ddpm_max_step = torch.tensor([1000.0],dtype=torch.float32).to(input.device) 
        
        # beta_t: torch.Tensor = (beta_T - beta_1) / ddpm_max_step  * timesteps + beta_1
        # beta_t_prev: torch.Tensor = (beta_T - beta_1) / ddpm_max_step  * nxt_ts + beta_1
        
        # alpha_t = beta_t.new_ones(beta_t.shape[0]) - beta_t
        # alpha_t = append_dims(alpha_t, e_t.ndim)
        # alpha_t_prev = beta_t_prev.new_ones(beta_t_prev.shape[0]) - beta_t_prev
        # alpha_t_prev = append_dims(alpha_t_prev, e_t.ndim)
        # alpha_cumprod_t_floor = self.alphas_cumprod[low_idx]
        # alpha_cumprod_t_floor = append_dims(alpha_cumprod_t_floor, e_t.ndim)
        # alpha_cumprod_t:torch.Tensor = (alpha_cumprod_t_floor * alpha_t) #.unsqueeze(1)
        # alpha_cumprod_t_prev:torch.Tensor = (alpha_cumprod_t_floor * alpha_t_prev) #.unsqueeze(1)

        # sqrt_one_minus_alphas = (1 - alpha_cumprod_t).sqrt()
        
        # dir_xt = (1. - alpha_cumprod_t_prev).sqrt() * e_t
        # x_prev = alpha_cumprod_t_prev.sqrt() * x_0 + dir_xt
        
        # e_t_uncond_ret, e_t_ret = self.get_eps(x_prev, nxt_ts, **kwargs).sample.chunk(2)
        # e_t_ret = e_t_uncond_ret + 1.0 * (e_t_ret - e_t_uncond_ret)
        # e_t_ret = torch.cat([e_t_ret] * 2)
        # x_ret = alpha_t.sqrt() * x_0 + sqrt_one_minus_alphas * e_t_ret
        
        # return x_ret
    
    def forward(self, input, sigma, tmp_list=[], need_cond = True, **kwargs):
        # c_out_1, c_in_1 = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        # if need_cond:
        #     tmp_img = input * c_in_1
        # else :
        #     tmp_img = input * c_in_1
        # tmp_list.append(tmp_img)
        # timestep = self.sigma_to_t(sigma)
        # eps = self.get_eps(sample = input * c_in_1, timestep = timestep, **kwargs).sample
        # c_skip, c_out = self.scalings_for_boundary_conditions(timestep=self.sigma_to_t(sigma))
        # # return (input + eps * c_out_1) * c_out + input * c_in_1 * c_skip
        # return (input + eps * c_out_1)
        c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        if need_cond:
            _, tmp_img = (input * c_in).chunk(2)
        else :
            tmp_img = input * c_in
        # print(tmp_img.max())
        eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs).sample
        tmp_x0 = input + eps * c_out 
        tmp_dict = {'tmp_z': tmp_img, 'tmp_x0': tmp_x0}
        tmp_list.append(tmp_dict)
        return tmp_x0 #input + eps * c_out
    
    def get_special_sigmas_with_timesteps(self,timesteps):
        low_idx, high_idx, w = np.minimum(np.floor(timesteps),999), np.minimum(np.ceil(timesteps),999), torch.from_numpy( timesteps - np.floor(timesteps))
        self.alphas_cumprod = self.alphas_cumprod.to('cpu')
        alphas = (1 - w) * self.alphas_cumprod[low_idx] + w * self.alphas_cumprod[high_idx]
        return ((1 - alphas) / alphas) ** 0.5

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""
    def __init__(self, x, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]

class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """
    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()

@torch.no_grad()
def sample_euler(model
                 , x
                 , sigmas
                 , extra_args=None
                 , callback=None
                 , disable=None
                 , s_churn=0.
                 , s_tmin=0.
                 , s_tmax=float('inf')
                 , tmp_list=[]
                 , uncond_list=[]
                 , need_distill_uncond=False
                 , start_free_step = 1
                 , noise_training_list={}
                 , s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    intermediates = {'x_inter': [x],'pred_x0': []}
    register_free_upblock2d(model.inner_model.pipe, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
    register_free_crossattn_upblock2d(model.inner_model.pipe, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
    for i in trange(len(sigmas) - 1, disable=disable):
        if i == start_free_step:
            register_free_upblock2d(model.inner_model.pipe, b1=1.3, b2=1.4, s1=0.9, s2=0.2)
            register_free_crossattn_upblock2d(model.inner_model.pipe, b1=1.3, b2=1.4, s1=0.9, s2=0.2)
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        prompt_embeds, denoised = model(x, sigmas[i] * s_in, tmp_list=tmp_list,need_distill_uncond=need_distill_uncond,uncond_list=uncond_list, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
        intermediates['pred_x0'].append(denoised)
        intermediates['x_inter'].append(x)
    return prompt_embeds, intermediates, x

@torch.no_grad()
def sample_heun(model
                , x
                , sigmas
                , extra_args=None
                , callback=None
                , disable=None
                , s_churn=0.
                , s_tmin=0.
                , s_tmax=float('inf')
                , tmp_list=[]
                , uncond_list=[]
                , need_distill_uncond=False
                , noise_training_list={}
                , s_noise=1.):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    intermediates = {'x_inter': [x],'pred_x0': []}
    register_free_upblock2d(model.inner_model.pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
    register_free_crossattn_upblock2d(model.inner_model.pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        prompt_embeds, denoised = model(x, sigmas[i] * s_in, tmp_list=tmp_list,need_distill_uncond=need_distill_uncond,uncond_list=uncond_list, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            _, denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
            intermediates['pred_x0'].append(denoised_2)
            intermediates['x_inter'].append(x)
    return prompt_embeds, intermediates, x

@torch.no_grad()
def sample_dpmpp_ode(model
                     , x
                     , sigmas
                     , need_golden_noise = False
                     , start_free_step = 1
                     , extra_args=None, callback=None
                     , disable=None,tmp_list=[]
                     , need_distill_uncond=False
                     , need_raw_noise=False
                     , uncond_list=[]
                     , noise_training_list={}):
    """DPM-Solver++."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    if need_raw_noise:
        x = model.get_golden_noised(x=x,sigma=sigmas[0] * s_in, sigma_nxt=(sigmas[0] - 0.28) * s_in, noise_training_list=noise_training_list,**extra_args)
    register_free_upblock2d(model.inner_model.pipe, b1=1, b2=1, s1=1, s2=1)
    register_free_crossattn_upblock2d(model.inner_model.pipe, b1=1, b2=1, s1=1, s2=1)
    intermediates = {'x_inter': [x],'pred_x0': []}

    for i in trange(len(sigmas) - 1, disable=disable):
        if i == start_free_step:
            register_free_upblock2d(model.inner_model.pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
            register_free_crossattn_upblock2d(model.inner_model.pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
        # macs, params = profile(model, inputs=(x, sigmas[i] * s_in,*extra_args.values(),need_distill_uncond,tmp_list,uncond_list, ))
        prompt_embeds, denoised = model(x, sigmas[i] * s_in, tmp_list=tmp_list,need_distill_uncond=need_distill_uncond,uncond_list=uncond_list, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
            
        x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        intermediates['pred_x0'].append(denoised)
        intermediates['x_inter'].append(x)
            
        # print(denoised_d.max())
        
        # intermediates['noise'].append(denoised_d)
    return prompt_embeds, intermediates,x

@torch.no_grad()
def sample_dpmpp_sde(model
                     , x
                     , sigmas
                     , need_golden_noise = False
                     , extra_args=None
                     , callback=None
                     , tmp_list=[]
                     , need_distill_uncond=False
                     , uncond_list=[]
                     , disable=None, eta=1.
                     , s_noise=1.
                     , noise_sampler=None
                     , r=1 / 2):
    """DPM-Solver++ (stochastic)."""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    if need_golden_noise:
        x = model.get_golden_noised(x=x,sigma=sigmas[0] * s_in, sigma_nxt=sigmas[1] * s_in,**extra_args)
    
    intermediates = {'x_inter': [x],'pred_x0': []}

    for i in trange(len(sigmas) - 1, disable=disable):
        prompt_embeds, denoised = model(x, sigmas[i] * s_in, tmp_list=tmp_list,need_distill_uncond=need_distill_uncond,uncond_list=uncond_list, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
            intermediates['pred_x0'].append(denoised)
            intermediates['x_inter'].append(x)
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            prompt_embeds, denoised_2 = model(x_2, sigma_fn(s) * s_in, tmp_list=tmp_list,need_distill_uncond=need_distill_uncond,uncond_list=uncond_list, **extra_args) #(x, sigmas[i] * s_in, tmp_list=tmp_list,need_distill_uncond=need_distill_uncond,uncond_list=uncond_list, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            intermediates['pred_x0'].append(x)
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
            intermediates['x_inter'].append(x)
    return  prompt_embeds, intermediates,x


@torch.no_grad()
def sample_dpmpp_2m(model
                    , x
                    , sigmas
                    # , need_golden_noise = True
                    , extra_args=None
                    , callback=None
                    , disable=None
                    , tmp_list=[]
                    , need_distill_uncond=False
                    , start_free_step=9
                    , uncond_list=[]
                    , stop_t = None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    # if need_golden_noise:
    #     x = model.get_golden_noised(x=x,sigma=sigmas[0] * s_in, sigma_nxt=sigmas[1] * s_in,**extra_args)
    intermediates = {'x_inter': [x],'pred_x0': []}
    register_free_upblock2d(model.inner_model.pipe, b1=1, b2=1, s1=1, s2=1)
    register_free_crossattn_upblock2d(model.inner_model.pipe, b1=1, b2=1, s1=1, s2=1)

    for i in trange(len(sigmas) - 1, disable=disable):
        if i == start_free_step and len(sigmas) > 6:
            register_free_upblock2d(model.inner_model.pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
            register_free_crossattn_upblock2d(model.inner_model.pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
        # else:
        #     register_free_upblock2d(model.inner_model.pipe, b1=1.1, b2=1.1, s1=1.0, s2=1.0)
        #     register_free_crossattn_upblock2d(model.inner_model.pipe, b1=1.1, b2=1.1, s1=1.0, s2=1.0)
        # macs, params = profile(model, inputs=(x, sigmas[i] * s_in,*extra_args.values(),need_distill_uncond,tmp_list,uncond_list, ))
        prompt_embeds, denoised = model(x, sigmas[i] * s_in, tmp_list=tmp_list,need_distill_uncond=need_distill_uncond,uncond_list=uncond_list, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
            intermediates['pred_x0'].append(denoised)
            intermediates['x_inter'].append(x)
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
            intermediates['x_inter'].append(x)
            intermediates['pred_x0'].append(denoised)
            # print(denoised_d.max())
        old_denoised = denoised
        if i is not None and i == stop_t:
            return intermediates, x
        # intermediates['noise'].append(denoised_d)
    return prompt_embeds, intermediates,x

# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def convert_caption_json_to_str(json):
    caption = json["caption"]
    return caption

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./gen_img_val_xl"
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=12,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--stop_steps",
        type=int,
        default=6,
        help="number of stop sampling steps",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=1024,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=1024,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        default='./instances_val2014.json',
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--npnet-checkpoint",
        type=str,
        default='./sdxl.pth',
        help="if specified, load prompts from this file",
    )
    
    parser.add_argument(
        "--use_free_net",
        action='store_true',
        default=True,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--force_not_use_ct",
        action='store_true',
        default=False,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--force_not_use_NPNet",
        action='store_true',
        default=False,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--use_retrain",
        action='store_true',
        default=True,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--use_raw_golden_noise",
        action='store_true',
        default=False,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--inner_lcm_step",
        action='store_true',
        default=1,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--use_8full_trcik",
        action='store_true',
        default=True,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    seed_everything(opt.seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())
    
    seed_everything(opt.seed)

    DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse
    device = "cuda" 
    # pipe = StableDiffusionPipeline.from_single_file( "./counterfeit/Counterfeit-V3.0_fp32.safetensors")
    
    # pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    
    repo_id = "madebyollin/sdxl-vae-fp16-fix"  # e.g., "distilbert/distilgpt2"
    filename = "sdxl_vae.safetensors"  # e.g., "pytorch_model.bin"
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename,cache_dir=".")

    # pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    vae = AutoencoderKL.from_single_file(downloaded_path, torch_dtype=DTYPE)
    
    pipe = StableDiffusionXLPipeline.from_pretrained("Lykon/dreamshaper-xl-1-0",torch_dtype=DTYPE,vae=vae)
    # pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16,vae=vae)
    
    pipe.to('cuda')
    npn_net = NPNet128('SDXL', opt.npnet_checkpoint)
    
    
    # pipe.to(device=device, torch_dtype=DTYPE)
    if opt.use_free_net:
        register_free_upblock2d(pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
        register_free_crossattn_upblock2d(pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
    # pipe.unet.enable_freeu(s1=1.4,s2=1.6,b1=0.9,b2=0.9)
    # unet = pipe.unet
    noise_scheduler = pipe.scheduler
    alpha_schedule = noise_scheduler.alphas_cumprod.to(device=device, dtype=DTYPE)
    model_wrap = DiscreteEpsDDPMDenoiser(pipe, alpha_schedule, quantize=False)
    
    # text_encoder = pipe.text_encoder 
    # vae = pipe.vae
    # tokenizer = pipe.tokenizer

    # vae.requires_grad_(False)
    # text_encoder.requires_grad_(False)
    # unet.requires_grad_(False)
    
    # vae.to(device=device, dtype=DTYPE)
    # text_encoder.to(device=device, dtype=DTYPE)
    # unet.to(device=device, dtype=DTYPE)
    
    def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
        prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
        return {"prompt_embeds": prompt_embeds}
    
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
    )


    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        coco_annotation_file_path = opt.from_file
        coco_caption_file_path = './captions_val2014.json'
        coco_annotation = COCO(annotation_file=coco_annotation_file_path)
        coco_caption = COCO(annotation_file=coco_caption_file_path)
        query_names = [] #['cup','broccoli','dining table','toaster','carrot','toilet','sink','fork','hot dog','knife','pizza','spoon','donut','clock','bowl','cake','vase','banana','scissors','couch','apple','sandwich','potted plant','microwave','orange','bed','oven']
        unselect_names = [] # ['person','airplane','bird','mouse','cat','dog','horse','clock']

        # 获取包含指定类别的图像ID
        query_ids = []
        img_ids = coco_annotation.getImgIds()
        # for query_name in query_names:
        # query_ids += coco_annotation.getCatIds(catNms=query_names)
        # for query_id in query_ids:
        #     img_ids += coco_annotation.getImgIds(catIds=query_id)

        # 获取包含不需要类别的图像ID
        unselect_id = []
        unselect_img_ids = []
        for unselect_name in unselect_names:
            unselect_id += coco_annotation.getCatIds(catNms=[unselect_name])
            unselect_img_ids += coco_annotation.getImgIds(catIds=unselect_id)

        # 过滤掉包含不需要类别的图像ID
        real_img_ids = [item for item in img_ids if item not in unselect_img_ids]
        random.shuffle(real_img_ids)
        
        real_img_ids = real_img_ids[0:10000]

        # 获取这些图像的caption ID
        caption_ids = coco_caption.getAnnIds(imgIds=real_img_ids)

        # 获取并显示这些图像的captions
        captions = coco_caption.loadAnns(caption_ids)
        tmp_caption = []
        for idx,caption in enumerate(captions):
            if idx % 5 != 0:
                continue
            tmp_caption.append(caption)
        captions = tmp_caption
        
        data = list(map(lambda x: x['caption'], captions))
        data = data[(0):10000]
        images = coco_caption.loadImgs(ids=real_img_ids)
        folder_name = 'E:\\txt2img-samples\\scls_coco_img_val_random'
        img_path = 'D:\\research_project\\archive(2)\\coco2017\\images\\val2014'
        # if not os.path.exists(folder_name):
        #     os.makedirs(name=folder_name,exist_ok=True)
        #     img_file_name = [ img['file_name'] for img in images ]
        #     for filename in os.listdir(path=img_path):
        #         if filename in img_file_name:
        #             shutil.copy(os.path.join(img_path, filename), folder_name)

    if opt.stop_steps !=-1:
        folder_name = f"samples-customedXL-{opt.stop_steps}"
        if  opt.use_retrain:
            folder_name += "-retrain"
        if opt.use_free_net:
            folder_name += "-free"
        if opt.force_not_use_NPNet:
            folder_name += "-notNPNet"
        if opt.force_not_use_ct:
            folder_name += "-noneCT"
        if opt.use_raw_golden_noise:
            folder_name += "-rawGoldenNoise"
        if opt.use_8full_trcik:
            folder_name += "-full-trick"
        
        folder_name +=f"-{opt.inner_lcm_step}"
        folder_name +=f"-{opt.scale}"
        sample_path = os.path.join(outpath, folder_name)
    elif opt.stop_steps == -1:
        folder_name = f"samples-org-{opt.ddim_steps}"
        if opt.use_free_net:
            folder_name += "-free"
        if opt.force_not_use_NPNet:
            folder_name += "-notNPNet"
        if opt.use_raw_golden_noise:
            folder_name += "-rawGoldenNoise"
        sample_path = os.path.join(outpath, folder_name)
    # npn_net = NPNet64('SD1.5', opt.npnet_checkpoint)
    
    os.makedirs(sample_path, exist_ok=True)
    
    
    base_count = len(os.listdir(sample_path))
    
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        tic = time.time()
        all_samples = list()
        for n in trange(opt.n_iter, desc="Sampling", disable =not accelerator.is_main_process):
            for prompts in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                    
                    # torch.cuda.empty_cache()
                intermediate_photos = list()
                    # prompts = prompts[0]
                            
                    # if isinstance(prompts, tuple) or isinstance(prompts, str):
                    #     prompts = list(prompts)
                if isinstance(prompts, str):
                    prompts = prompts #+ 'high quality, best quality, masterpiece, 4K, highres, extremely detailed, ultra-detailed'
                    prompts = (prompts,)
                if isinstance(prompts, tuple) or isinstance(prompts, str):
                    prompts = list(prompts)
                    
                    
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                start_free_step = opt.stop_steps
                fir_stage_sigmas_ct = None
                sec_stage_sigmas_ct = None
                    # sigmas = model_wrap.get_sigmas(opt.ddim_steps).to(device=device)
                if opt.stop_steps == 4 and not opt.use_retrain and not opt.force_not_use_ct:
                    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
                    sigmas = get_sigmas_karras(8, sigma_min, sigma_max,rho=5.0, device=device)# 6.0 if 5 else  10.0
                    
                    ct_start, ct_end = model_wrap.sigma_to_t(sigmas[0]), model_wrap.sigma_to_t(sigmas[6])
                        # sigma_kct_start, sigma_kct_end = sigmas[0].item(), sigmas[5].item()
                    ct = get_sigmas_karras(5, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                    sigmas_ct = model_wrap.get_special_sigmas_with_timesteps(ct).to(device=device)

                        # timesteps = get_sigmas_karras(opt.ddim_steps, 1, 999,rho=1.2, device=device).to('cpu').numpy()
                        # sigmas = model_wrap.get_special_sigmas_with_timesteps(timesteps)
                elif opt.stop_steps == 4 and opt.use_retrain and not opt.force_not_use_ct:
                    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
                    if opt.use_free_net:
                        sigmas = get_sigmas_karras(8, sigma_min, sigma_max,rho=3.25, device=device)# 6.0 if 5 else  10.0
                    else:
                        sigmas = get_sigmas_karras(8, sigma_min, sigma_max,rho=7, device=device)
                    ct_start, ct_end = model_wrap.sigma_to_t(sigmas[0]), model_wrap.sigma_to_t(sigmas[6])
                        # sigma_kct_start, sigma_kct_end = sigmas[0].item(), sigmas[5].item()
                    ct = get_sigmas_karras(5, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                    sigmas_ct = model_wrap.get_special_sigmas_with_timesteps(ct).to(device=device)

                elif (opt.stop_steps == 5 or (opt.stop_steps == 6 and opt.use_raw_golden_noise)) and not opt.force_not_use_ct:
                    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
                    sigmas = get_sigmas_karras(8, sigma_min, sigma_max, rho=5.0, device=device)# 6.0 if 5 else  10.0
                    
                    ct_start, ct_end = model_wrap.sigma_to_t(sigmas[0]), model_wrap.sigma_to_t(sigmas[6])
                        # sigma_kct_start, sigma_kct_end = sigmas[0].item(), sigmas[5].item()
                    ct = get_sigmas_karras(6, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                    sigmas_ct = model_wrap.get_special_sigmas_with_timesteps(ct).to(device=device)
                    start_free_step = 5
                    fir_stage_sigmas_ct = sigmas_ct[:-1]
                    sec_stage_sigmas_ct = sigmas_ct[-2:]

                elif opt.stop_steps == 6 and not opt.force_not_use_ct:
                    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
                    sigmas = get_sigmas_karras(8, sigma_min, sigma_max,rho=5.0, device=device)# 6.0 if 5 else  10.0
                    
                    ct_start, ct_end = model_wrap.sigma_to_t(sigmas[0]), model_wrap.sigma_to_t(sigmas[6])
                        # sigma_kct_start, sigma_kct_end = sigmas[0].item(), sigmas[5].item()
                    ct = get_sigmas_karras(7, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                    sigmas_ct = model_wrap.get_special_sigmas_with_timesteps(ct).to(device=device)
                    start_free_step = 6
                    fir_stage_sigmas_ct = sigmas_ct[:-2]
                    sec_stage_sigmas_ct = sigmas_ct[-3:]

                elif opt.stop_steps == 8:
                    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
                    if not opt.use_8full_trcik:
                        sigmas = get_sigmas_karras(12, sigma_min, sigma_max,rho=12.0, device=device)# 6.0 if 5 else  10.0
                        ct_start, ct_end = model_wrap.sigma_to_t(sigmas[0]), model_wrap.sigma_to_t(sigmas[9])
                        
                    else:
                        sigmas = get_sigmas_karras(12, sigma_min, sigma_max,rho=12.0, device=device)# 6.0 if 5 else  10.0
                        ct_start, ct_end = model_wrap.sigma_to_t(sigmas[0]), model_wrap.sigma_to_t(sigmas[10])
                        
                    ct = get_sigmas_karras(opt.stop_steps +1, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                    sigmas_ct = model_wrap.get_special_sigmas_with_timesteps(ct).to(device=device)
                    start_free_step = 8
                elif opt.stop_steps == -1:
                    ct = get_sigmas_karras(opt.ddim_steps, 1, 999,rho=1.2, device=device).to('cpu').numpy()
                    sigmas_ct = model_wrap.get_special_sigmas_with_timesteps(ct).to(device=device)
                else:
                    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
                    sigmas = get_sigmas_karras(opt.ddim_steps, sigma_min, sigma_max,rho=12.0, device=device)# 6.0 if 5 else  10.0

                    ct_start, ct_end = model_wrap.sigma_to_t(sigmas[0]), model_wrap.sigma_to_t(sigmas[opt.stop_steps])
                        # sigma_kct_start, sigma_kct_end = sigmas[0].item(), sigmas[5].item()
                    ct = get_sigmas_karras(opt.stop_steps + 1, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
                    sigmas_ct = model_wrap.get_special_sigmas_with_timesteps(ct).to(device=device)

                ts = []
                for sigma in sigmas_ct:
                    t = model_wrap.sigma_to_t(sigma)
                    ts.append(t)
                    
                c_in = model_wrap.get_c_ins(sigmas=sigmas_ct)
                x = torch.randn([opt.n_samples, *shape], device=device) * sigmas_ct[0]
                model_wrap_cfg = CFGDenoiser(model_wrap)
                (
                    c,
                    uc,
                    _,
                    _,
                ) = pipe.encode_prompt(
                    prompt=prompts,
                    device=device,
                    do_classifier_free_guidance=True,
                )
                    # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

                if (opt.stop_steps != -1 or opt.ddim_steps <= 8) and not opt.force_not_use_NPNet:
                    x = npn_net(x,c)
                    
                extra_args = {'prompt': prompts, 'cond_scale': opt.scale}
                noise_training_list = {}
                if sec_stage_sigmas_ct is not None and fir_stage_sigmas_ct is not None:
                    if (opt.stop_steps != -1 or opt.ddim_steps <= 10) and not (opt.stop_steps == 8 or opt.stop_steps == 7):
                        prompt_embeds, guide_distill, samples_ddim = sample_dpmpp_ode(model_wrap_cfg
                                                                    , x
                                                                    , fir_stage_sigmas_ct
                                                                    , extra_args=extra_args
                                                                    , disable=not accelerator.is_main_process
                                                                    , need_raw_noise=opt.stop_steps == 6 and opt.use_raw_golden_noise
                                                                    , tmp_list=intermediate_photos)
                        _, _, samples_ddim = sample_euler(model_wrap_cfg
                                                                    , samples_ddim
                                                                    , sec_stage_sigmas_ct
                                                                    , extra_args=extra_args
                                                                    , disable=not accelerator.is_main_process
                                                                    , s_noise = 0.3
                                                                    , tmp_list=intermediate_photos)
                    else: 
                        prompt_embeds, guide_distill, samples_ddim = sample_dpmpp_2m(model_wrap_cfg
                                                                    , x
                                                                    , fir_stage_sigmas_ct
                                                                    , extra_args=extra_args
                                                                    , disable=not accelerator.is_main_process
                                                                    , tmp_list=intermediate_photos)
                        _, _, samples_ddim = sample_dpmpp_sde(model_wrap_cfg
                                                                    , samples_ddim
                                                                    , sec_stage_sigmas_ct
                                                                    , extra_args=extra_args
                                                                    , disable=not accelerator.is_main_process
                                                                    , tmp_list=intermediate_photos)
                else:
                    if (opt.stop_steps != -1 or opt.ddim_steps <= 10) and not (opt.stop_steps == 8 or opt.stop_steps == 7):
                        prompt_embeds, guide_distill, samples_ddim = sample_dpmpp_ode(model_wrap_cfg
                                                                            , x
                                                                            , sigmas_ct
                                                                            , extra_args=extra_args
                                                                            , disable=not accelerator.is_main_process
                                                                            , need_raw_noise=opt.stop_steps == 6 and opt.use_raw_golden_noise
                                                                            , tmp_list=intermediate_photos)
                                                                #    , stop_t=4)
                    else:
                        prompt_embeds, guide_distill, samples_ddim = sample_dpmpp_2m(model_wrap_cfg
                                                                                 , x
                                                                                 , sigmas_ct 
                                                                                 , extra_args=extra_args
                                                                                 , start_free_step=start_free_step
                                                                                 , disable=not accelerator.is_main_process
                                                                                 , tmp_list=intermediate_photos)
                        # print('2m')
                x_samples_ddim = pipe.vae.decode(samples_ddim / pipe.vae.config.scaling_factor).sample
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                if True: # not opt.skip_save:
                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1



        toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()