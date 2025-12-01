import argparse, os
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
from pycocotools.coco import COCO
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
# from huggingface_hub import login
from SVDNoiseUnet import NPNet64
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

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./gen_img_val_v15"
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--stop_steps",
        type=int,
        default=-1,
        help="number of stop sampling steps",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
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
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default='an astronaut riding a horse',
        help="text to generate images",
    )
    parser.add_argument(
        "--npnet-checkpoint",
        type=str,
        default='./HPSFilterFix.pth',
        help="if specified, load prompts from this file",
    )
    
    parser.add_argument(
        "--use_free_net",
        action='store_true',
        default=False,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--force_not_use_ct",
        action='store_true',
        default=False,
        help="use the customized timeschedule.",
    )
    parser.add_argument(
        "--force_not_use_NPNet",
        action='store_true',
        default=True,
        help="use the Golden Noise for inference.",
    )
    parser.add_argument(
        "--use_raw_golden_noise",
        action='store_true',
        default=False,
        help="use the Raw Golden Noise for inference.",
    )
    parser.add_argument(
        "--use_8full_trcik",
        action='store_true',
        default=True,
        help="use the 8 full trick for inference.",
    )
    parser.add_argument(
        "--is_acgn",
        action='store_true',
        default=False,
        help="use the 8 full trick for inference.",
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

    DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse
    device = "cuda" 
    if opt.is_acgn:
        pipe = StableDiffusionPipeline.from_single_file( "./counterfeit/Counterfeit-V3.0_fp32.safetensors")
        neg_emb = pipe.load_textual_inversion("./EasyNegative.safetensors", device=device, dtype=DTYPE)
    else:
        pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    # pipe = StableDiffusionPipeline.from_pretrained('sd-legacy/stable-diffusion-v1-5')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    # pipe = StableDiffusionPipeline.from_single_file( "./v1-5-pruned-emaonly.safetensors")
    
    # npn_net = NPNet64('SD1.5', opt.npnet_checkpoint)
    
    
    pipe.to(device=device, torch_dtype=DTYPE)
    if opt.use_free_net:
        register_free_upblock2d(pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
        register_free_crossattn_upblock2d(pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
    
    

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples

    prompt = opt.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

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
    direct_distill_intermediate_count = 0
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
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

                    x_samples_ddim = pipe(prompt=prompts,num_inference_steps=opt.ddim_steps,guidance_scale=opt.scale,height=opt.H,width=opt.W).images
                    if True:
                        for x_sample in x_samples_ddim:
                            # x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            x_sample.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1

                            


            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()