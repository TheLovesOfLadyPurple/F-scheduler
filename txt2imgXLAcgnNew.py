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
    AutoencoderKL,
    StableDiffusionXLPipeline
)
from customed_unipc_scheduler import CustomedUniPCMultistepScheduler
from test_unipc_scheduler import UniPCMultistepScheduler
from huggingface_hub import hf_hub_download
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

def prepare_sdxl_pipeline_step_parameter( pipe: StableDiffusionXLPipeline
                                         , prompts
                                         , need_cfg
                                         , device
                                         , negative_prompt = None
                                         , W = 1024
                                         , H = 1024): # need to correct the format
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            negative_prompt=negative_prompt,
            device=device,
            do_classifier_free_guidance=need_cfg,
        )
    # timesteps = pipe.scheduler.timesteps
    
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = pooled_prompt_embeds.to(device)
        original_size = (W, H)
        crops_coords_top_left = (0, 0)
        target_size = (W, H)
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
        default=6,
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
        "--prompt",
        type=str,
        default='((masterpiece,best quality)) , 1girl, ((school uniform)),brown blazer, black skirt,small breasts,necktie,red plaid skirt,looking at viewer',#'A man holding a cell phone with a pack of Marlboro Lights on his lap',
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
        default=True,
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
        default=True,
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
    parser.add_argument(
        "--start_free_u_step",
        type=int,
        default=-1,
        help="starting step for free U-Net",
    )
    
    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    
    seed_everything(opt.seed)

    DTYPE = torch.float16  # torch.float16 works as well, but pictures seem to be a bit worse
    device = "cuda" 
    vae = AutoencoderKL.from_single_file("./sdxl_vae_fp16.safetensors", torch_dtype=torch.float16)
    vae.to('cuda')
    
    pipe = StableDiffusionXLPipeline.from_single_file("./novaAnimeXL_ilV120.safetensors",torch_dtype=torch.float16,vae=vae)
    # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=DTYPE, vae=vae
    # )
    scheduler = CustomedUniPCMultistepScheduler.from_config(pipe.scheduler.config
                                                            , solver_order = 2 if opt.ddim_steps==8 else 1
                                                            ,denoise_to_zero = False)
    pipe.scheduler = scheduler
    pipe.to('cuda')

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
                    
                    intermediate_photos = list()
                    
                    latents = torch.randn(
                        (batch_size, pipe.unet.config.in_channels, opt.H // 8, opt.W // 8),
                        device=device,
                    )
                    latents = latents * pipe.scheduler.init_noise_sigma
                    
                    pipe.scheduler.set_timesteps(opt.ddim_steps)
                    idx = 0
                    register_free_upblock2d(pipe, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
                    register_free_crossattn_upblock2d(pipe, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
                    for t in tqdm(pipe.scheduler.timesteps):
                        # Still not enough. I will tell you, what is the best implementation.  Although not via the following code.
                        
                        # if idx == len(pipe.scheduler.timesteps) - 1: 
                        #     break
                        if idx == opt.start_free_u_step and opt.start_free_u_step >=0:
                            register_free_upblock2d(pipe, b1=1.2, b2=1.2, s1=0.9, s2=0.9)
                            register_free_crossattn_upblock2d(pipe, b1=1.2, b2=1.2, s1=0.9, s2=0.9)
                        latent_model_input  = torch.cat([latents] * 2)
                        
                        latent_model_input  = pipe.scheduler.scale_model_input(latent_model_input , timestep=t)
                        negative_prompts = ''#'(worst quality:2), (low quality:2), (normal quality:2), bad anatomy, bad proportions, poorly drawn face, poorly drawn hands, missing fingers, extra limbs, blurry, pixelated, distorted, lowres, jpeg artifacts, watermark, signature, text, (deformed:1.5), (bad hands:1.3), overexposed, underexposed, censored, mutated, extra fingers, cloned face, bad eyes'
                        negative_prompts = batch_size * [negative_prompts]
                        
                        prompt_embeds, cond_kwargs = prepare_sdxl_pipeline_step_parameter(pipe
                                                                                      , prompts
                                                                                      , need_cfg=True
                                                                                      , device=pipe.device
                                                                                      , negative_prompt=negative_prompts
                                                                                      , W=opt.W
                                                                                      , H=opt.H)
                        noise_pred  = pipe.unet(latent_model_input 
                                        , t
                                        , encoder_hidden_states=prompt_embeds.to(device=latents.device, dtype=latents.dtype)
                                        , added_cond_kwargs=cond_kwargs).sample
                        uncond, cond = noise_pred.chunk(2)
                        noise_pred  = uncond + (cond - uncond) * opt.scale
                        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
                        idx += 1
                        
                    x_samples_ddim = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    if True:
                        for x_sample in x_samples_ddim:
                                # x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1

                            


            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()