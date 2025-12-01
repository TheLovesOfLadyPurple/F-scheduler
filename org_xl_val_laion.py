
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse, os, sys, glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice

from huggingface_hub import login, hf_hub_download
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import accelerate
import torchsde
import pandas as pd
import diffusers
from pycocotools.coco import COCO
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LatentConsistencyModelPipeline,
    StableDiffusionXLPipeline
)
import datasets
from huggingface_hub import login
import shutil
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

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./gen_img_org"
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
        "--iDDD_stop_steps",
        type=int,
        default=5,
        help="number of iDDD sampling steps",
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
        default='./extracted_texts_10k.txt',
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--npnet-checkpoint",
        type=str,
        default= './ACGGoldenNoiseLargeHPS3000.pth', #'./HPSFilterFix.pth',
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--naf-opt",
        type=str,
        default='options/test/improved-DDD/LCMXABWithPromptNAFVal-acgn.yml',#'options/test/improved-DDD/LCMXABWithPromptNAFVal.yml',
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--use_free_net",
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
        "--use_acgn",
        action='store_true',
        default=False,
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
    # login("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") # login to HuggingFace Hub
    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    seed_everything(opt.seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())
    
    seed_everything(opt.seed)

    DTYPE = torch.float16  # torch.float16 works as well, but pictures seem to be a bit worse
    device = "cuda" 
    repo_id = "madebyollin/sdxl-vae-fp16-fix"  # e.g., "distilbert/distilgpt2"
    filename = "sdxl_vae.safetensors"  # e.g., "pytorch_model.bin"
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename,cache_dir=".")

    vae = AutoencoderKL.from_single_file(downloaded_path, torch_dtype=DTYPE)
    vae.to('cuda')
    
        
    pipe = StableDiffusionXLPipeline.from_pretrained("Lykon/dreamshaper-xl-1-0",torch_dtype=DTYPE,vae=vae)
    # pipe = StableDiffusionXLPipeline.from_pretrained(
        # "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=DTYPE, vae=vae
    # )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, final_sigmas_type="sigma_min")


    # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=DTYPE, vae=vae
    # )
    pipe.to(device=device, torch_dtype=DTYPE)
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    elif opt.use_acgn:
        dataset = datasets.load_dataset("FredZhang7/anime-prompts-180K")
        data = dataset["train"]
        data = data["safebooru_clean"]
        data = data[0:0 + 10000]#[53000+9274:53000+9274+120000]
    else:
        print(f"reading prompts from {opt.from_file}")
        # Read captions from text file with format "Image XXXXX: caption"
        data = []
        try:
            with open(opt.from_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line and ':' in line:
                    # Split on first colon to get the caption part
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        caption = parts[1].strip()
                        if caption:  # Only add non-empty captions
                            data.append(caption)
            
            print(f"Loaded {len(data)} captions from {opt.from_file}")
            
            # Limit to 10000 captions if more are available
            if len(data) > 10000:
                data = data[:10000]
                print(f"Limited to first 10000 captions")
                
        except FileNotFoundError:
            print(f"Error: Could not find file {opt.from_file}")
            print("Falling back to default prompt...")
            data = ["a beautiful landscape"]
        except Exception as e:
            print(f"Error reading file {opt.from_file}: {e}")
            print("Falling back to default prompt...")
            data = ["a beautiful landscape"]

    
    folder_name = f"samples-org-xl-{opt.ddim_steps}"
    if opt.use_free_net:
        folder_name += "-free"
    if opt.force_not_use_NPNet:
        folder_name += "-notNPNet"
    W, H = opt.W, opt.H
    sample_path = os.path.join(outpath, folder_name)
    os.makedirs(sample_path, exist_ok=True)
    
    
    base_count = len(os.listdir(sample_path))
    
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
                        
                    samples_ddim = pipe(prompt=prompts
                                          ,num_inference_steps=opt.ddim_steps
                                          ,guidance_scale=opt.scale
                                          ,height=H
                                          ,width=W).images
                    if True:
                        # x_samples_ddim = pipe.vae.decode(samples_ddim / pipe.vae.config.scaling_factor).sample
                        for x_sample in samples_ddim:
                            # x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            x_sample.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1

                    # for idx,imgs in enumerate(intermediate_photos):
                    #     if idx > 6:
                    #         continue
                    #     tmp_photos = guide_distill['x_inter'][-1] #if idx < len(guide_distill['pred_x0']) else intermediate_photos[-1]
                    #     for secidx,img in enumerate(imgs['tmp_z']):
                    #         img = img.permute(1,2,0)
                    #         torch.save(img,os.path.join(intermediate_path, f"{direct_distill_intermediate_count:05}_{(int(ts[idx])):05}.pth"))
                    #     for secidx,img in enumerate(imgs['tmp_x0']):
                    #         img = img.permute(1,2,0)
                    #         torch.save(img,os.path.join(tmp_x0_path, f"{direct_distill_intermediate_count:05}_{(int(ts[idx])):05}.pth"))
                        
                    #     for secidx,img in enumerate(tmp_photos):
                    #         img = img.permute(1,2,0)
                    #         torch.save(img,os.path.join(final_x0_path, f"{direct_distill_intermediate_count:05}_{(int(ts[idx])):05}.pth"))
                    #     for secidx,prompt in enumerate(c):
                    #         torch.save(prompt,os.path.join(prompt_path, f"{direct_distill_intermediate_count:05}_{(int(ts[idx])):05}.pth"))
                    #     for secidx,prompt in enumerate(uc):
                    #         torch.save(prompt,os.path.join(negative_prompt_path, f"{direct_distill_intermediate_count:05}_{(int(ts[idx])):05}.pth"))
                    #     direct_distill_intermediate_count += 1
                            


            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()