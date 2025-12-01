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
    AutoencoderKL,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)
import functools
import random
from sampler import UniPCSampler
import functools
from huggingface_hub import login, hf_hub_download
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from SVDNoiseUnet import NPNet128

import json
import subprocess
import os
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from sampler import UniPCSampler

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

def prepare_sdxl_pipeline_step_parameter(pipe, prompts, need_cfg, device, negative_prompts, W = 1024, H = 1024):
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompts,
        negative_prompt=negative_prompts,
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


def model_closure(pipe):
    def model_fn(x, t, c):
        prompt = c[0]
        cond_kwargs = c[1] if len(c) > 1 else None
        # prompt_embeds, cond_kwargs = prepare_sdxl_pipeline_step_parameter(pipe=pipe,prompts = prompt, need_cfg=True, device=pipe.device,negative_prompts=negative_prompt)
        # prompt_embeds, cond_kwargs = c
        return pipe.unet(x
                         , t
                         , encoder_hidden_states=prompt.to(device=x.device, dtype=x.dtype)
                         , added_cond_kwargs=cond_kwargs).sample

    return model_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./gen_img_val_v15_unipc"
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
        default='./extracted_texts_10k.txt',
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
        default=True,
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
        default=4,
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
    parser.add_argument(
        "--start_free_u_step",
        type=int,
        default=4,
        help="starting step for free U-Net",
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
    repo_id = "madebyollin/sdxl-vae-fp16-fix"  # e.g., "distilbert/distilgpt2"
    filename = "sdxl_vae.safetensors"  # e.g., "pytorch_model.bin"
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename,cache_dir=".")
    npn_net = NPNet128('SDXL', opt.npnet_checkpoint)

    vae = AutoencoderKL.from_single_file(downloaded_path, torch_dtype=DTYPE)
    vae.to('cuda')
    
    pipe = StableDiffusionXLPipeline.from_pretrained("Lykon/dreamshaper-xl-1-0",torch_dtype=DTYPE,vae=vae)
    pipe.to('cuda')
    sampler = UniPCSampler(pipe, model_closure=model_closure, steps=opt.stop_steps, guidance_scale=opt.scale)
    
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

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
        # if not os.path.exists(folder_name):
        #     os.makedirs(name=folder_name,exist_ok=True)
        #     img_file_name = [ img['file_name'] for img in images ]
        #     for filename in os.listdir(path=img_path):
        #         if filename in img_file_name:
        #             shutil.copy(os.path.join(img_path, filename), folder_name)

    if opt.stop_steps !=-1:
        folder_name = f"samples-customed-{opt.stop_steps}-unipc"
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
    intermediate_path = os.path.join(outpath, f'intermediate-{opt.ddim_steps}-{opt.scale}')
    final_x0_path = os.path.join(outpath, f'final_x0-{opt.ddim_steps}-{opt.scale}')
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(final_x0_path, exist_ok=True)

    base_count = len(os.listdir(sample_path))
    direct_distill_intermediate_count = 0
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            tic = time.time()
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling", disable =not accelerator.is_main_process):
                for prompts in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                    c = [prompts]
                    uc = [''] 
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
                    # if opt.method == "dpm_solver_v3":
                            # batch_size, shape, conditioning, x_T, unconditional_conditioning
                    samples, _ = sampler.sample(
                        conditioning=c,
                        batch_size=opt.n_samples,
                        x_T=start_code,
                        shape=shape,
                        unconditional_conditioning=uc,
                        start_free_u_step=opt.start_free_u_step if opt.start_free_u_step > 0 else None,
                        xl_preprocess_closure = prepare_sdxl_pipeline_step_parameter,
                        # npnet = npn_net,
                        use_corrector=True,
                    )

                    x_samples = pipe.vae.decode(samples / pipe.vae.config.scaling_factor).sample
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

                    x_image_torch = torch.from_numpy(x_samples).permute(0, 3, 1, 2) # need to pay attention

                    for x_sample in x_image_torch:
                        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1

                    all_samples.append(x_image_torch)

            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()