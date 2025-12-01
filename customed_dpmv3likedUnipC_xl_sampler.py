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
from customed_unipc_scheduler import CustomedUniPCMultistepScheduler
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
        default="./gen_img_val_v15_coco2014_unipc"
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
        default=-1,
        help="starting step for free U-Net",
    )
    
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
    npn_net = NPNet128('SDXL', opt.npnet_checkpoint)

    vae = AutoencoderKL.from_single_file(downloaded_path, torch_dtype=DTYPE)
    vae.to('cuda')
    
    # pipe = StableDiffusionXLPipeline.from_pretrained("Lykon/dreamshaper-xl-1-0",torch_dtype=DTYPE,vae=vae)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=DTYPE, vae=vae
    )
    scheduler = CustomedUniPCMultistepScheduler.from_config(pipe.scheduler.config
                                                            , solver_order = 2 if opt.ddim_steps==8 else 1
                                                            ,denoise_to_zero = False)
    pipe.scheduler = scheduler
    pipe.to('cuda')
    
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
        img_path = 'D:\\research_project\\archive(2)\\coco2014\\images\\val2014'
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
                    
                    intermediate_photos = list()
                    
                    latents = torch.randn(
                        (batch_size, pipe.unet.config.in_channels, opt.H // 8, opt.W // 8),
                        device=device,
                    )
                    latents = latents * pipe.scheduler.init_noise_sigma
                    
                    pipe.scheduler.set_timesteps(opt.stop_steps)
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
                        negative_prompts = ''
                        negative_prompts = batch_size * [negative_prompts]
                        
                        prompt_embeds, cond_kwargs = prepare_sdxl_pipeline_step_parameter(pipe
                                                                                      , prompts
                                                                                      , need_cfg=True
                                                                                      , device=pipe.device
                                                                                      , negative_prompts=negative_prompts
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