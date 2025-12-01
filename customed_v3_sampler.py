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
from sampler import DPMSolverv3Sampler

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
    pipe = StableDiffusionPipeline.from_pretrained('sd-legacy/stable-diffusion-v1-5')
    # pipe = StableDiffusionPipeline.from_single_file( "./v1-5-pruned-emaonly.safetensors")
    
    npn_net = NPNet64('SD1.5', opt.npnet_checkpoint)
    
    pipe.to(device=device, torch_dtype=DTYPE)
    if opt.use_free_net:
        register_free_upblock2d(pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
        register_free_crossattn_upblock2d(pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
    sampler = DPMSolverv3Sampler('./7.5_250_1024',pipe, steps=opt.stop_steps, guidance_scale=opt.scale)
    
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
        img_path = 'D:\\research_project\\archive(2)\\coco2014\\images\\val2014'
        # if not os.path.exists(folder_name):
        #     os.makedirs(name=folder_name,exist_ok=True)
        #     img_file_name = [ img['file_name'] for img in images ]
        #     for filename in os.listdir(path=img_path):
        #         if filename in img_file_name:
        #             shutil.copy(os.path.join(img_path, filename), folder_name)

    if opt.stop_steps !=-1:
        folder_name = f"samples-customed-{opt.stop_steps}"
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
                    encoded_text = compute_embeddings_fn(prompts)
                    uc = None
                    if opt.scale != 1.0:
                        uc = compute_embeddings_fn(batch_size * [""])
                    uc = uc.pop("prompt_embeds") if uc is not None else None
                    c =  encoded_text.pop("prompt_embeds")
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    
                    
                    x = torch.randn([opt.n_samples, *shape], device=device) 
                    if (opt.stop_steps != -1 or opt.ddim_steps <= 8) and not opt.force_not_use_NPNet:
                        x = npn_net(x,c)
                    extra_args = {'cond': c, 'uncond': uc, 'cond_scale': opt.scale}
                    noise_training_list = {}
                    samples, _ = sampler.sample(
                        conditioning=c,
                        batch_size=opt.n_samples,
                        shape=shape,
                        unconditional_conditioning=uc,
                        x_T=x,
                        start_free_u_step=2 if opt.use_free_net else -1,
                        use_corrector=True,
                    )
                    
                        
                    x_samples_ddim = pipe.vae.decode(samples / pipe.vae.config.scaling_factor).sample
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    if True: # not opt.skip_save:
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1

                    # for idx,imgs in enumerate(guide_distill['x_inter']):
                    #     if idx > 6:
                    #         continue
                    #     tmp_photos = guide_distill['x_inter'][-1] #if idx < len(guide_distill['pred_x0']) else intermediate_photos[-1]
                    #     for secidx,img in enumerate(imgs):
                    #         img = img.permute(1,2,0)
                    #         torch.save(img,os.path.join(intermediate_path, f"{direct_distill_intermediate_count:05}_{(int(ts[idx])):05}.pth"))
                    #     for secidx,img in enumerate(tmp_photos):
                    #         img = img.permute(1,2,0)
                    #         torch.save(img,os.path.join(final_x0_path, f"{direct_distill_intermediate_count:05}_{(int(ts[idx])):05}.pth"))
                    #     direct_distill_intermediate_count += 1
                            


            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()