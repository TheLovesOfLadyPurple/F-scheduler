import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
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
import random
print(sys.path)
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)
from sampler import UniPCSampler
import functools
from huggingface_hub import login, hf_hub_download
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from SVDNoiseUnet import NPNet128

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


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
        "--prompt",
        type=str,
        nargs="?",
        default="A man holding a cell phone with a pack of Marlboro Lights on his lap",
        help="the prompt to render",
    )
    parser.add_argument(
        "--unconditional_prompt",
        type=str,
        nargs="?",
        default="",
        help="the unconditional prompt to render",
    )
    parser.add_argument(
        "--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/txt2img-samples-xl"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="number of sampling steps",
    )
    parser.add_argument("--method", default="dpm_solver_v3", choices=["ddim", "plms", "dpm_solver++", "uni_pc", "dpm_solver_v3"])
    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=4,
        help="sample this often",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
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
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--npnet-checkpoint",
        type=str,
        default='./sdxl.pth',
        help="if specified, load prompts from this file",
    )
    parser.add_argument("--statistics_dir", type=str, default='./ems/statistics/sdv1-5', help="Statistics path for DPM-Solver-v3.")
    parser.add_argument(
        "--config",
        type=str,
        default="./codebases/stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
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
        "--use_free_net",
        action='store_true',
        default=False,
        help="use the free network for inference.",
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
        "--start_free_u_step",
        type=int,
        default=-1,
        help="starting step for free U-Net",
    )
    parser.add_argument(
        "--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast"
    )
    opt = parser.parse_args()
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    
    seed_everything(opt.seed)

    DTYPE = torch.float16  # torch.float16 works as well, but pictures seem to be a bit worse
    device = "cuda" 
    repo_id = "madebyollin/sdxl-vae-fp16-fix"  # e.g., "distilbert/distilgpt2"
    filename = "sdxl_vae.safetensors"  # e.g., "pytorch_model.bin"
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename,cache_dir=".")
    npn_net = NPNet128('SDXL', opt.npnet_checkpoint)

    vae = AutoencoderKL.from_single_file(downloaded_path, torch_dtype=DTYPE)
    vae.to('cuda')
    
    # pipe = StableDiffusionXLPipeline.from_single_file("./novaAnimeXL_ilV120.safetensors",torch_dtype=DTYPE,vae=vae)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=DTYPE, vae=vae
    )
    pipe.to('cuda')
    sampler = UniPCSampler(pipe,model_closure=model_closure
                           , steps=opt.steps
                           , guidance_scale=opt.scale
                           , denoise_to_zero=True
                           , need_fp16_discrete_method=True
                           , force_not_use_afs=True)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        if prompt is None:
            prompt = ""
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, f"samples-unipc-{opt.steps}-free_at{opt.start_free_u_step}-scale-{opt.scale}")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            tic = time.time()
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    # uc = None
                    # if opt.scale != 1.0:
                    #     uc = compute_embeddings_fn(batch_size * [""])
                    # uc = uc.pop("prompt_embeds") if uc is not None else None
                    # if isinstance(prompts, tuple):
                    #     prompts = list(prompts)
                    # c = compute_embeddings_fn(prompts).pop("prompt_embeds")
                    c = prompts
                    uc = [opt.unconditional_prompt] * len(c) if opt.scale != 1.0 else None
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    # if opt.method == "dpm_solver_v3":
                            # batch_size, shape, conditioning, x_T, unconditional_conditioning
                    samples, _ = sampler.sample(
                        conditioning=c,
                        batch_size=opt.n_samples,
                        shape=shape,
                        unconditional_conditioning=uc,
                        x_T=start_code,
                        start_free_u_step=opt.start_free_u_step if opt.start_free_u_step >= 0 else None,
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


    print(f"Your samples are ready and waiting for you here: \n{outpath} \n" f" \nEnjoy.")


if __name__ == "__main__":
    main()
