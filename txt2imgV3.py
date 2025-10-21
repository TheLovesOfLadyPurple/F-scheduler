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
from diffusers import StableDiffusionPipeline
from sampler import DPMSolverv3Sampler
import functools

from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="an astronaut riding a horse",
        help="the prompt to render",
    )
    parser.add_argument(
        "--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/txt2img-samples"
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
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
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
        default=7.5,
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
    parser.add_argument("--statistics_dir", type=str, default='./7.5_250_1024', help="Statistics path for DPM-Solver-v3.")
    parser.add_argument(
        "--config",
        type=str,
        default="./codebases/stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
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
        default=4,
        help="starting step for free U-Net",
    )
    parser.add_argument(
        "--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast"
    )
    opt = parser.parse_args()
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    
    seed_everything(opt.seed)

    DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse
    device = "cuda" 
    # pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    pipe = StableDiffusionPipeline.from_pretrained('sd-legacy/stable-diffusion-v1-5')
    pipe.to(device=device, torch_dtype=DTYPE)
    # if opt.use_free_net:
    #     register_free_upblock2d(pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
    #     register_free_crossattn_upblock2d(pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
    # pipe.unet.enable_freeu(s1=1.4,s2=1.6,b1=0.9,b2=0.9)
    # unet = pipe.unet
    
    sampler = DPMSolverv3Sampler(opt.statistics_dir, pipe, steps=opt.steps, guidance_scale=opt.scale)


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

    sample_path = os.path.join(outpath, "samples")
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
                    uc = None
                    if opt.scale != 1.0:
                        uc = compute_embeddings_fn(batch_size * [""])
                    uc = uc.pop("prompt_embeds") if uc is not None else None
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = compute_embeddings_fn(prompts).pop("prompt_embeds")
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    # if opt.method == "dpm_solver_v3":
                            # batch_size, shape, conditioning, x_T, unconditional_conditioning
                    samples, _ = sampler.sample(
                        conditioning=c,
                        batch_size=opt.n_samples,
                        shape=shape,
                        unconditional_conditioning=uc,
                        x_T=start_code,
                        start_free_u_step=opt.start_free_u_step if opt.start_free_u_step > 0 else None,
                        use_corrector=opt.scale < 5.0,
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
