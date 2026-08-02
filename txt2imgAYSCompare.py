import argparse
import os
import time

import accelerate
import numpy as np
import torch
from PIL import Image
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import trange

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.schedulers.scheduling_utils import AysSchedules


def save_image_grid(images, output_path: str, rows: int = 1):
    tensor_images = []
    for image in images:
        image_array = np.array(image, copy=True)
        tensor_image = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        tensor_images.append(tensor_image)

    grid = make_grid(tensor_images, nrow=max(1, len(images) // rows), padding=8)
    grid = (grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(grid).save(output_path)


def make_generator(seed: int, device: str) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./gen_img_val_v15")
    parser.add_argument("--ddim_steps", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--precision",
        type=str,
        choices=["full", "autocast"],
        default="autocast",
    )
    parser.add_argument("--is_acgn", action="store_true", default=False)
    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    seed_everything(opt.seed)

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else accelerator.device.type

    if opt.is_acgn:
        pipe = StableDiffusionPipeline.from_single_file("./counterfeit/Counterfeit-V3.0_fp32.safetensors")
        pipe.load_textual_inversion("./EasyNegative.safetensors", device=device, dtype=dtype)
    else:
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device=device, torch_dtype=dtype)

    schedule = AysSchedules["StableDiffusionTimesteps"]

    folder_name = f"samples-ays-compare-{opt.ddim_steps}-steps"
    if opt.is_acgn:
        folder_name += "-acgn"

    sample_path = os.path.join(opt.outdir, folder_name)
    default_path = os.path.join(sample_path, "default")
    ays_path = os.path.join(sample_path, "ays")
    compare_path = os.path.join(sample_path, "compare")
    os.makedirs(default_path, exist_ok=True)
    os.makedirs(ays_path, exist_ok=True)
    os.makedirs(compare_path, exist_ok=True)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    prompts = [opt.prompt] * opt.n_samples

    with torch.no_grad():
        with precision_scope("cuda") if device == "cuda" else nullcontext():
            tic = time.time()
            for iteration in trange(opt.n_iter, desc="Sampling", disable=not accelerator.is_main_process):
                current_seed = opt.seed + iteration
                default_images = pipe(
                    prompt=prompts,
                    num_inference_steps=opt.ddim_steps,
                    guidance_scale=opt.scale,
                    height=opt.H,
                    width=opt.W,
                    generator=make_generator(current_seed, device),
                ).images

                ays_images = pipe(
                    prompt=prompts,
                    timesteps=schedule,
                    guidance_scale=opt.scale,
                    height=opt.H,
                    width=opt.W,
                    generator=make_generator(current_seed, device),
                ).images

                for image_index, image in enumerate(default_images):
                    image.save(os.path.join(default_path, f"iter{iteration:03}_img{image_index:02}.png"))

                for image_index, image in enumerate(ays_images):
                    image.save(os.path.join(ays_path, f"iter{iteration:03}_img{image_index:02}.png"))

                comparison_images = []
                for default_image, ays_image in zip(default_images, ays_images):
                    comparison_images.extend([default_image, ays_image])

                save_image_grid(
                    comparison_images,
                    os.path.join(compare_path, f"iter{iteration:03}_grid.png"),
                    rows=1,
                )

            toc = time.time()

    print(f"Saved comparison samples to: {sample_path}")
    print(f"Elapsed time: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()