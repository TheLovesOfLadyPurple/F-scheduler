import argparse
import json
import os
import random
import time
from contextlib import nullcontext

import accelerate
import torch
from pycocotools.coco import COCO
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from txt2imgLD3CompareCoco import LD3UniPCMultistepScheduler

from dpm_solver_v3 import NoiseScheduleVP
from step_optim import StepOptim


def load_coco_prompts(annotation_file: str, caption_file: str, num_prompts: int) -> list[str]:
    coco_annotation = COCO(annotation_file=annotation_file)
    coco_caption = COCO(annotation_file=caption_file)

    img_ids = coco_annotation.getImgIds()
    random.shuffle(img_ids)
    selected_img_ids = img_ids[:num_prompts]

    caption_ids = coco_caption.getAnnIds(imgIds=selected_img_ids)
    captions = coco_caption.loadAnns(caption_ids)
    captions = [caption for index, caption in enumerate(captions) if index % 5 == 0]

    prompts = [caption["caption"] for caption in captions][:num_prompts]
    if not prompts:
        raise ValueError("No prompts were loaded from the supplied COCO files.")

    return prompts


def save_images(images, output_dir: str, start_index: int):
    for offset, image in enumerate(images):
        image.save(os.path.join(output_dir, f"{start_index + offset:05}.png"))


def append_metadata(metadata_path: str, records: list[dict]):
    with open(metadata_path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_scheduler(pipe: StableDiffusionPipeline, sampler: str):
    if sampler == "dpmpp":
        return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if sampler == "unipc":
        return LD3UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    raise ValueError(f"Unsupported sampler: {sampler}")


def get_dmn_timesteps(pipe, num_inference_steps: int, t_0: float, optimized_type: str, device: str) -> list[int]:
    alphas_cumprod = pipe.scheduler.alphas_cumprod.detach().clone().to(dtype=torch.float32)
    noise_schedule = NoiseScheduleVP("discrete", alphas_cumprod=alphas_cumprod)

    optimizer = StepOptim(noise_schedule)
    t, _ = optimizer.get_ts_lambdas(num_inference_steps, t_0, optimized_type)
    t = t.to(device).to(torch.float32)

    discrete_timesteps = (t * 1000.0).round().to(torch.int64)
    discrete_timesteps = discrete_timesteps[:-1].clamp_(0, len(alphas_cumprod) - 1)
    return discrete_timesteps.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Run DMN inference on COCO captions.")
    parser.add_argument("--outdir", type=str, default="./gen_img_val_v15_coco2014_dmn")
    parser.add_argument("--from-file", type=str, default="./instances_val2014.json", help="COCO instance annotation file.")
    parser.add_argument("--caption-file", type=str, default="./captions_val2014.json", help="COCO caption annotation file.")
    parser.add_argument("--num_prompts", type=int, default=10000, help="Number of COCO prompts to sample.")
    parser.add_argument("--ddim_steps", type=int, default=6, help="Number of DMN inference steps.")
    parser.add_argument("--dmn_t0", type=float, default=1e-3, help="Terminal VP time used by StepOptim.")
    parser.add_argument(
        "--optimized_type",
        type=str,
        default="unif_t",
        choices=["unif", "unif_t", "edm", "quad", "unif_origin", "unif_t_origin", "edm_origin", "quad_origin"],
        help="StepOptim initialization or optimized schedule type.",
    )
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=1, help="Batch size used for inference.")
    parser.add_argument("--scale", type=float, default=5.5)
    parser.add_argument("--model_id", type=str, default="sd-legacy/stable-diffusion-v1-5")
    parser.add_argument("--sampler", type=str, choices=["dpmpp", "unipc"], default="unipc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast")
    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    seed_everything(opt.seed)

    device = "cuda" if torch.cuda.is_available() else accelerator.device.type
    dtype = torch.float32

    print(f"reading prompts from {opt.from_file}")
    prompts = load_coco_prompts(opt.from_file, opt.caption_file, opt.num_prompts)

    pipe = StableDiffusionPipeline.from_pretrained(opt.model_id, safety_checker=None)
    pipe.scheduler = build_scheduler(pipe, opt.sampler)
    pipe.to(device=device, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    dmn_schedule = get_dmn_timesteps(
        pipe=pipe,
        num_inference_steps=opt.ddim_steps,
        t_0=opt.dmn_t0,
        optimized_type=opt.optimized_type,
        device=device,
    )
    print(f"Using DMN timesteps: {dmn_schedule}")

    folder_name = f"samples-dmn-coco-{opt.ddim_steps}-{opt.scale}-{opt.optimized_type}-{opt.sampler}"
    sample_path = os.path.join(opt.outdir, folder_name)
    dmn_path = os.path.join(sample_path, "dmn")
    os.makedirs(dmn_path, exist_ok=True)

    metadata_path = os.path.join(sample_path, "prompts.jsonl")
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    with torch.no_grad():
        with precision_scope("cuda") if device == "cuda" else nullcontext():
            tic = time.time()
            for start_index in tqdm(range(0, len(prompts), opt.n_samples), desc="Sampling", disable=not accelerator.is_main_process):
                prompt_batch = prompts[start_index:start_index + opt.n_samples]
                base_seed = opt.seed + start_index
                torch.manual_seed(base_seed)
                latents = torch.randn(
                    [len(prompt_batch), pipe.unet.config.in_channels, opt.H // 8, opt.W // 8],
                    device=device,
                    dtype=pipe.unet.dtype,
                )

                dmn_images = pipe(
                    prompt=prompt_batch,
                    timesteps=dmn_schedule,
                    guidance_scale=opt.scale,
                    height=opt.H,
                    width=opt.W,
                    latents=latents,
                ).images

                save_images(dmn_images, dmn_path, start_index)

                metadata_records = []
                for offset, prompt in enumerate(prompt_batch):
                    image_index = start_index + offset
                    metadata_records.append(
                        {
                            "index": image_index,
                            "seed": base_seed + offset,
                            "prompt": prompt,
                            "sampler": opt.sampler,
                            "optimized_type": opt.optimized_type,
                            "dmn_t0": opt.dmn_t0,
                            "dmn_timesteps": dmn_schedule,
                            "dmn_file": os.path.join("dmn", f"{image_index:05}.png"),
                        }
                    )
                append_metadata(metadata_path, metadata_records)

            toc = time.time()

    print(f"Saved DMN samples to: {sample_path}")
    print(f"Elapsed time: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()