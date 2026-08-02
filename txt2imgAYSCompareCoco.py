import argparse
import json
import os
import random
import time
from contextlib import nullcontext

import accelerate
import torch
from PIL import Image
from pycocotools.coco import COCO
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from txt2imgLD3CompareCoco import LD3UniPCMultistepScheduler

try:
    from diffusers.schedulers.scheduling_utils import AysSchedules
except ImportError:
    AysSchedules = {
        "StableDiffusionTimesteps": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24],
    }


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


def main():
    parser = argparse.ArgumentParser(description="Run official AYS inference on COCO captions.")
    parser.add_argument("--outdir", type=str, default="./gen_img_val_v15_coco2014_ays_compare")
    parser.add_argument("--from-file", type=str, default="./instances_val2014.json", help="COCO instance annotation file.")
    parser.add_argument("--caption-file", type=str, default="./captions_val2014.json", help="COCO caption annotation file.")
    parser.add_argument("--num_prompts", type=int, default=10000, help="Number of COCO prompts to sample.")
    parser.add_argument("--ddim_steps", type=int, default=10, help="AYS step count. Must match the official AYS schedule length.")
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=1, help="Batch size used for inference.")
    parser.add_argument("--scale", type=float, default=5.5)
    parser.add_argument("--model_id", type=str, default="sd-legacy/stable-diffusion-v1-5")
    parser.add_argument("--sampler", type=str, choices=["dpmpp", "unipc"], default="dpmpp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast")
    opt = parser.parse_args()

    ays_schedule = AysSchedules["StableDiffusionTimesteps"]
    if opt.ddim_steps != len(ays_schedule):
        raise ValueError(
            f"Official Stable Diffusion AYS uses {len(ays_schedule)} steps, but --ddim_steps={opt.ddim_steps}."
        )

    accelerator = accelerate.Accelerator()
    seed_everything(opt.seed)

    device = "cuda" if torch.cuda.is_available() else accelerator.device.type
    dtype = torch.float32

    print(f"reading prompts from {opt.from_file}")
    prompts = load_coco_prompts(opt.from_file, opt.caption_file, opt.num_prompts)

    pipe = StableDiffusionPipeline.from_pretrained(opt.model_id,safety_checker=None)
    pipe.scheduler = build_scheduler(pipe, opt.sampler)
    pipe.to(device=device, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    folder_name = f"samples-ays-coco-{opt.ddim_steps}-{opt.scale}-{opt.sampler}"
    sample_path = os.path.join(opt.outdir, folder_name)
    ays_path = os.path.join(sample_path, "ays")
    os.makedirs(ays_path, exist_ok=True)

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

                ays_images = pipe(
                    prompt=prompt_batch,
                    timesteps=ays_schedule,
                    guidance_scale=opt.scale,
                    height=opt.H,
                    width=opt.W,
                    latents=latents,
                ).images

                save_images(ays_images, ays_path, start_index)

                metadata_records = []
                for offset, prompt in enumerate(prompt_batch):
                    image_index = start_index + offset
                    metadata_records.append(
                        {
                            "index": image_index,
                            "seed": base_seed + offset,
                            "prompt": prompt,
                            "sampler": opt.sampler,
                            "ays_file": os.path.join("ays", f"{image_index:05}.png"),
                        }
                    )
                append_metadata(metadata_path, metadata_records)

            toc = time.time()

    print(f"Saved AYS samples to: {sample_path}")
    print(f"Elapsed time: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()