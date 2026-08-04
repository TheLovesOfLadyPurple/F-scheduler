import argparse
import json
import os
import random
import time
from contextlib import nullcontext
import math
import accelerate
import numpy as np
import torch
from pycocotools.coco import COCO
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm

from diffusers import IPNDMScheduler, StableDiffusionPipeline, UniPCMultistepScheduler


LD3_TIME_SCHEDULES = {
    5: [1.0000e00, 7.8365e-01, 5.5040e-01, 3.2539e-01, 1.0350e-01, 1.0000e-03],
    6: [1.0000e00, 8.5002e-01, 6.0043e-01, 3.9634e-01, 1.6598e-01, 4.5637e-02, 1.0000e-03],
    7: [1.0000e00, 8.5022e-01, 6.5012e-01, 4.6245e-01, 2.3517e-01, 1.2118e-01, 4.6843e-02, 1.0000e-03],
}

LD3_INPUT_TIME_SCHEDULES = {
    6: [1.0000e00, 8.5009e-01, 6.0054e-01, 3.9623e-01, 1.6583e-01, 4.5343e-02, 1.0000e-03],
}


class LD3UniPCMultistepScheduler(UniPCMultistepScheduler):
    def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None):
        if timesteps is None:
            super().set_timesteps(num_inference_steps=num_inference_steps, device=device)
            return

        timestep_array = np.asarray(timesteps, dtype=np.int64)
        if timestep_array.ndim != 1 or len(timestep_array) < 2:
            raise ValueError("Custom timesteps must be a 1D sequence with at least two entries.")
        if np.any(np.diff(timestep_array) > 0):
            raise ValueError("Custom timesteps must be in descending order.")

        sigmas = np.array(((1 - self.alphas_cumprod.cpu().numpy()) / self.alphas_cumprod.cpu().numpy()) ** 0.5)
        clipped_timesteps = np.clip(timestep_array, 0, len(sigmas) - 1)
        
        sigmas = np.interp(clipped_timesteps, np.arange(0, len(sigmas)), sigmas)
        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        # self.sigmas = torch.from_numpy(all_sigmas[clipped_timesteps])
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.tensor(clipped_timesteps, device=device, dtype=torch.int64)
        self.num_inference_steps = len(clipped_timesteps) - 1
        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")


class LD3IPNDMScheduler(IPNDMScheduler):
    def set_ld3_timesteps(self, solver_timesteps, input_timesteps, device=None):
        solver_array = np.asarray(solver_timesteps, dtype=np.float32)
        input_array = np.asarray(input_timesteps, dtype=np.float32)

        if solver_array.ndim != 1 or input_array.ndim != 1:
            raise ValueError("LD3 schedules must be 1D sequences.")
        if len(solver_array) < 2 or len(input_array) < 2:
            raise ValueError("LD3 schedules must contain at least two entries.")
        if len(solver_array) != len(input_array):
            raise ValueError("LD3 solver and input schedules must have the same length.")
        if np.any(np.diff(solver_array) > 0) or np.any(np.diff(input_array) > 0):
            raise ValueError("LD3 schedules must be in descending order.")

        solver_tensor = torch.tensor(solver_array, device=device, dtype=torch.float32)
        input_tensor = torch.tensor(input_array[:-1], device=device, dtype=torch.float32)

        self.num_inference_steps = len(solver_array) - 1
        self.timesteps = solver_tensor[:-1]
        self.input_timesteps = input_tensor
        if self.config.trained_betas is not None:
            self.betas = torch.tensor(self.config.trained_betas, dtype=torch.float32)
        else:
            self.betas = torch.sin(solver_tensor * math.pi / 2) ** 2

        self.alphas = (1.0 - self.betas**2) ** 0.5

        self.ets = []
        self._step_index = None
        self._begin_index = None


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


def get_ld3_schedules(ddim_steps: int) -> tuple[list[float], list[float]]:
    if ddim_steps not in LD3_TIME_SCHEDULES:
        supported = ", ".join(str(step) for step in sorted(LD3_TIME_SCHEDULES))
        raise ValueError(f"LD3 schedule supports ddim_steps in {{{supported}}}, but got {ddim_steps}.")

    if ddim_steps not in LD3_INPUT_TIME_SCHEDULES:
        supported = ", ".join(str(step) for step in sorted(LD3_INPUT_TIME_SCHEDULES))
        raise ValueError(f"LD3 input schedule supports ddim_steps in {{{supported}}}, but got {ddim_steps}.")

    return LD3_TIME_SCHEDULES[ddim_steps], LD3_INPUT_TIME_SCHEDULES[ddim_steps]


def run_ld3_ipndm(pipe, prompt_batch, latents, guidance_scale, solver_schedule, input_schedule, device):
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt_batch,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=None,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    pipe.scheduler.set_ld3_timesteps([t * 1000 for t in solver_schedule], [t * 1000 for t in input_schedule], device=device)
    latents = latents * pipe.scheduler.init_noise_sigma

    for solver_timestep, input_timestep in zip(pipe.scheduler.timesteps, pipe.scheduler.input_timesteps):
        latent_model_input = latents
        if do_classifier_free_guidance:
            latent_model_input = torch.cat([latents] * 2)

        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, solver_timestep)
        noise_pred = pipe.unet(
            latent_model_input,
            input_timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, solver_timestep, latents, return_dict=False)[0]

    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image


def main():
    parser = argparse.ArgumentParser(description="Run LD3 IPNDM inference on COCO captions.")
    supported_ld3_steps = sorted(set(LD3_TIME_SCHEDULES) & set(LD3_INPUT_TIME_SCHEDULES))
    parser.add_argument("--outdir", type=str, default="./gen_img_val_v15_coco2014_ld3_compare")
    parser.add_argument("--from-file", type=str, default="./instances_val2014.json", help="COCO instance annotation file.")
    parser.add_argument("--caption-file", type=str, default="./captions_val2014.json", help="COCO caption annotation file.")
    parser.add_argument("--num_prompts", type=int, default=10000, help="Number of COCO prompts to sample.")
    parser.add_argument("--ddim_steps", type=int, default=6, choices=supported_ld3_steps, help="Number of LD3 inference steps.")
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=1, help="Batch size used for inference.")
    parser.add_argument("--scale", type=float, default=5.5)
    parser.add_argument("--model_id", type=str, default="sd-legacy/stable-diffusion-v1-5")
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
    print(pipe.scheduler.config)
    pipe.scheduler = LD3IPNDMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device=device, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    ld3_time_schedule, ld3_input_time_schedule = get_ld3_schedules(opt.ddim_steps)
    print(f"Using LD3 solver schedule: {ld3_time_schedule}")
    print(f"Using LD3 input schedule: {ld3_input_time_schedule}")

    folder_name = f"samples-ld3-coco-{opt.ddim_steps}-{opt.scale}"
    sample_path = os.path.join(opt.outdir, folder_name)
    ld3_path = os.path.join(sample_path, "ld3")
    os.makedirs(ld3_path, exist_ok=True)

    metadata_path = os.path.join(sample_path, "prompts.jsonl")
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    with torch.no_grad():
        with precision_scope("cuda") if device == "cuda" else nullcontext():
            tic = time.time()
            for start_index in tqdm(
                range(0, len(prompts), opt.n_samples),
                desc="Sampling",
                disable=not accelerator.is_main_process,
            ):
                prompt_batch = prompts[start_index:start_index + opt.n_samples]
                base_seed = opt.seed + start_index
                torch.manual_seed(base_seed)
                latents = torch.randn(
                    [len(prompt_batch), pipe.unet.config.in_channels, opt.H // 8, opt.W // 8],
                    device=device,
                    dtype=pipe.unet.dtype,
                )

                ld3_images = run_ld3_ipndm(
                    pipe=pipe,
                    prompt_batch=prompt_batch,
                    latents=latents,
                    guidance_scale=opt.scale,
                    solver_schedule=ld3_time_schedule,
                    input_schedule=ld3_input_time_schedule,
                    device=device,
                )

                save_images(ld3_images, ld3_path, start_index)

                metadata_records = []
                for offset, prompt in enumerate(prompt_batch):
                    image_index = start_index + offset
                    metadata_records.append(
                        {
                            "index": image_index,
                            "seed": base_seed + offset,
                            "prompt": prompt,
                            "ld3_time_schedule": ld3_time_schedule,
                            "ld3_input_time_schedule": ld3_input_time_schedule,
                            "ld3_file": os.path.join("ld3", f"{image_index:05}.png"),
                        }
                    )
                append_metadata(metadata_path, metadata_records)

            toc = time.time()

    print(f"Saved LD3 samples to: {sample_path}")
    print(f"Elapsed time: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()