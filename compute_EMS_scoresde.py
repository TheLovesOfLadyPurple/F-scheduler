"""Training and evaluation"""

from absl import app
from absl import flags
import argparse, os
import logging
import os
import torch
import random
from models.utils import get_noise_fn
import torch.autograd.forward_ad as fwAD
from samplers.utils import NoiseScheduleVP
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from pycocotools.coco import COCO
import sde_lib
import numpy as np
from tqdm import tqdm
import time
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
from PIL import Image
import functools
from torch.nn.attention import sdpa_kernel, SDPBackend

def get_data_scaler():
    """Data normalizer. Assume data are always in [0, 1]."""
    return lambda x: x * 2.0 - 1.0

def get_time_steps(ns, skip_type, t_T, t_0, N, device):
    """Compute the intermediate time steps for sampling.

    Args:
        skip_type: A `str`. The type for the spacing of the time steps. We support three types:
            - 'logSNR': uniform logSNR for the time steps.
            - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
            - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
        t_T: A `float`. The starting time of the sampling (default is T).
        t_0: A `float`. The ending time of the sampling (default is epsilon).
        N: A `int`. The total number of the spacing of the time steps.
        device: A torch device.
    Returns:
        A pytorch tensor of the time steps, with the shape (N + 1,).
    """
    if skip_type == "logSNR":
        lambda_T = ns.marginal_lambda(torch.tensor(t_T).to(device))
        lambda_0 = ns.marginal_lambda(torch.tensor(t_0).to(device))
        logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
        return ns.inverse_lambda(logSNR_steps)
    elif skip_type == "time_uniform":
        return torch.linspace(t_T, t_0, N + 1).to(device)
    elif skip_type == "time_quadratic":
        t_order = 2
        t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
        return t
    else:
        raise ValueError(
            "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type)
        )


@torch.no_grad()
def get_noise_and_jvp_x(x, t, v, noise_pred_fn, uc):
    def fn(data):
        return noise_pred_fn(data, torch.ones(data.shape[0], device=data.device) * t, encoder_hidden_states=uc)
    noise, noise_jvp_x = torch.autograd.functional.jvp(fn, x, v=v, create_graph=False, strict=False)
    # with sdpa_kernel(SDPBackend.MATH):
    #     with fwAD.dual_level():
    #         dual_x = fwAD.make_dual(x, v)
    #         noise, noise_jvp_x = fwAD.unpack_dual(fn(dual_x))
    return noise, noise_jvp_x


@torch.no_grad()
def get_noise_and_total_derivative(x, t, noise_pred_fn, ns):
    def fn(data, time):
        return noise_pred_fn(data, torch.ones(x.shape[0], device=x.device) * time)

    alpha_t, sigma_t = ns.marginal_alpha(t), ns.marginal_std(t)
    with fwAD.dual_level():
        vt = torch.ones_like(t)
        dual_t = fwAD.make_dual(t, vt)
        _, d_lambda_d_t = fwAD.unpack_dual(ns.marginal_lambda(dual_t))
        noise = fn(x, t)

        vt = torch.ones_like(t) / d_lambda_d_t
        vx = sigma_t**2 * x - sigma_t * noise
        dual_x = fwAD.make_dual(x, vx)
        dual_t = fwAD.make_dual(t, vt)
        _, noise_jvp = fwAD.unpack_dual(fn(dual_x, dual_t))
    return noise, noise_jvp


def compute_l(
    statistics_dir, MAX_BATCH, eps, n_timesteps, batch_size, num_gpus, ns, sde, device, r, pipe, dataset
):
    torch.cuda.set_device(r)
    # config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scaler = get_data_scaler()
    train_ds = dataset#, _, _ = get_dataset_multi_host(config, batch_size, num_slices=num_gpus, slice=r)
    score_model = pipe.unet #mutils.create_model(config)
    # optimizer = losses.get_optimizer(config, score_model.parameters())
    # ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    # state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    # state = restore_checkpoint(state, loaded_state)
    # ema.copy_to(score_model.parameters())
    noise_pred_fn = get_noise_fn(sde, score_model, train=False, continuous=True)
    timesteps = get_time_steps(ns, "logSNR", sde.T, eps, n_timesteps, device)
    if os.path.exists(os.path.join(statistics_dir, f"l_{r}.npz")):
        return
    l_lst = [0] * len(timesteps)
    
    def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
        prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
        return {"prompt_embeds": prompt_embeds}
    
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
    )
    uc = compute_embeddings_fn(batch_size * [""])
    uc = uc.pop("prompt_embeds") if uc is not None else None
    with torch.no_grad():
        for j, t in tqdm(enumerate(timesteps), desc="Computing l..."):
            time_start = time.time()
            for i, batch in enumerate(train_ds):
                if i >= MAX_BATCH:
                    break
                time_spent = time.time() - time_start
                
                img = Image.open(batch).convert('RGBA')
                img = img.resize((512,512), Image.Resampling.BICUBIC)
                img = [img]
                # Convert to numpy array with values in [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0

                print(f"Batch {i}/{MAX_BATCH}, {time_spent:.2f} s")
                train_batch = torch.from_numpy(img_array).to(device).float()
                train_batch = train_batch.permute(0, 3, 1, 2)
                x = scaler(train_batch)

                v = torch.randint(0, 2, x.shape, device=device) * 2.0 - 1
                z = torch.randn_like(x)
                alpha_t, sigma_t = ns.marginal_alpha(t), ns.marginal_std(t)
                perturbed_data = alpha_t * x + sigma_t * z
                _, noise_jvp_x = get_noise_and_jvp_x(perturbed_data, t, v, noise_pred_fn, uc=uc)
                l = (sigma_t * noise_jvp_x * v).mean(dim=0).cpu().numpy()
                l_lst[j] += l
            l_lst[j] = l_lst[j] / MAX_BATCH
    l_lst = np.asarray(l_lst)
    np.savez_compressed(os.path.join(statistics_dir, f"l_{r}.npz"), l=l_lst)


def collect_l(statistics_dir):
    print("Collecting l...")
    l_lsts = []
    for file in os.listdir(statistics_dir):
        if file.startswith("l_") and not file.startswith("l_d"):
            l_lst = np.load(os.path.join(statistics_dir, file))["l"]
            l_lsts.append(l_lst)
    np.savez_compressed(os.path.join(statistics_dir, "l.npz"), l=np.mean(l_lsts, axis=0))


def compute_l_d(statistics_dir, lambda_0, lambda_T):
    l_lst = np.load(os.path.join(statistics_dir, "l.npz"))["l"]
    print("Computing l_d...")
    l_len = len(l_lst)
    l_d_lst = []
    gap = (lambda_0 - lambda_T) / (l_len - 1)
    for i in range(l_len):
        if i == 0:
            l_d_lst.append((l_lst[i + 1] - l_lst[i]) / gap)
        elif i == l_len - 1:
            l_d_lst.append((l_lst[i] - l_lst[i - 1]) / gap)
        else:
            l_d_lst.append((l_lst[i + 1] - l_lst[i - 1]) / (2 * gap))

    window = 5
    l_d_smooth_lst = []
    for i in range(l_len):
        if i < window:
            l_d_smooth_lst.append(np.sum(l_d_lst[: i + window + 1], axis=0) / (i + window + 1))
        elif i >= l_len - window:
            l_d_smooth_lst.append(np.sum(l_d_lst[i - window : l_len], axis=0) / (l_len - i + window))
        else:
            l_d_smooth_lst.append(np.sum(l_d_lst[i - window : i + window + 1], axis=0) / (2 * window + 1))
    l_d_smooth_lst = np.asarray(l_d_smooth_lst)
    np.savez_compressed(os.path.join(statistics_dir, "l_d.npz"), l_d=l_d_smooth_lst)


def compute_f(
    statistics_dir, MAX_BATCH, eps, n_timesteps, batch_size, num_gpus, ns, sde, device, r, pipe, dataset
):
    torch.cuda.set_device(r)
    # config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scaler = get_data_scaler()
    train_ds = dataset #, _, _ = get_dataset_multi_host(config, batch_size, num_slices=num_gpus, slice=r)
    score_model = pipe.unet #mutils.create_model(config)
    # optimizer = losses.get_optimizer(config, score_model.parameters())
    # ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    # state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    # state = restore_checkpoint(state, loaded_state)
    # ema.copy_to(score_model.parameters())
    noise_pred_fn = get_noise_fn(sde, score_model, train=False, continuous=True)
    timesteps = get_time_steps(ns, "logSNR", sde.T, eps, n_timesteps, device)
    if os.path.exists(os.path.join(statistics_dir, f"f_{r}.npz")):
        return
    l_lst = np.load(os.path.join(statistics_dir, "l.npz"))["l"]
    l_d_lst = np.load(os.path.join(statistics_dir, "l_d.npz"))["l_d"]
    f = [0] * len(timesteps)
    f_d = [0] * len(timesteps)
    f_f = [0] * len(timesteps)
    f_f_d = [0] * len(timesteps)

    with torch.no_grad():
        for j, t in tqdm(enumerate(timesteps), desc="Computing f..."):
            time_start = time.time()
            for i, batch in enumerate(train_ds):
                if i >= MAX_BATCH:
                    break
                time_spent = time.time() - time_start
                img = Image.open(batch).convert('RGB')
                img = img.resize((512,512), Image.Resampling.BICUBIC)
                img = [img]
                # Convert to numpy array with values in [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0

                print(f"Batch {i}/{MAX_BATCH}, {time_spent:.2f} s")
                train_batch = torch.from_numpy(img_array).to(device).float()
                train_batch = train_batch.permute(0, 3, 1, 2)
                x = scaler(train_batch)

                z = torch.randn_like(x)
                alpha_t, sigma_t = ns.marginal_alpha(t), ns.marginal_std(t)
                perturbed_data = alpha_t * x + sigma_t * z
                noise, eps_d = get_noise_and_total_derivative(perturbed_data, t, noise_pred_fn, ns)
                l = torch.from_numpy(l_lst[j]).to(device)
                l_d = torch.from_numpy(l_d_lst[j]).to(device)
                lamb = ns.marginal_lambda(t)

                a = (sigma_t * noise - l * perturbed_data) / alpha_t
                b = torch.exp(-lamb) * ((l - 1) * noise + eps_d) - l_d * perturbed_data / alpha_t
                f[j] += a.mean(dim=0).cpu().numpy()
                f_d[j] += b.mean(dim=0).cpu().numpy()
                f_f[j] += (a * a).mean(dim=0).cpu().numpy()
                f_f_d[j] += (a * b).mean(dim=0).cpu().numpy()
            f[j] /= MAX_BATCH
            f_d[j] /= MAX_BATCH
            f_f[j] /= MAX_BATCH
            f_f_d[j] /= MAX_BATCH
    f = np.asarray(f)
    f_d = np.asarray(f_d)
    f_f = np.asarray(f_f)
    f_f_d = np.asarray(f_f_d)
    np.savez_compressed(os.path.join(statistics_dir, f"f_{r}.npz"), f=f, f_d=f_d, f_f=f_f, f_f_d=f_f_d)


def collect_f(statistics_dir):
    print("Collecting f...")
    f_lsts, f_d_lsts, f_f_lsts, f_f_d_lsts = [], [], [], []
    for file in os.listdir(statistics_dir):
        if file.startswith("f_"):
            fs_lsts = np.load(os.path.join(statistics_dir, file))
            f_lst, f_d_lst, f_f_lst, f_f_d_lst = fs_lsts["f"], fs_lsts["f_d"], fs_lsts["f_f"], fs_lsts["f_f_d"]
            f_lsts.append(f_lst)
            f_d_lsts.append(f_d_lst)
            f_f_lsts.append(f_f_lst)
            f_f_d_lsts.append(f_f_d_lst)
    np.savez_compressed(
        os.path.join(statistics_dir, "f.npz"),
        f=np.mean(f_lsts, axis=0),
        f_d=np.mean(f_d_lsts, axis=0),
        f_f=np.mean(f_f_lsts, axis=0),
        f_f_d=np.mean(f_f_d_lsts, axis=0),
    )

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


def compute_sb(statistics_dir):
    print("Computing s, b...")
    fs_lst = np.load(os.path.join(statistics_dir, "f.npz"))
    f, f_d, f_f, f_f_d = fs_lst["f"], fs_lst["f_d"], fs_lst["f_f"], fs_lst["f_f_d"]
    s = (f_f_d - f * f_d) / (f_f - f * f)
    b = f_d - s * f
    np.savez_compressed(os.path.join(statistics_dir, "sb.npz"), s=s, b=b)


def compute_lsb(opt):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    coco_annotation_file_path = './instances_train2014.json'
    coco_caption_file_path = './captions_train2014.json'
    coco_annotation = COCO(annotation_file=coco_annotation_file_path)
    coco_caption = COCO(annotation_file=coco_caption_file_path)
    
    img_ids = coco_annotation.getImgIds()
    

    # 过滤掉包含不需要类别的图像ID
    real_img_ids = [item for item in img_ids]
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
    data = data[(0):20000]
    images = coco_caption.loadImgs(ids=real_img_ids)
    images = images[0:10000]
    folder_name = 'E:\\txt2img-samples\\scls_coco_img_train_random'
    img_path = 'D:\\train2014'
    data_path_list = []
    # if not os.path.exists(folder_name):
    #     os.makedirs(name=folder_name,exist_ok=True)
    img_file_name = [ img['file_name'] for img in images ]
    for filename in os.listdir(path=img_path):
        if filename in img_file_name:
            data_path_list.append(os.path.join(img_path, filename))
    
    DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse
    device = "cuda" 
    # pipe = StableDiffusionPipeline.from_single_file( "./counterfeit/Counterfeit-V3.0_fp32.safetensors")
    
    # pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    pipe = StableDiffusionPipeline.from_pretrained('sd-legacy/stable-diffusion-v1-5')
    # pipe.unet.set_attn_processor(AttnProcessor())
    pipe.to(device=device, torch_dtype=DTYPE)
    
    num_gpus = 1

    # Create data normalizer and its inverse
    workdir = opt.workdir
    
    noise_scheduler = pipe.scheduler
    alpha_schedule = noise_scheduler.alphas_cumprod.to(device=device, dtype=DTYPE)
    betas = noise_scheduler.betas.to(device=device, dtype=DTYPE)

    sde = sde_lib.VPSDE(beta_min=betas[0], beta_max=betas[-1], N=1000)
    

    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)
    timesteps = get_time_steps(ns, "logSNR", sde.T, 1e-3, 200, "cuda")
    logSNR_steps = ns.marginal_lambda(timesteps)
    lambda_T = ns.marginal_lambda(torch.tensor(sde.T)).item()
    lambda_0 = ns.marginal_lambda(torch.tensor(1e-3)).item()

    # begin_ckpt = config.eval.begin_ckpt
    # logging.info("begin checkpoint: %d" % (begin_ckpt,))
    # for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    #     ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{ckpt}.pth")
    #     if not tf.io.gfile.exists(ckpt_path):
    #         logging.warning(f"No checkpoint found at {ckpt_path}.")
    #         continue

    #     loaded_state = torch.load(ckpt_path, map_location="cpu")
    statistics_dir = os.path.join(
        workdir, "statistics", f"sdv1-5"
    )
    os.makedirs(statistics_dir, exist_ok=True)

    import torch.multiprocessing as mp
    compute_l(
                statistics_dir,
                4096,#opt.n_batch, #parallel it later
                # opt.config,
                1e-3,
                200,
                1,#opt.batch_size, #parallel it later
                num_gpus,
                ns,
                sde,
                #loaded_state,
                device,
                0,
                pipe,
                data_path_list
            )
    # mp.set_start_method(method="spawn", force=True)
    # processes_l = [
    #     mp.Process(
    #         target=compute_l,
    #         args=(
    #             statistics_dir,
    #             4096,#opt.n_batch, #parallel it later
    #             # opt.config,
    #             1e-3,
    #             200,
    #             1,#opt.batch_size, #parallel it later
    #             num_gpus,
    #             ns,
    #             sde,
    #             #loaded_state,
    #             device,
    #             i,
    #             pipe,
    #             data_path_list
    #         ),
    #     )
    #     for i in range(num_gpus)
    # ]

    # [p.start() for p in processes_l]
    # [p.join() for p in processes_l]

    collect_l(statistics_dir)

    compute_l_d(statistics_dir, lambda_0, lambda_T)
    compute_f(
                statistics_dir,
                4096,#opt.n_batch,
                1e-3,
                200,
                1,#opt.batch_size,
                num_gpus,
                ns,
                sde,
                device,
                0,
                pipe,
                data_path_list
            )
    # processes_f = [
    #     mp.Process(
    #         target=compute_f,
    #         args=(
    #             statistics_dir,
    #             4096,#opt.n_batch,
    #             1e-3,
    #             200,
    #             1,#opt.batch_size,
    #             num_gpus,
    #             ns,
    #             sde,
    #             device,
    #             i,
    #             pipe,
    #             data_path_list
    #         ),
    #     )
    #     for i in range(num_gpus)
    # ]

    # [p.start() for p in processes_f]
    # [p.join() for p in processes_f]

    collect_f(statistics_dir)

    compute_sb(statistics_dir)


def main(argv):
    parser = argparse.ArgumentParser()
    
    
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
        default='./instances_train2014.json',
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default='./ems',
        help="if specified, load prompts from this file",
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

    compute_lsb(opt)


if __name__ == "__main__":
    app.run(main)
