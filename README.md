## F-scheduler: illuminating the free-lunch design space for fast sampling of diffusion models [[Huggingface Repo](https://huggingface.co/spaces/coralLight/F-scheduler-4k)] [[Huggingface Repo](https://huggingface.co/spaces/coralLight/F-scheduler-DreamshaperXL)] [[Huggingface Repo](https://huggingface.co/spaces/coralLight/F-scheduler-DreamshaperXL)] 

<div align="center">
<img src=gen_img_val_xl/comparison_grid_small.jpg />
</div>


<div align="center">

<a href="https://arxiv.org/abs/2510.02390" style="display: inline-block;">
    <img src="https://img.shields.io/badge/arXiv%20paper-2510.02390-b31b1b.svg" alt="arXiv" style="height: 20px; vertical-align: middle;">
</a>&nbsp;

</div>

**Abstract:** Diffusion models are the state-of-the-art generative models for high-resolution images, but sampling from pretrained models is computationally expensive, motivating interest in fast sampling. Although Free-U Net is a training-free enhancement for improving image quality, we find it ineffective under few-step ($<10$) sampling. We analyze the discrete diffusion ODE and propose F-scheduler, a scheduler designed for ODE solvers with Free-U Net.  Our proposed scheduler consists of a special time schedule that does not fully denoise the feature to enable the use of the KL-term in the $\beta$-VAE decoder, and the schedule of a proper inference stage for modifying the U-Net skip-connection via Free-U Net. Via information theory, we provide insights into how the better scheduled ODE solvers for the diffusion model can outperform the training-based diffusion distillation model. The newly proposed scheduler is compatible with most of the few-step ODE solvers and can sample a 1024 x 1024-resolution image in 6 steps and a 512 x 512-resolution image in 5 steps when it applies to DPM++ 2m and UniPC, with an FID result that outperforms the SOTA distillation models and the 20-step DPM++ 2m solver, respectively.  

```
python customed_timeschedule_sampler.py
python customed_timeschedule_sampler_xl.py
python customed_timeschedule_sampler_laion.py
python customed_timeschedule_sampler_xl_laion.py
```
 
## Requirements
This project use diffusers, which means you can simply install the environment by using pip install without confronting any conflict.  
```
conda env create --name hyper python=3.9
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt 
```
To find a proper version of torch, please use the following link:
https://pytorch.org/get-started/locally/

In this project, we also provide a upgrade implementation, which ultilize the golden noise to generate the images.  The checkpoint is in: https://1drv.ms/u/c/4e158dd7b255cd87/EaI2QngMC_lArhWGcjG5v7ABSm-3z8-Tm_sd2dN5nNIAYQ?e=tNKvzR and https://1drv.ms/u/c/4e158dd7b255cd87/EYzPIaAnN9dEpmxvHfys7M0Bv8_qsIGdt9wMf5yosMNq2w?e=t5Fd6b .  To run the code properly, you should also download coco 2014 and coco 2017 dataset from https://cocodataset.org/#home 
And the fp16 vae is in https://huggingface.co/madebyollin/sdxl-vae-fp16-fix .  or download it directly from the following link: https://1drv.ms/u/c/4e158dd7b255cd87/ETUoIRuJcJxBhcWA4yq0_kIBwXoU0WRxXcpp6Z5QU2w9iA?e=Vo9p2I
You could also try to use fp32 vae.
Meanwhile, you can use the counterfeit v3.0 to generate acgn image.  The result is in ./gen_img_val_v15. The model is in: https://civitai.com/models/4468/counterfeit-v30 .  You should place the model into the ./counterfeit


## Text to Image XL Version
using the following command to generate images from the new solver:


<img src=gen_img_val_xl/samples-customedXL-8-retrain-free-full-trick-1-7.5/00001.png width=512 />

```
python txt2imgXL.py --prompt "a painting of a virus monster playing guitar" --n_samples 1 --n_iter 1 --scale 7.5  --stop_steps 8
```

using the following command to generate images from the original solver:

<img src=gen_img_val_xl/samples-org-50-notNPNet/00001.png width=512 />

```
python txt2imgOrgXL.py --prompt "a painting of a virus monster playing guitar" --n_samples 1 --n_iter 1 --scale 7.5  --ddim_steps 50
```
we also provide a 6 step version

<img src=outputs/txt2img-samples-xl/samples-unipc/00000.png width=512 />

```
python txt2imgUniPCXL.py --prompt "a painting of a virus monster playing guitar" --n_samples 1 --n_iter 1 --scale 7.5  --stop_steps 6
```

## Text to Image
using the following command to generate images from the new solver:


<img src=gen_img_val_v15/samples-customed-8-notNPNet-full-trick-5.0/00000.png width=512 />

```
python txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --n_samples 4 --n_iter 4 --scale 5.0  --stop_steps 8
```

using the following command to generate images from the original solver:

<img src=gen_img_val_v15/samples-org-50-notNPNet/00000.png width=512 />

```
python txt2imgOrg.py --prompt "a virus monster is playing guitar, oil on canvas" --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50
```

we also provide 5 step method:


<img src=gen_img_val_v15/samples-customed-5-notNPNet-full-trick-7.5/00000.png width=512 />

```
python txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --n_samples 4 --n_iter 4 --scale 7.5  --stop_steps 5
```

<!-- ## Text to Image ACGN version
using the following command to generate images from the new solver:

<img src=gen_img_val_v15/samples-customed-8-free-notNPNet-full-trick-7.5/00002.png width=768 />

```
python txt2imgACGN.py --prompt "((masterpiece,best quality)) , 1girl, ((school uniform)),brown blazer, black skirt,small breasts,necktie,red plaid skirt,looking at viewer" --ddim_steps 20 --n_samples 4 --n_iter 1 --scale 7.5 --W 768 --H 1024 --use_free
```

using the following command to generate images from the original solver:

<img src=gen_img_val_v15/samples-org-20-free-notNPNet/00002.png width=768 />

```
 python txt2imgOrg.py --prompt "((masterpiece,best quality)) , 1girl, ((school uniform)),brown blazer, black skirt,small breasts,necktie,red plaid skirt,looking at viewer" --ddim_steps 20 --n_samples 4 --n_iter 1 --scale 7.5 --W 768 --H 1024 --use_free --is_acgn
``` -->
<!-- ```
python txt2imgXL.py --prompt "((masterpiece,best quality)) , ((1girl)), ((school uniform)),brown blazer, black skirt, necktie,red plaid skirt,looking at viewer, masterpiece, best quality, ultra-detailed, 8k resolution, high dynamic range, absurdres, stunningly beautiful, intricate details, sharp focus, detailed eyes, cinematic color grading, high-resolution texture,photorealistic portrait, nails" --n_samples 1 --n_iter 1 --scale 5.5 --W 1024 --H 1024 --stop_step 8
``` -->