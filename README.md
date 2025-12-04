## Hyperparameters-are-all-you-need [[Huggingface Repo](https://huggingface.co/spaces/coralLight/Hyperparameters-are-all-you-need-UniPC-Dreamshaper)]

<div align="center">
<img src=gen_img_val_xl/comparison_grid_small.jpg />
</div>


<div align="center">

<a href="https://arxiv.org/abs/2510.02390" style="display: inline-block;">
    <img src="https://img.shields.io/badge/arXiv%20paper-2510.02390-b31b1b.svg" alt="arXiv" style="height: 20px; vertical-align: middle;">
</a>&nbsp;

</div>

**Abstract:** The diffusion model is a state-of-the-art generative model that generates an image by applying a neural network iteratively. Moreover, this generation process is regarded as an algorithm solving an ordinary differential equation or a stochastic differential equation. Based on the analysis of the truncation error of the diffusion ODE and SDE, our study proposes a training-free algorithm that generates high-quality 512 x 512 and 1024 x 1024 images in eight steps, with flexible guidance scales. To the best of my knowledge, our algorithm is the first one that samples a 1024 x 1024 resolution image in 8 steps with an FID performance comparable to that of the latest distillation model, but without additional training. Meanwhile, our algorithm can also generate a 512 x 512 image in 8 steps, and its FID performance is better than the inference result using state-of-the-art ODE solver DPM++ 2m in 20 steps. We validate our eight-step image generation algorithm using the COCO 2014, COCO 2017, and LAION datasets. And our best FID performance is 15.7, 22.35, and 17.52. While the FID performance of DPM++2m is 17.3, 23.75, and 17.33. Further, it also outperforms the state-of-the-art AMED-plugin solver, whose FID performance is 19.07, 25.50, and 18.06. We also apply the algorithm in five-step inference without additional training, for which the best FID performance in the datasets mentioned above is 19.18, 23.24, and 19.61, respectively, and is comparable to the performance of the state-of-the-art AMED Pulgin solver in eight steps, SDXL-turbo in four steps, and the state-of-the-art diffusion distillation model Flash Diffusion in five steps. We also validate our algorithm in synthesizing 1024 * 1024 images within 6 steps, whose FID performance only has a limited distance to the latest distillation algorithm. 

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

<img src=gen_img_val_xl/samples-customedXL-6-retrain-free-full-trick-1-7.5/00000.png width=512 />

```
python txt2imgXL.py --prompt "a painting of a virus monster playing guitar" --n_samples 1 --n_iter 1 --scale 7.5  --stop_steps 6
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