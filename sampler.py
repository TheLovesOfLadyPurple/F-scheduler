"""SAMPLING ONLY."""

import torch

from dpm_solver_v3 import NoiseScheduleVP, model_wrapper, DPM_Solver_v3
from uni_pc import UniPC
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d


class DPMSolverv3Sampler:
    def __init__(self, stats_dir, pipe, steps, guidance_scale, **kwargs):
        super().__init__()
        self.model = pipe
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(pipe.device)
        DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse
        device = "cuda" 
        noise_scheduler = pipe.scheduler
        alpha_schedule = noise_scheduler.alphas_cumprod.to(device=device, dtype=DTYPE)
        self.alphas_cumprod = alpha_schedule #to_torch(model.alphas_cumprod)
        self.device = device
        self.guidance_scale = guidance_scale

        self.ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)

        assert stats_dir is not None, f"No statistics file found in {stats_dir}."
        print("Use statistics", stats_dir)
        self.dpm_solver_v3 = DPM_Solver_v3(
            statistics_dir=stats_dir,
            noise_schedule=self.ns,
            steps=steps,
            t_start=None,
            t_end=None,
            skip_type="customed_time_karras",
            degenerated=False,
            device=self.device,
        )
        self.steps = steps

    @torch.no_grad()
    def apply_free_unet(self):
        register_free_upblock2d(self.model, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
        register_free_crossattn_upblock2d(self.model, b1=1.1, b2=1.1, s1=0.9, s2=0.2)

    @torch.no_grad()
    def stop_free_unet(self):
        register_free_upblock2d(self.model, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
        register_free_crossattn_upblock2d(self.model, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
    
    @torch.no_grad()
    def sample(
        self,
        batch_size,
        shape,
        conditioning=None,
        x_T=None,
        unconditional_conditioning=None,
        use_corrector=False,
        half=False,
        start_free_u_step=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
            cond_in = torch.cat([unconditional_conditioning, conditioning])
            # extra_args = {'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': self.guidance_scale}
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        if x_T is None:
            img = torch.randn(size, device=self.device)
        else:
            img = x_T

        if conditioning is None:
            model_fn = model_wrapper(
                lambda x, t, c: self.model.unet(x, t, encoder_hidden_states=c).sample,
                self.ns,
                model_type="noise",
                guidance_type="uncond",
            )
            ORDER = 3
        else:
            model_fn = model_wrapper(
                lambda x, t, c: self.model.unet(x, t, encoder_hidden_states=c).sample,
                self.ns,
                model_type="noise",
                guidance_type="classifier-free",
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                guidance_scale=self.guidance_scale,
            )
            if self.steps == 8:
                ORDER = 2
            else:
                ORDER = 1

        x = self.dpm_solver_v3.sample(
            img,
            model_fn,
            order=ORDER,
            p_pseudo=False,
            c_pseudo=True,
            lower_order_final=True,
            use_corrector=use_corrector,
            start_free_u_step=start_free_u_step,
            free_u_apply_callback=self.apply_free_unet if start_free_u_step is not None else None,
            free_u_stop_callback=self.stop_free_unet if start_free_u_step is not None else None,
            half=half,
        )

        return x.to(self.device), None


class UniPCSampler:
    def __init__(self
                 , pipe
                 , model_closure
                 , steps
                 , guidance_scale
                 , denoise_to_zero=False
                 , need_fp16_discrete_method = False
                 , ultilize_vae_in_fp16 = False
                 , is_high_resoulution = True
                 , skip_type="customed_time_karras"
                 , force_not_use_afs=False
                 , **kwargs):
        super().__init__()
        # self.model = pipe
        self.model = model_closure(pipe)
        self.pipe = pipe
        self.need_fp16_discrete_method = need_fp16_discrete_method
        # to_torch = lambda x: x.clone().detach().to(torch.float32).to(pipe.device)
        DTYPE = self.pipe.unet.dtype  # torch.float16 works as well, but pictures seem to be a bit worse
        device = self.pipe.device 
        noise_scheduler = pipe.scheduler
        alpha_schedule = noise_scheduler.alphas_cumprod.to(device=device, dtype=DTYPE)
        self.alphas_cumprod = alpha_schedule #to_torch(model.alphas_cumprod)
        self.device = device
        self.guidance_scale = guidance_scale
        self.use_afs = steps <= 8 and is_high_resoulution and not force_not_use_afs

        self.ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)

        self.unipc_solver = UniPC(
            noise_schedule=self.ns,
            steps=steps,
            t_start=None,
            t_end=None,
            skip_type=skip_type,
            degenerated=False,
            use_afs=self.use_afs,
            device=self.device,
            denoise_to_zero=denoise_to_zero,
            need_fp16_discrete_method = self.need_fp16_discrete_method,
            ultilize_vae_in_fp16 = ultilize_vae_in_fp16,
            is_high_resoulution=is_high_resoulution,
        )
        self.steps = steps

    @torch.no_grad()
    def apply_free_unet(self):
        register_free_upblock2d(self.pipe, b1=1.2, b2=1.2, s1=0.9, s2=0.2)
        register_free_crossattn_upblock2d(self.pipe, b1=1.2, b2=1.2, s1=0.9, s2=0.2)

    @torch.no_grad()
    def stop_free_unet(self):
        register_free_upblock2d(self.pipe, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
        register_free_crossattn_upblock2d(self.pipe, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
    
    @torch.no_grad()
    def sample(
        self,
        batch_size,
        shape,
        conditioning=None,
        x_T=None,
        unconditional_conditioning=None,
        use_corrector=False,
        half=False,
        start_free_u_step=None,
        xl_preprocess_closure=None,
        npnet=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        new_img = None
        if xl_preprocess_closure is not None:
            prompt_embeds, cond_kwargs = xl_preprocess_closure(pipe=self.pipe,prompts = conditioning, need_cfg=True, device=self.device,negative_prompts=unconditional_conditioning)
        if x_T is None:
            img = torch.randn(size, device=self.device)
        else:
            img = x_T
        if xl_preprocess_closure is not None and npnet is not None:
            c, _ = prompt_embeds
            c = c.unsqueeze(0)  # add dummy dimension for npnet
            new_img = npnet(img, c)

        if conditioning is None:
            model_fn = model_wrapper(
                lambda x, t, c: self.model(x, t, c),
                self.ns,
                model_type="noise",
                guidance_type="uncond",
            )
            ORDER = 3
        else:
            model_fn = model_wrapper(
                lambda x, t, c: self.model(x, t, c),
                self.ns,
                model_type="noise",
                guidance_type="classifier-free",
                condition=conditioning if xl_preprocess_closure is None else prompt_embeds,
                unconditional_condition=unconditional_conditioning if xl_preprocess_closure is None else cond_kwargs,
                guidance_scale=self.guidance_scale,
            )
            if self.steps >= 7:
                ORDER = 2
            else:
                ORDER = 1

        x, full_cache = self.unipc_solver.sample(
            x=img,
            model_fn=model_fn,
            order=ORDER,
            use_corrector=use_corrector,
            lower_order_final=True,
            start_free_u_step=start_free_u_step,
            free_u_apply_callback=self.apply_free_unet if start_free_u_step is not None else None,
            free_u_stop_callback=self.stop_free_unet if start_free_u_step is not None else None,
            npnet_x=new_img if new_img is not None else None,
            npnet_scale=self.guidance_scale if new_img is not None else None,
            half=half,
        )

        return x.to(self.device), full_cache
    
    @torch.no_grad()
    def sample_mix(
        self,
        batch_size,
        shape,
        conditioning=None,
        x_T=None,
        unconditional_conditioning=None,
        use_corrector=False,
        half=False,
        start_free_u_step=None,
        xl_preprocess_closure=None,
        npnet=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        if xl_preprocess_closure is not None:
            prompt_embeds, cond_kwargs = xl_preprocess_closure(pipe=self.pipe,prompts = conditioning, need_cfg=True, device=self.device,negative_prompts=unconditional_conditioning)
        if x_T is None:
            img = torch.randn(size, device=self.device)
        else:
            img = x_T
        if xl_preprocess_closure is not None and npnet is not None:
            c, _ = prompt_embeds
            c = c.unsqueeze(0)  # add dummy dimension for npnet
            img = npnet(img, c)

        if conditioning is None:
            model_fn = model_wrapper(
                lambda x, t, c: self.model(x, t, c),
                self.ns,
                model_type="noise",
                guidance_type="uncond",
            )
            ORDER = 3
        else:
            model_fn = model_wrapper(
                lambda x, t, c: self.model(x, t, c),
                self.ns,
                model_type="noise",
                guidance_type="classifier-free",
                condition=conditioning if xl_preprocess_closure is None else prompt_embeds,
                unconditional_condition=unconditional_conditioning if xl_preprocess_closure is None else cond_kwargs,
                guidance_scale=self.guidance_scale,
            )
            if self.steps >= 8 and not self.need_fp16_discrete_method:
                ORDER = 2
            else:
                ORDER = 1

        x, full_cache = self.unipc_solver.sample_mix(
            x=img,
            model_fn=model_fn,
            order=ORDER,
            use_corrector=use_corrector,
            lower_order_final=True,
            start_free_u_step=start_free_u_step,
            free_u_apply_callback=self.apply_free_unet if start_free_u_step is not None else None,
            free_u_stop_callback=self.stop_free_unet if start_free_u_step is not None else None,
            half=half,
        )

        return x.to(self.device), full_cache