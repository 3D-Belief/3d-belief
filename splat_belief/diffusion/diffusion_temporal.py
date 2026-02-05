import math
from pathlib import Path
from random import random
from collections import OrderedDict
import numpy as np
from functools import partial
from collections import namedtuple
from omegaconf import OmegaConf

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam, AdamW
from torchvision import utils

from einops import rearrange, reduce, repeat
from torch.utils.data._utils.collate import default_collate

from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from accelerate.utils import broadcast
from torchvision.utils import make_grid

import sys
import os
import time
import wandb
import imageio
from copy import deepcopy
from PIL import Image
from accelerate import DistributedDataParallelKwargs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from splat_belief.utils.vision_utils import *
from splat_belief.splat.layers import *
import lpips

from scipy import interpolate
from splat_belief.utils.vision_utils import JsonLogger
from splat_belief.splat.splat_belief_state import SplatBeliefState

# constants
ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "pred_x_start_high_res"]
)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def prepare_video_viz(out):
    frames = out["videos"]
    for f in range(len(frames)):
        frames[f] = rearrange(frames[f], "b h w c -> h (b w) c")

    depth_frames = out["depth_videos"]
    for f in range(len(depth_frames)):
        depth_frames[f] = rearrange(depth_frames[f], "(n b) h w -> n h (b w)", n=1)

    depth = torch.cat(depth_frames, dim=0)
    print(f"depth shape: {depth.shape}")

    depth = (
        torch.from_numpy(
            jet_depth(depth[:].cpu().detach().view(depth.shape[0], depth.shape[-1], -1))
        )
        * 255
    )
    # convert depth to list of images
    depth_frames = []
    for i in range(depth.shape[0]):
        depth_frames.append(depth[i].cpu().detach().numpy().astype(np.uint8))

    # semantic
    semantics = out["semantic_videos"]

    return (
        frames,
        depth_frames,
        semantics
    )

class DiffusionTemporal(nn.Module):
    def __init__(
        self,
        model: SplatBeliefState,
        image_size: int,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type="l1",
        objective="pred_noise",
        beta_schedule="sigmoid",
        schedule_fn_kwargs=dict(),
        p2_loss_weight_gamma=0.0,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        ddim_sampling_eta=0.0,
        auto_normalize=True,
        use_guidance=False,
        guidance_scale=1.0,
        temperature=1.0,
        clean_target=False,
        use_depth_supervision=True,
        use_depth_mask=False,
        use_object_binary_mask=False,
        background_weight=0.3,
        use_semantic=False,
        use_reg_model=False,
    ):
        super().__init__()
        assert not (type(self) == DiffusionTemporal and model.channels != model.out_dim)

        self.model = model
        self.channels = self.model.channels
        self.temperature = temperature

        self.image_size = image_size
        self.use_guidance = use_guidance
        self.guidance_scale = guidance_scale
        self.objective = objective

        self.clean_target = clean_target
        self.use_depth_supervision = use_depth_supervision
        self.use_semantic = use_semantic
        self.use_depth_mask = use_depth_mask
        self.use_reg_model = use_reg_model
        self.use_object_binary_mask = use_object_binary_mask
        self.background_weight = background_weight

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting
        use_constant_p2_weight = True
        if use_constant_p2_weight:
            register_buffer(
                "p2_loss_weight",
                (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
                ** -p2_loss_weight_gamma,
            )
        else:
            snr = alphas_cumprod / (1 - alphas_cumprod)
            register_buffer(
                "p2_loss_weight", torch.minimum(snr, torch.ones_like(snr) * 5.0)
            )  # https://arxiv.org/pdf/2303.09556.pdf

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        self.perceptual_loss = lpips.LPIPS(net="vgg")

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        inp,
        t,
        clip_x_start=False,
        render_high_res=False,
        global_step=100000,
        state_t=None,
    ):
        x = inp["noisy_trgt_rgb"]
        _, _, _, h, w = inp["ctxt_rgb"].shape
        if self.use_guidance and self.guidance_scale > 1.0:
            model_uncond = deepcopy(self.model)
        self.model(inp, t=t, global_step=global_step, state_t=state_t) # Update history and belief
        
        # Augmented gaussians
        gaussians = self.model.history_gaussians + self.model.belief_gaussians
        output = self.model.decoder.forward(
            gaussians.float(),
            inp["trgt_c2w"],
            inp["intrinsics"].unsqueeze(1),
            inp["near"].float().unsqueeze(1),
            inp["far"].float().unsqueeze(1),
            (h, w),
            depth_mode=self.model.depth_mode,
        )

        if self.use_guidance and self.guidance_scale > 1.0:
            inpt_uncond = deepcopy(inp)
            inpt_uncond["ctxt_rgb"] = inpt_uncond["ctxt_rgb"] * 0.0
            context_gaussians_uncond, target_gaussians_uncond = model_uncond.encoder(
                inpt_uncond, 
                t=t,
                global_step=global_step, 
                deterministic=False
            )

            # Augmented gaussians
            gaussians_uncond = context_gaussians_uncond + target_gaussians_uncond

            output_uncond = model_uncond.decoder.forward(
                gaussians_uncond.float(),
                inp["trgt_c2w"],
                inp["intrinsics"].unsqueeze(1),
                inp["near"].float().unsqueeze(1),
                inp["far"].float().unsqueeze(1),
                (h, w),
                depth_mode=model_uncond.depth_mode,
            )

            uncond_model_output = self.normalize(output_uncond.color)[:, 0, ...]

        color = output.color
        # apply depth mask if available
        if self.use_depth_mask:
            depth_mask = self.model.encoder.get_depth_mask(output.features)
            # apply sigmoid to depth mask
            depth_mask = torch.sigmoid(depth_mask)
            # make the mask binary
            depth_mask = (depth_mask > 0.5).float()
            # apply the mask to the color
            color = color * depth_mask
        model_output = self.normalize(color)[:, 0, ...]

        dynamic_threshold = True
        dynamic_thresholding_percentile = 0.95
        if dynamic_threshold:
            # following pseudocode in appendix
            # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element

            def maybe_clip(x_start):
                s = torch.quantile(
                    rearrange(x_start, "b ... -> b (...)").abs(),
                    dynamic_thresholding_percentile,
                    dim=-1,
                )

                s.clamp_(min=1.0)
                s = right_pad_dims_to(x_start, s)
                x_start = x_start.clamp(-s, s) / s
                return x_start

        else:
            maybe_clip = (
                partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
            )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == "pred_x0":
            x_start = model_output

            if self.use_guidance and self.guidance_scale > 1.0:
                uncond_x_start = uncond_model_output
                x_start = uncond_x_start + self.guidance_scale * (x_start - uncond_x_start)

            x_start = maybe_clip(x_start)
            num_targets = x_start.shape[0] // x.shape[0]
            x_start = rearrange(x_start, "(b nt) c h w -> b nt c h w", nt=num_targets)[
                :, 0, ...
            ]
            x_start_high_res = None
            if render_high_res:
                x_start_high_res = x_start
                x_start = F.interpolate(
                    x_start, size=(64, 64), mode="bilinear", antialias=True,
                )
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start, x_start_high_res)

    def p_mean_variance(self, inp, t, clip_denoised=True, global_step=100000, state_t=None):
        x = inp["noisy_trgt_rgb"]
        preds = self.model_predictions(inp, t, global_step=global_step, state_t=state_t)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, inp, t: int, global_step=100000, state_t=None):
        x = inp["noisy_trgt_rgb"]
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            inp=inp, t=batched_times, clip_denoised=True, global_step=global_step, state_t=state_t,
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False, inp=None, global_step=100000, state_t=None):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        print("p sample loop")

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            if self.clean_target:
                inp["noisy_trgt_rgb"] = inp["trgt_rgb"][:, 0, :, :, :]
            else:
                inp["noisy_trgt_rgb"] = img

            img, x_start = self.p_sample(inp, t, global_step=global_step, state_t=state_t)
            imgs.append(img)

        if self.clean_target:
            inp["noisy_trgt_rgb"] = inp["trgt_rgb"][:, 0, :, :, :]
        else:
            inp["noisy_trgt_rgb"] = img

        time_embed = torch.full((1,), t, device=device, dtype=torch.long)
        frames, depth_frames, *_ = self.model.render_video(
            inp, time_embed, 20,
        )

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)

        out_dict = {
            "images": ret,
            "videos": frames,
            "depth_videos": depth_frames,
        }
        return out_dict

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps=False, inp=None, global_step=100000, state_t=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )
        """Normalize input images"""
        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        temperature = self.temperature  # 0.85
        img = torch.randn(shape, device=device) * temperature
        imgs = [img]
        x_start = None
        # """
        if "num_frames_render" in inp.keys():
            num_frames_render = inp["num_frames_render"]
        else:
            num_frames_render = 20

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            if self.clean_target:
                inp["noisy_trgt_rgb"] = inp["trgt_rgb"][:, 0, :, :, :]
            else:
                inp["noisy_trgt_rgb"] = img

            if time_next < 0:
                render_high_res = True
            else:
                render_high_res = False
            render_high_res = False
            pred_noise, x_start, _ = self.model_predictions(
                inp,
                time_cond,
                clip_x_start=True,
                render_high_res=render_high_res,
                global_step=global_step,
                state_t=state_t,
            )
            if time_next < 0:
                img = x_start
                imgs.append(img)

                # render the video
                depth_masks = None
                if "render_poses" not in inp:
                    frames = None
                    depth_frames = None
                    render_poses = None
                    semantics = None
                else:
                    if self.use_depth_mask:
                        frames, depth_frames, semantics, render_poses, depth_masks = self.model.render_video(
                            inp,
                            time_cond,
                            n=num_frames_render,
                            render_high_res=render_high_res,
                        )
                    else:
                        frames, depth_frames, semantics, render_poses = self.model.render_video(
                            inp,
                            time_cond,
                            n=num_frames_render,
                            render_high_res=render_high_res,
                        )
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img) * temperature
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)
        out_dict = {
            "images": ret,
            "videos": frames,
            "depth_masks": depth_masks,
            "depth_videos": depth_frames,
            "semantic_videos": semantics,
            "inp": inp,
            "render_poses": render_poses,
            "time_cond": time_cond,
        }
        return out_dict

    @torch.no_grad()
    def sample(self, batch_size=2, return_all_timesteps=False, inp=None, state_t=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        return sample_fn(
            (batch_size, channels, image_size, image_size),
            return_all_timesteps=return_all_timesteps,
            inp=inp,
            state_t=state_t,
        )

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(
            reversed(range(0, t)), desc="interpolation sample time step", total=t
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def compute_errors_ssim(self, img0, img1, mask=None):
        b, c, h, w = img0.shape
        assert img0.shape == img1.shape
        assert c == 3

        errors = torch.mean(
            ssim(
                img0, img1, pad_reflection=False, gaussian_average=True, comp_mode=True
            ),
            dim=1,
        )
        return errors

    def edge_aware_smoothness(self, gt_img, depth, mask=None):
        bd, hd, wd = depth.shape
        depth = depth.reshape(-1, 1, hd, wd)

        b, c, h, w = gt_img.shape
        assert bd == b and hd == h and wd == w

        depth = 1 / depth.reshape(-1, 1, h, w).clamp(1e-3, 80)
        depth = depth / torch.mean(depth, dim=[2, 3], keepdim=True)

        d_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        d_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

        i_dx = torch.mean(
            torch.abs(gt_img[:, :, :, :-1] - gt_img[:, :, :, 1:]), 1, keepdim=True
        )
        i_dy = torch.mean(
            torch.abs(gt_img[:, :, :-1, :] - gt_img[:, :, 1:, :]), 1, keepdim=True
        )

        d_dx *= torch.exp(-i_dx)
        d_dy *= torch.exp(-i_dy)

        errors = F.pad(d_dx, pad=(0, 1), mode="constant", value=0) + F.pad(
            d_dy, pad=(0, 0, 0, 1), mode="constant", value=0
        )
        errors = errors.view(b, h, w)
        return errors

    def p_losses(self, inp, t, noise=None, state_t=None):
        num_target = inp["trgt_rgb"].shape[1]
        num_context = inp["ctxt_rgb"].shape[1]
        x_start = inp["trgt_rgb"][:, 0, ...]
        noise = default(noise, lambda: torch.randn_like(x_start))

        # print("ctxt c2w: ", inp["ctxt_c2w"])
        # print("trgt c2w: ", inp["trgt_c2w"])

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step
        if self.clean_target:
            inp["noisy_trgt_rgb"] = inp["trgt_rgb"][:, 0, :, :, :]
        else:
            inp["noisy_trgt_rgb"] = x
        if self.use_depth_supervision:
            depth_gt = inp["trgt_depth"][:, 0, ...]
        if self.use_guidance:
            uncond = np.random.rand() > 0.9
            if uncond:
                inp["ctxt_rgb"] = inp["ctxt_rgb"] * 0.0
        else:
            uncond = False
        
        if self.use_depth_mask:
            depth_mask_gt = inp["trgt_depth_mask"][:, 0, ...]

        model_out, depth, misc = self.model(inp, t, state_t=state_t)

        (
            rendered_ctxt_img,
            rendered_trgt_img,
            rendered_ctxt_depth,
            rendered_trgt_depth,
        ) = (
            misc["rendered_ctxt_rgb"],
            misc["rendered_trgt_rgb"],
            misc["rendered_ctxt_depth"],
            misc["rendered_trgt_depth"],
        )

        rendered_ctxt_depth_mask = misc.get("rendered_ctxt_depth_mask")
        rendered_trgt_depth_mask = misc.get("rendered_trgt_depth_mask")

        ctxt_semantic = misc.get("ctxt_semantic")
        trgt_semantic = misc.get("trgt_semantic")

        rendered_intm_rgb = misc.get("rendered_intm_rgb")
        rendered_intm_depth = misc.get("rendered_intm_depth")
        rendered_intm_depth_mask = misc.get("rendered_intm_depth_mask")
        intm_semantic = misc.get("intm_semantic")

        rgb_ctxt = self.normalize(rendered_ctxt_img)
        depth_ctxt = rendered_ctxt_depth

        frames = None
        depth_frames = None
        full_images = None
        full_depths = None
        
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = inp["trgt_rgb"][:, 0, ...]
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        target = target.view(model_out.shape)
        t = repeat(t, "b -> (b c)", c=num_target)

        loss = self.loss_fn(model_out, target, reduction="none")
        if self.use_depth_mask and depth_mask_gt is not None:
            loss = loss * depth_mask_gt.unsqueeze(2) # from logits to 0-1
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        depth_loss = None
        if self.use_depth_supervision:
            depth_loss = self.loss_fn(depth_gt, depth, reduction="none")
            if self.use_depth_mask and depth_mask_gt is not None:
                depth_loss = depth_loss * depth_mask_gt
            depth_loss = reduce(depth_loss, "b ... -> b (...)", "mean")
            depth_loss = depth_loss * extract(self.p2_loss_weight, t, depth_loss.shape)
        
        depth_mask_loss = None
        if self.use_depth_mask:
            depth_mask_loss = self.loss_fn_mask(rendered_trgt_depth_mask.squeeze(2), depth_mask_gt, reduction="none")
            depth_mask_loss = reduce(depth_mask_loss, "b ... -> b (...)", "mean")
            depth_mask_loss = depth_mask_loss * extract(self.p2_loss_weight, t, depth_mask_loss.shape)

        semantic_loss = None
        semantic_loss_reg = None
        if self.use_semantic:
            semantic_gt = trgt_semantic["clip_embeddings"]
            rendered_trgt_semantic = trgt_semantic["center_features"]
            semantic_gt = semantic_gt.to(rendered_trgt_semantic.dtype)
            semantic_loss = self.loss_fn(semantic_gt, rendered_trgt_semantic, reduction="none")
            semantic_loss = reduce(semantic_loss, "b ... -> b (...)", "mean")
            semantic_loss = semantic_loss * extract(self.p2_loss_weight, t, semantic_loss.shape)
            if self.use_reg_model:
                semantic_reg_gt = trgt_semantic["dino_embeddings"]
                rendered_trgt_semantic_reg = trgt_semantic["center_features_reg"]
                semantic_reg_gt = semantic_reg_gt.to(rendered_trgt_semantic_reg.dtype)
                semantic_loss_reg = self.loss_fn(semantic_reg_gt, rendered_trgt_semantic_reg, reduction="none")
                semantic_loss_reg = reduce(semantic_loss_reg, "b ... -> b (...)", "mean")
                semantic_loss_reg = semantic_loss_reg * extract(self.p2_loss_weight, t, semantic_loss_reg.shape)

        lpips_loss = torch.zeros(1, device=model_out.device)
        depth_smooth_loss = torch.zeros(1, device=model_out.device)

        b, v, c, h, w = model_out.shape
        model_out = model_out.reshape(b * v, c, h, w)
        target = target.reshape(b * v, c, h, w)
        lpips_loss = self.perceptual_loss(model_out, target)

        depth_smooth_loss = self.edge_aware_smoothness(target, depth.squeeze(1))
        
        ## Losses for intermediate frames
        loss_intermediate = None
        if rendered_intm_rgb is not None:
            intermediate_gt = inp["intm_rgb"]
            b, v, c, h, w = intermediate_gt.shape
            _, num_intm, _, _, _ = rendered_intm_rgb.shape
            assert num_intm==v
            intermediate_gt = intermediate_gt.reshape(b * v, c, h, w)
            rendered_intm_rgb = rendered_intm_rgb.reshape(b * v, c, h, w)
            loss_intermediate = self.loss_fn(
            self.normalize(rendered_intm_rgb), intermediate_gt, reduction="none"
            )
            if self.use_depth_mask and rendered_intm_depth_mask is not None:
                intermediate_depth_mask_gt = inp["intm_depth_mask"]
                intermediate_depth_mask_gt_rgb = rearrange(intermediate_depth_mask_gt, "b v c h w -> (b v) c h w", b=b, v=v)
                loss_intermediate = loss_intermediate * intermediate_depth_mask_gt_rgb
            loss_intermediate = reduce(loss_intermediate, "b ... -> b (...)", "mean")
            loss_intermediate = loss_intermediate.view(b, v, -1).mean(dim=1)
            loss_intermediate = loss_intermediate * extract(
                self.p2_loss_weight, t, loss_intermediate.shape
            )

        loss_intermediate_depth = None
        if self.use_depth_supervision and rendered_intm_depth is not None:
            intermediate_depth_gt = inp["intm_depth"]
            b, v, c, h, w = intermediate_depth_gt.shape
            _, num_intm, _, _ = rendered_intm_depth.shape
            assert num_intm==v
            intermediate_depth_gt = intermediate_depth_gt.reshape(b * v, h, w)
            rendered_intm_depth = rendered_intm_depth.reshape(b * v, h, w)
            loss_intermediate_depth = self.loss_fn(
            rendered_intm_depth, intermediate_depth_gt, reduction="none"
            )
            if self.use_depth_mask:
                intermediate_depth_mask_gt = inp["intm_depth_mask"]
                intermediate_depth_mask_gt_depth = rearrange(intermediate_depth_mask_gt, "b v c h w -> (b v) c h w", b=b, v=v).squeeze(1)
                loss_intermediate_depth = loss_intermediate_depth * intermediate_depth_mask_gt_depth
            loss_intermediate_depth = reduce(loss_intermediate_depth, "b ... -> b (...)", "mean")
            loss_intermediate_depth = loss_intermediate_depth.view(b, v, -1).mean(dim=1) 
            loss_intermediate_depth = loss_intermediate_depth * extract(
                self.p2_loss_weight, t, loss_intermediate_depth.shape
            )
        
        loss_intermediate_depth_mask = None
        if self.use_depth_mask and rendered_intm_depth_mask is not None:
            intermediate_depth_mask_gt = inp["intm_depth_mask"]
            b, v, c, h, w = intermediate_depth_mask_gt.shape
            _, num_intm, _, _, _ = rendered_intm_depth_mask.shape
            assert num_intm==v
            intermediate_depth_mask_gt = intermediate_depth_mask_gt.reshape(b * v, c, h, w)
            rendered_intm_depth_mask = rendered_intm_depth_mask.reshape(b * v, c, h, w)
            loss_intermediate_depth_mask = self.loss_fn_mask(
                rendered_intm_depth_mask, intermediate_depth_mask_gt, reduction="none"
            )
            loss_intermediate_depth_mask = reduce(loss_intermediate_depth_mask, "b ... -> b (...)", "mean")
            loss_intermediate_depth_mask = loss_intermediate_depth_mask.view(b, v, -1).mean(dim=1) 
            loss_intermediate_depth_mask = loss_intermediate_depth_mask * extract(
                self.p2_loss_weight, t, loss_intermediate_depth_mask.shape
            )

        loss_intermediate_semantic = None
        loss_intermediate_semantic_reg = None
        if self.use_semantic and intm_semantic is not None:
            intermediate_semantic_gt = intm_semantic["clip_embeddings"]
            rendered_intm_semantic = intm_semantic["center_features"]
            intermediate_semantic_gt = intermediate_semantic_gt.to(rendered_intm_semantic.dtype)
            loss_intermediate_semantic = self.loss_fn(
            rendered_intm_semantic, intermediate_semantic_gt, reduction="none"
            )
            loss_intermediate_semantic = reduce(loss_intermediate_semantic, "b ... -> b (...)", "mean")
            loss_intermediate_semantic = loss_intermediate_semantic * extract(
                self.p2_loss_weight, t, loss_intermediate_semantic.shape
            )
            if self.use_reg_model:
                intermediate_semantic_reg_gt = intm_semantic["dino_embeddings"]
                rendered_intm_semantic_reg = intm_semantic["center_features_reg"]
                intermediate_semantic_reg_gt = intermediate_semantic_reg_gt.to(rendered_intm_semantic_reg.dtype)
                loss_intermediate_semantic_reg = self.loss_fn(
                rendered_intm_semantic_reg, intermediate_semantic_reg_gt, reduction="none"
                )
                loss_intermediate_semantic_reg = reduce(loss_intermediate_semantic_reg, "b ... -> b (...)", "mean")
                loss_intermediate_semantic_reg = loss_intermediate_semantic_reg * extract(
                    self.p2_loss_weight, t, loss_intermediate_semantic_reg.shape
                )

        ## Losses for the ctxt image
        if self.objective == "pred_x0":
            b, v, c, h, w = rgb_ctxt.shape
            t_zero = torch.zeros((b,), device=rgb_ctxt.device).long()
            context = inp["ctxt_rgb"][:, 0, ...]
            context = context.view(rgb_ctxt.shape)
            if self.use_depth_supervision:
                depth_gt_ctxt = inp["ctxt_depth"][:, 0, ...]
            if self.use_depth_mask:
                depth_mask_gt_ctxt = inp["ctxt_depth_mask"][:, 0, ...]

            t_zero = repeat(t_zero, "b -> (b c)", c=num_context)

            loss_ctxt = self.loss_fn(rgb_ctxt, context, reduction="none")
            if self.use_depth_mask and depth_mask_gt_ctxt is not None:
                loss_ctxt = loss_ctxt * depth_mask_gt_ctxt.unsqueeze(2)
            loss_ctxt = reduce(loss_ctxt, "b ... -> b (...)", "mean")
            loss_ctxt = loss_ctxt * extract(self.p2_loss_weight, t_zero, loss_ctxt.shape)

            depth_loss_ctxt = None
            if self.use_depth_supervision:
                depth_loss_ctxt = self.loss_fn(depth_gt_ctxt, depth_ctxt, reduction="none")
                if self.use_depth_mask and depth_mask_gt_ctxt is not None:
                    depth_loss_ctxt = depth_loss_ctxt * depth_mask_gt_ctxt
                depth_loss_ctxt = reduce(depth_loss_ctxt, "b ... -> b (...)", "mean")
                depth_loss_ctxt = depth_loss_ctxt * extract(self.p2_loss_weight, t_zero, depth_loss_ctxt.shape)
            
            depth_mask_loss_ctxt = None
            if self.use_depth_mask and rendered_ctxt_depth_mask is not None:
                depth_mask_loss_ctxt = self.loss_fn_mask(
                    rendered_ctxt_depth_mask.squeeze(2), depth_mask_gt_ctxt, reduction="none"
                )
                depth_mask_loss_ctxt = reduce(depth_mask_loss_ctxt, "b ... -> b (...)", "mean")
                depth_mask_loss_ctxt = depth_mask_loss_ctxt * extract(self.p2_loss_weight, t_zero, depth_mask_loss_ctxt.shape)
            
            semantic_loss_ctxt = None
            semantic_loss_ctxt_reg = None
            if self.use_semantic:
                semantic_gt_ctxt = ctxt_semantic["clip_embeddings"]
                rendered_ctxt_semantic = ctxt_semantic["center_features"]
                semantic_gt_ctxt = semantic_gt_ctxt.to(rendered_ctxt_semantic.dtype)
                semantic_loss_ctxt = self.loss_fn(semantic_gt_ctxt, rendered_ctxt_semantic, reduction="none")
                semantic_loss_ctxt = reduce(semantic_loss_ctxt, "b ... -> b (...)", "mean")
                semantic_loss_ctxt = semantic_loss_ctxt * extract(self.p2_loss_weight, t_zero, semantic_loss_ctxt.shape)
                if self.use_reg_model:
                    semantic_reg_gt_ctxt = ctxt_semantic["dino_embeddings"]
                    rendered_ctxt_semantic_reg = ctxt_semantic["center_features_reg"]
                    semantic_reg_gt_ctxt = semantic_reg_gt_ctxt.to(rendered_ctxt_semantic_reg.dtype)
                    semantic_loss_ctxt_reg = self.loss_fn(rendered_ctxt_semantic_reg, semantic_reg_gt_ctxt, reduction="none")
                    semantic_loss_ctxt_reg = reduce(semantic_loss_ctxt_reg, "b ... -> b (...)", "mean")
                    semantic_loss_ctxt_reg = semantic_loss_ctxt_reg * extract(self.p2_loss_weight, t_zero, semantic_loss_ctxt_reg.shape)

            lpips_loss_ctxt = torch.zeros(1, device=rgb_ctxt.device)
            depth_smooth_loss_ctxt = torch.zeros(1, device=rgb_ctxt.device)

            rgb_ctxt = rgb_ctxt.reshape(b * v, c, h, w)
            context = context.reshape(b * v, c, h, w)
            lpips_loss_ctxt = self.perceptual_loss(rgb_ctxt, context)
            depth_smooth_loss_ctxt = self.edge_aware_smoothness(context, depth_ctxt.squeeze(1))

        losses = {
            "rgb_loss": loss.mean(),
            "depth_loss": depth_loss.mean() if depth_loss is not None else None,
            "depth_mask_loss": depth_mask_loss.mean() if depth_mask_loss is not None else None,
            "depth_smooth_loss": depth_smooth_loss.mean(),
            "semantic_loss": semantic_loss.mean() if semantic_loss is not None else None,
            "semantic_loss_reg": semantic_loss_reg.mean() if semantic_loss_reg is not None else None,
            "lpips_loss": lpips_loss.mean(),
            "rgb_intermediate_loss": loss_intermediate.mean() if loss_intermediate is not None else None,
            "depth_intermediate_loss": loss_intermediate_depth.mean() if loss_intermediate_depth is not None else None,
            "depth_mask_intermediate_loss": loss_intermediate_depth_mask.mean() if loss_intermediate_depth_mask is not None else None,
            "semantic_intermediate_loss": loss_intermediate_semantic.mean() if loss_intermediate_semantic is not None else None,
            "semantic_intermediate_loss_reg": loss_intermediate_semantic_reg.mean() if loss_intermediate_semantic_reg is not None else None,
            "rgb_loss_ctxt": loss_ctxt.mean(), 
            "depth_loss_ctxt": depth_loss_ctxt.mean() if depth_loss_ctxt is not None else None,
            "depth_mask_loss_ctxt": depth_mask_loss_ctxt.mean() if depth_mask_loss_ctxt is not None else None,
            "semantic_loss_ctxt": semantic_loss_ctxt.mean() if semantic_loss_ctxt is not None else None,
            "semantic_loss_ctxt_reg": semantic_loss_ctxt_reg.mean() if semantic_loss_ctxt_reg is not None else None,
            "lpips_loss_ctxt": lpips_loss_ctxt.mean(), 
            "depth_smooth_loss_ctxt": depth_smooth_loss_ctxt.mean(), 
        }

        rendered_ctxt_img = (
            None if rendered_ctxt_img is None else rendered_ctxt_img
        )
        rendered_trgt_img = (
            None
            if misc["rendered_trgt_rgb"] is None
            else misc["rendered_trgt_rgb"]
        )
        return (
            losses,
            (
                self.unnormalize(x),
                self.unnormalize(inp["trgt_rgb"]),
                self.unnormalize(inp["ctxt_rgb"]),
                t,
                full_depths,
                full_images,
                frames,
                rendered_ctxt_img,
                rendered_ctxt_depth,
                rendered_trgt_img,
                rendered_trgt_depth,
                depth_frames,
            ),
        )

    def forward(self, inp, state_t, is_t_zero=False, *args, **kwargs):
        img = inp["trgt_rgb"][:, 0, ...]
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"
        if is_t_zero:
            t = torch.zeros((b,), device=device).long()
        else:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(inp, t, state_t=state_t, *args, **kwargs)

# trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        accelerator,
        dataloader=None,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        sample_every=1000,
        wandb_every=100,
        save_every=1000,
        num_samples=25,
        results_folder="./outputs",
        amp=False,
        fp16=False,
        split_batches=True,
        warmup_period=0,
        checkpoint_path=None,
        wandb_config=None,
        run_name="diffusion",
        depth_loss_weight=0.0,
        depth_smooth_loss_weight=0.0,
        semantic_loss_weight=0.0,
        semantic_reg_loss_weight=0.0,
        lpips_loss_weight=0.0,
        cfg=None,
        num_context=1,
        load_enc=False,
        lock_enc_steps=0,
        use_identity=False,
        load_optimizer=False,
        use_depth_smoothness=False,
        use_lpips_loss=True,
        use_depth_supervision=True,
        use_semantic=False,
        intermediate=True,
        num_intermediate=10,
    ):
        super().__init__()

        self.accelerator = accelerator
        if self.accelerator is None:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            self.accelerator = Accelerator(
                split_batches=True, mixed_precision="no", kwargs_handlers=[ddp_kwargs],
            )

        # self.accelerator.native_amp = amp
        self.num_context = num_context
        self.model = diffusion_model
        self.load_optimizer = load_optimizer
        self.use_depth_smoothness = use_depth_smoothness
        self.use_lpips_loss = use_lpips_loss
        self.use_depth_supervision = use_depth_supervision
        self.use_semantic = use_semantic
        self.use_intermediate = intermediate
        self.num_intermediate = num_intermediate

        self.num_samples = num_samples
        self.sample_every = sample_every
        self.save_every = save_every
        self.wandb_every = wandb_every
        self.depth_loss_weight = depth_loss_weight
        self.depth_smooth_loss_weight = depth_smooth_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        self.semantic_reg_loss_weight = semantic_reg_loss_weight
        self.lpips_loss_weight = lpips_loss_weight
        assert self.sample_every % self.wandb_every == 0

        self.adjacent_angle = cfg.adjacent_angle
        self.adjacent_distance = cfg.adjacent_distance

        self.ctxt_min = cfg.ctxt_min
        self.ctxt_max = cfg.ctxt_max

        self.batch_size = train_batch_size
        print(f"batch size: {self.batch_size}")
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = self.model.image_size

        EXCLUDE_KEYS = ("perceptual", "clip_model", "dino_model", "semantic_mapper")

        # optimizer
        params = [
            p for n, p in diffusion_model.named_parameters()
            if all(k not in n for k in EXCLUDE_KEYS)
        ]

        # self.opt = Adam(params, lr=train_lr, betas=adam_betas)
        self.opt = AdamW(params, lr=train_lr, weight_decay=1e-3, amsgrad=True,)
        lr_scheduler = get_cosine_schedule_with_warmup(
            self.opt, warmup_period, train_num_steps
        )
        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model, beta=ema_decay, update_every=ema_update_every
            )

        self.step = 0

        self.load_enc = load_enc
        self.lock_enc_steps = lock_enc_steps

        self.use_identity = use_identity

        # dataset and dataloader
        self.dataloader = self.accelerator.prepare(dataloader)
        self.dl = cycle(self.dataloader)
        self.model, self.opt, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.opt, lr_scheduler
        )

        # load model and opt
        if checkpoint_path is not None and self.accelerator.is_main_process:
            print(f"checkpoint path: {checkpoint_path}")
            
            if self.load_enc:
                self.load_enc_weights(checkpoint_path)
            if not self.load_enc:
                self.load(checkpoint_path)

        if self.accelerator.is_main_process:
            os.makedirs(results_folder, exist_ok=True)
        self.results_folder = Path(results_folder)

        if self.accelerator.is_main_process:
            if cfg.wandb_id is not None:
                wandb.init(
                    config=OmegaConf.to_container(cfg, resolve=True), **wandb_config, id=cfg.wandb_id, resume="allow"
                )
            elif cfg.resume_id is not None:
                wandb.init(
                    config=OmegaConf.to_container(cfg, resolve=True), **wandb_config, id=cfg.resume_id, resume="must"
                )
            else:
                wandb.init(config=OmegaConf.to_container(cfg, resolve=True), **wandb_config)
            wandb.run.name = run_name

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
        }
        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))
        # delete prev checkpoint if exists
        prev_milestone = milestone - 1
        prev_path = self.results_folder / f"model-{prev_milestone}.pt"
        if os.path.exists(prev_path):
            # delete prev checkpoint
            os.remove(prev_path)

    def load(self, path):
        data = torch.load(str(path), map_location=torch.device("cpu"),)
        model = self.accelerator.unwrap_model(self.model)
        # model = self.model
        # load all parameteres
        # data["model"].pop("model.encoder.backbone.pos_embed")
        orig = data["model"]
        filtered = {
            k: v
            for k, v in orig.items()
        #     if not k.startswith("model.semantic_mapper.") and not k.startswith("model.encoder.to_semantic.") and not k.startswith("model.encoder.to_semantic_reg.")
        # }
            if not k.startswith("model.semantic_mapper.")
        }
        print(model.load_state_dict(filtered, strict=False))
        
        if self.load_optimizer and "opt" in data:
            try:
                self.step = data["step"]
                self.opt.load_state_dict(data["opt"])
                print("loading step optimizer")
                for state in self.opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.accelerator.device, non_blocking=True)
            except:
                self.step = 0
                print("fail to load optimizer")

        if self.accelerator.is_main_process and "ema" in data:
            # data["ema"].pop("ema_model.model.encoder.backbone.pos_embed")
            # data["ema"].pop("online_model.model.encoder.backbone.pos_embed")
            ema_ckpt = data["ema"]
            for k in list(ema_ckpt.keys()):
                # if "semantic_mapper" in k or "to_semantic" in k or "to_semantic_reg" in k:
                if "semantic_mapper" in k:
                    ema_ckpt.pop(k)
            print(self.ema.load_state_dict(data["ema"], strict=False))

        if self.load_optimizer and "scaler" in data and exists(self.accelerator.scaler):
            self.accelerator.scaler.load_state_dict(data["scaler"])

        del data

    def load_enc_weights(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(path, map_location=device)

        encoder_weights = {k.replace('model.encoder.backbone.', ''): v
                           for k, v in data['model'].items()
                           if k.startswith('model.encoder.backbone.')}
        encoder_weights.pop("pos_embed")

        model_encoder_state_dict = self.model.model.encoder.backbone.state_dict()
        model_encoder_state_dict.update(encoder_weights)
        self.model.model.encoder.backbone.load_state_dict(model_encoder_state_dict)

        print(f"Encoder weights loaded successfully from {path}")

    # def lock_encoder(self):
    #     model = self.accelerator.unwrap_model(self.model)
    #     for param in model.model.encoder.backbone.parameters():
    #         param.requires_grad = False
    #     print("Encoder weights locked.")

    # def unlock_encoder(self):
    #     model = self.accelerator.unwrap_model(self.model)
    #     for param in model.model.encoder.backbone.parameters():
    #         param.requires_grad = True
    #     print("Encoder weights unlocked.")
    
    def lock_encoder(self):
        model = self.accelerator.unwrap_model(self.model)
        for name, param in model.model.named_parameters():
            if "to_semantic" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("Model weights locked (except to_semantic layers).")

    def unlock_encoder(self):
        model = self.accelerator.unwrap_model(self.model)
        for param in model.model.parameters():
            param.requires_grad = True
        print("Model weights unlocked.")

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        # torch.cuda.set_device(device)
        print(f"device: {device}")
        torch.cuda.empty_cache()
        log_path = os.path.join(str(self.results_folder), 'logs.json.txt')
        start_time = time.time()

        # lock encoder if needed
        if self.lock_enc_steps > 0:
            self.lock_encoder()
        
        if accelerator.is_main_process:  # or not accelerator.is_main_process:
            self.ema.to(device)
        
        accelerator.wait_for_everyone()

        def slice_to_length(batch, length):
            processed_batch = []
            for sample in batch:
                sample = list(sample)
                # Process each sample using the common random value.
                sample[0]["render_poses"] = sample[0]["render_poses"][:length]
                sample[0]["rgbs"] = sample[0]["rgbs"][:length]
                if "depth" in sample[0]:
                    sample[0]["depth"] = sample[0]["depth"][:length]
                if "semantic" in sample[0]:
                    sample[0]["semantic"] = sample[0]["semantic"][:length]
                if "intms" in sample[0]:
                    sample[0]["intms"] = sample[0]["intms"][:length]
                sample[0]["intrinsics"] = sample[0]["intrinsics"][:length]
                sample[1] = sample[1][:length]
                sample[2] = length

                processed_batch.append(sample)
            
            return default_collate(processed_batch)
        
        with JsonLogger(log_path) as json_logger:
            with tqdm(
                initial=self.step,
                total=self.train_num_steps,
                disable=not accelerator.is_main_process,
            ) as pbar:

                while self.step < self.train_num_steps:

                    # unlock encoder if reached the max lock step
                    if self.step == self.lock_enc_steps:
                        self.unlock_encoder()
                    
                    total_loss = 0.0
                    total_rgb_loss = 0.0
                    total_depth_loss = 0.0
                    total_depth_smooth_loss = 0.0
                    total_semantic_loss = 0.0
                    total_semantic_reg_loss = 0.0
                    total_lpips_loss = 0.0
                    total_rgb_intermediate_loss = 0.0
                    total_depth_intermediate_loss = 0.0
                    total_semantic_intermediate_loss = 0.0
                    total_semantic_intermediate_reg_loss = 0.0

                    total_rgb_loss_ctxt = 0.0
                    total_depth_loss_ctxt = 0.0
                    total_depth_smooth_loss_ctxt = 0.0
                    total_semantic_loss_ctxt = 0.0
                    total_semantic_loss_ctxt_reg = 0.0
                    total_lpips_loss_ctxt = 0.0

                    # move data to device
                    data_full = next(self.dl)
                    if accelerator.is_main_process:
                        seed = hash((self.step, self.train_num_steps)) % (2**32) + time.time_ns() % (2**32)
                        self.rng = np.random.default_rng(seed)
                        length_tensor = torch.tensor(
                            self.rng.integers(self.ctxt_min, self.ctxt_max),
                            device=accelerator.device,
                            dtype=torch.long,
                        )
                    else:
                        length_tensor = torch.zeros((), device=device, dtype=torch.long)
                    
                    length_tensor = broadcast(length_tensor)

                    accelerator.wait_for_everyone()

                    common_length = int(length_tensor.item())
                    data_full = slice_to_length(data_full, common_length)

                    assert isinstance(data_full, list)
                    data = data_full[0]
                    rgb_frames = data_full[1]
                    rgbs = [(frame[0, :].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8') for frame in rgb_frames]
                    seq_length = data_full[2]
                    
                    # load data
                    video_dict = data
                    data_rgbs = video_dict["rgbs"]
                    if self.use_depth_supervision:
                        data_depth = video_dict["depth"]

                    render_poses = video_dict["render_poses"]
                    intrinsics = video_dict["intrinsics"]
                    image_shape = video_dict["image_shape"]
                    lang = video_dict["lang"]
                    far = video_dict["far"]
                    near = video_dict["near"]
                    intms = video_dict["intms"]

                    video_length = len(render_poses)
                    assert video_length==seq_length[0]

                    ## compute key frame indices based on rotation or translation difference
                    key_frame_indices = list(range(1, len(render_poses)))

                    # reset state timestep
                    model = self.accelerator.unwrap_model(self.model)
                    model.model.reset_timestep()
                    if accelerator.is_main_process:
                        self.ema.ema_model.model.reset_timestep()

                    previous_t = 0
                    inp = {}
                    inp["ctxt_c2w"] = torch.cat(render_poses[:1], dim=0)
                    inp["ctxt_rgb"] = torch.cat(data_rgbs[:1], dim=0)
                    if self.use_depth_supervision:
                        inp["ctxt_depth"] = torch.cat(data_depth[:1], dim=0)

                    for state_t, update_t in enumerate(key_frame_indices):
                        accelerator.wait_for_everyone()
                        ## construct inp dict
                        inp["trgt_c2w"] = render_poses[update_t].clone()
                        inp["trgt_rgb"] = data_rgbs[update_t].clone()
                        if self.use_depth_supervision:
                            inp["trgt_depth"] = data_depth[update_t].clone()

                        if self.use_intermediate and update_t <= video_length - 1:
                            inp["intm_c2w"] = torch.cat(intms[update_t]["pose"], dim=1).clone()
                            inp["intm_rgb"] = torch.cat(intms[update_t]["rgbs"], dim=1).clone()
                            if self.use_depth_supervision:
                                inp["intm_depth"] = torch.cat(intms[update_t]["depth"], dim=1).clone()

                        inp["intrinsics"] = intrinsics[0].clone()
                        inp["image_shape"] = image_shape.clone()
                        inp["render_poses"] = torch.cat(render_poses, dim=1).clone()
                        inp["lang"] = lang
                        inp["num_frames_render"] = video_length
                        inp["near"] = near
                        inp["far"] = far

                        inp = to_gpu(inp, device)

                        with self.accelerator.autocast():
                            losses, misc = self.model(inp, state_t=state_t)

                            rgb_loss = losses["rgb_loss"]
                            rgb_loss = rgb_loss / self.gradient_accumulate_every
                            total_rgb_loss += rgb_loss.item()

                            depth_loss = losses.get("depth_loss", None)
                            if depth_loss is not None:
                                depth_loss = depth_loss / self.gradient_accumulate_every
                                total_depth_loss += depth_loss.item()

                            semantic_loss = losses.get("semantic_loss", None)
                            if semantic_loss is not None:
                                semantic_loss = semantic_loss / self.gradient_accumulate_every
                                total_semantic_loss += semantic_loss.item()
                            
                            semantic_loss_reg = losses.get("semantic_loss_reg", None)
                            if semantic_loss_reg is not None:
                                semantic_loss_reg = semantic_loss_reg / self.gradient_accumulate_every
                                total_semantic_reg_loss += semantic_loss_reg.item()

                            depth_smooth_loss = losses["depth_smooth_loss"]
                            depth_smooth_loss = (
                                depth_smooth_loss / self.gradient_accumulate_every
                            )

                            total_depth_smooth_loss += depth_smooth_loss.item()

                            rgb_intermediate_loss = losses.get("rgb_intermediate_loss")
                            if rgb_intermediate_loss is not None:
                                rgb_intermediate_loss = (
                                    rgb_intermediate_loss / self.gradient_accumulate_every
                                )
                                total_rgb_intermediate_loss += rgb_intermediate_loss.item()

                            depth_intermediate_loss = losses.get("depth_intermediate_loss")
                            if depth_intermediate_loss is not None:
                                depth_intermediate_loss = (
                                    depth_intermediate_loss / self.gradient_accumulate_every
                                )
                                total_depth_intermediate_loss += depth_intermediate_loss.item()

                            semantic_intermediate_loss = losses.get("semantic_intermediate_loss")
                            if semantic_intermediate_loss is not None:
                                semantic_intermediate_loss = (
                                    semantic_intermediate_loss / self.gradient_accumulate_every
                                )
                                total_semantic_intermediate_loss += semantic_intermediate_loss.item()
                            
                            semantic_intermediate_reg_loss = losses.get("semantic_intermediate_loss_reg")
                            if semantic_intermediate_reg_loss is not None:
                                semantic_intermediate_reg_loss = (
                                    semantic_intermediate_reg_loss / self.gradient_accumulate_every
                                )
                                total_semantic_intermediate_reg_loss += semantic_intermediate_reg_loss.item()

                            lpips_loss = losses["lpips_loss"]
                            lpips_loss = lpips_loss / self.gradient_accumulate_every
                            total_lpips_loss += lpips_loss.item()

                            # identity losses
                            if self.use_identity:
                                rgb_loss_ctxt = losses["rgb_loss_ctxt"]
                                rgb_loss_ctxt = rgb_loss_ctxt / self.gradient_accumulate_every
                                total_rgb_loss_ctxt += rgb_loss_ctxt.item()

                                depth_loss_ctxt = losses["depth_loss_ctxt"]
                                if depth_loss_ctxt is not None:
                                    depth_loss_ctxt = depth_loss_ctxt / self.gradient_accumulate_every
                                    total_depth_loss_ctxt += depth_loss_ctxt.item()

                                semantic_loss_ctxt = losses["semantic_loss_ctxt"]
                                if semantic_loss_ctxt is not None:
                                    semantic_loss_ctxt = semantic_loss_ctxt / self.gradient_accumulate_every
                                    total_semantic_loss_ctxt += semantic_loss_ctxt.item()
                                
                                semantic_loss_ctxt_reg = losses.get("semantic_loss_ctxt_reg", None)
                                if semantic_loss_ctxt_reg is not None:
                                    semantic_loss_ctxt_reg = (
                                        semantic_loss_ctxt_reg / self.gradient_accumulate_every
                                    )
                                    total_semantic_loss_ctxt_reg += semantic_loss_ctxt_reg.item()

                                depth_smooth_loss_ctxt = losses["depth_smooth_loss_ctxt"]
                                depth_smooth_loss_ctxt = (
                                    depth_smooth_loss_ctxt / self.gradient_accumulate_every
                                )
                                total_depth_smooth_loss_ctxt += depth_smooth_loss_ctxt.item()

                                lpips_loss_ctxt = losses["lpips_loss_ctxt"]
                                lpips_loss_ctxt = lpips_loss_ctxt / self.gradient_accumulate_every
                                total_lpips_loss_ctxt += lpips_loss_ctxt.item()


                            loss = rgb_loss

                            if self.use_lpips_loss:
                                loss += self.lpips_loss_weight * lpips_loss

                            if depth_loss is not None:
                                loss += self.depth_loss_weight * depth_loss

                            if semantic_loss is not None:
                                loss += self.semantic_loss_weight * semantic_loss
                            
                            if semantic_loss_reg is not None:
                                loss += self.semantic_reg_loss_weight * semantic_loss_reg

                            if self.use_depth_smoothness:
                                loss += self.depth_smooth_loss_weight * depth_smooth_loss

                            if self.use_identity:
                                loss = (
                                    loss
                                    + rgb_loss_ctxt
                                )
                                if self.use_lpips_loss:
                                    loss += self.lpips_loss_weight * lpips_loss_ctxt

                                if depth_loss_ctxt is not None:
                                    loss += self.depth_loss_weight * depth_loss_ctxt
                                
                                if semantic_loss_ctxt is not None:
                                    loss += self.semantic_loss_weight * semantic_loss_ctxt
                                
                                if semantic_loss_ctxt_reg is not None:
                                    loss += self.semantic_reg_loss_weight * semantic_loss_ctxt_reg

                                if self.use_depth_smoothness:
                                    loss += self.depth_smooth_loss_weight * depth_smooth_loss_ctxt
                            
                            if rgb_intermediate_loss is not None:
                                loss += rgb_intermediate_loss

                            if depth_intermediate_loss is not None:
                                loss += self.depth_loss_weight * depth_intermediate_loss
                            
                            if semantic_intermediate_loss is not None:
                                loss += self.semantic_loss_weight * semantic_intermediate_loss
                            
                            if semantic_intermediate_reg_loss is not None:
                                loss += self.semantic_reg_loss_weight * semantic_intermediate_reg_loss

                            total_loss += loss.item()

                            self.accelerator.backward(loss)  # TODO check if this is correct

                            accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                            pbar.set_description(f"loss: {total_loss:.4f}")

                            accelerator.wait_for_everyone()
                            self.opt.step()
                            self.opt.zero_grad()
                            accelerator.wait_for_everyone()

                            # accumulate losses every several steps
                            if (self.step!=0 and self.step % self.gradient_accumulate_every == 0):
                                # log the step
                                elapsed_time_sec = time.time() - start_time
                                if accelerator.is_main_process:
                                    step_log = {
                                        "step": self.step,
                                        "loss": total_loss,
                                        "rgb_loss": total_rgb_loss,
                                        "depth_loss": total_depth_loss * self.depth_loss_weight,
                                        "semantic_loss": total_semantic_loss * self.semantic_loss_weight,
                                        "semantic_loss_reg": total_semantic_reg_loss * self.semantic_reg_loss_weight,
                                        "rgb_intermediate_loss": total_rgb_intermediate_loss,
                                        "depth_intermediate_loss": total_depth_intermediate_loss * self.depth_loss_weight,
                                        "semantic_intermediate_loss": total_semantic_intermediate_loss * self.semantic_loss_weight,
                                        "semantic_intermediate_reg_loss": total_semantic_intermediate_reg_loss * self.semantic_reg_loss_weight,
                                        "lr": self.lr_scheduler.get_last_lr()[0],
                                        "num_context": inp["ctxt_rgb"].shape[1],
                                        "elapsed_time_sec": elapsed_time_sec,
                                    }
                                    if self.use_lpips_loss:
                                        step_log.update({
                                            "lpips_loss": total_lpips_loss * self.lpips_loss_weight,
                                        })
                                    if self.use_depth_smoothness:
                                        step_log.update({
                                            "depth_smooth_loss": total_depth_smooth_loss * self.depth_smooth_loss_weight,
                                        })
                                    if self.use_identity:
                                        step_log.update({
                                            "rgb_loss_ctxt": total_rgb_loss_ctxt,
                                            "depth_loss_ctxt": total_depth_loss_ctxt * self.depth_loss_weight,
                                            "semantic_loss_ctxt": total_semantic_loss_ctxt * self.semantic_loss_weight,
                                            "semantic_loss_ctxt_reg": total_semantic_loss_ctxt_reg * self.semantic_reg_loss_weight,
                                        })
                                        if self.use_lpips_loss:
                                            step_log.update({
                                                "lpips_loss_ctxt": total_lpips_loss_ctxt * self.lpips_loss_weight,
                                            })
                                        if self.use_depth_smoothness:
                                            step_log.update({
                                                "depth_smooth_loss_ctxt": total_depth_smooth_loss_ctxt * self.depth_smooth_loss_weight,
                                            })

                                    json_logger.log(step_log)
                                    wandb.log(step_log, step=self.step)

                                accelerator.wait_for_everyone()

                                # empty the losses
                                total_loss = 0.0
                                total_rgb_loss = 0.0
                                total_depth_loss = 0.0
                                total_semantic_loss = 0.0
                                total_semantic_reg_loss = 0.0
                                total_depth_smooth_loss = 0.0
                                total_lpips_loss = 0.0
                                total_rgb_intermediate_loss = 0.0
                                total_depth_intermediate_loss = 0.0
                                total_semantic_intermediate_loss = 0.0
                                total_semantic_intermediate_reg_loss = 0.0

                                total_rgb_loss_ctxt = 0.0
                                total_depth_loss_ctxt = 0.0
                                total_semantic_loss_ctxt = 0.0
                                total_semantic_loss_ctxt_reg = 0.0
                                total_depth_smooth_loss_ctxt = 0.0
                                total_lpips_loss_ctxt = 0.0

                        accelerator.wait_for_everyone()
                    
                        if accelerator.is_main_process:  # or not accelerator.is_main_process:
                            self.ema.update()
                            
                        # diffusion sample
                        consider_step = self.step + random.choice([-1, 0, 1]) # more steps can be sampled
                        if accelerator.is_main_process and (self.step != 0 and consider_step % self.sample_every == 0):
                            seed = random.randint(0, 99999)
                            model.model.copy_states_to_ema(self.ema.ema_model.model)
                            self.ema.ema_model.eval()

                            with torch.no_grad():
                                milestone = self.step // self.sample_every
                                inp_sample = {k: v[:1] for k, v in inp.items() if k in [
                                    "ctxt_c2w", "trgt_c2w", "intm_c2w", "ctxt_rgb", "trgt_rgb", "intm_rgb", 
                                    "ctxt_depth", "trgt_depth", "intm_depth",
                                    "ctxt_semantic", "trgt_semantic", "intm_semantic",
                                    "intrinsics", "image_shape",
                                    "render_poses", "lang", "near", "far"
                                ]}

                                inp_sample["num_frames_render"] = video_length

                                out = self.ema.ema_model.sample(batch_size=1, inp=inp_sample, state_t=state_t)
                                
                            frames, depth_frames, semantics = prepare_video_viz(out)
                            ## save samples
                            # create save folder for this step
                            save_folder_step = os.path.join(self.results_folder, f'milestone_{milestone}_{seed}')
                            os.makedirs(
                                save_folder_step, exist_ok=True,
                            )

                            save_folder_step_frames = os.path.join(self.results_folder, f'milestone_{milestone}_{seed}', f"rendered_frames_t_{previous_t}")
                            os.makedirs(
                                save_folder_step_frames, exist_ok=True,
                            )
                            # save video
                            denoised_f = os.path.join(save_folder_step, f"rendered_exploration_t_{previous_t}.mp4")
                            imageio.mimwrite(denoised_f, frames, fps=10, quality=10)
                            denoised_f_depth = os.path.join(save_folder_step, f"rendered_depth_t_{previous_t}.mp4")
                            imageio.mimwrite(denoised_f_depth, depth_frames, fps=10, quality=10)
                            denoised_f_semantic = os.path.join(save_folder_step, f"rendered_semantic_t_{previous_t}.mp4")
                            imageio.mimwrite(denoised_f_semantic, semantics, fps=10, quality=10)

                            # save all frames
                            for p, frame in enumerate(frames):
                                Image.fromarray(frame).save(
                                    os.path.join(save_folder_step_frames, f"rendered_{p}.png")
                                )

                            ## save GT video
                            GT_f = os.path.join(save_folder_step, f"GT_exploration.mp4")
                            imageio.mimwrite(GT_f, rgbs, fps=10, quality=10)
                            
                            # Log videos to wandb
                            if wandb.run is not None:
                                wandb.log({
                                    f"video/rendered_exploration_t_{previous_t}": wandb.Video(denoised_f, fps=10, format="mp4"),
                                    f"video/rendered_depth_t_{previous_t}": wandb.Video(denoised_f_depth, fps=10, format="mp4"),
                                    "video/GT_exploration": wandb.Video(GT_f, fps=10, format="mp4")
                                }, step=self.step)
                                if self.use_semantic:
                                    wandb.log({
                                        f"video/rendered_semantic_t_{previous_t}": wandb.Video(denoised_f_semantic, fps=10, format="mp4")
                                    }, step=self.step)
                            
                            save_folder_GT_frames = os.path.join(self.results_folder, f'milestone_{milestone}_{seed}', "GT_frames")
                            os.makedirs(
                                save_folder_GT_frames, exist_ok=True,
                            )

                            # save all frames
                            for p, frame in enumerate(rgbs):
                                Image.fromarray(frame).save(
                                    os.path.join(save_folder_GT_frames, f"GT_{p}.png")
                                )
                        
                        if accelerator.is_main_process and (self.step != 0 and self.step % self.sample_every == 0):                
                            ## save model
                            if (
                                self.step % self.wandb_every == 0
                            ):
                                if self.step != 0 and self.step % self.save_every == 0:
                                    milestone = self.step // self.save_every
                                    self.save(milestone)
                                    print(f"saved model at {milestone} milestones")
                        
                        accelerator.wait_for_everyone()

                        # update obs
                        previous_t = update_t
                        inp["ctxt_c2w"] = torch.cat(render_poses[previous_t:previous_t+1], dim=0)
                        inp["ctxt_rgb"] = torch.cat(data_rgbs[previous_t:previous_t+1], dim=0)
                        if self.use_depth_supervision:
                            inp["ctxt_depth"] = torch.cat(data_depth[previous_t:previous_t+1], dim=0)

                        self.step += 1

                        self.lr_scheduler.step()
                        accelerator.wait_for_everyone()
                        pbar.update(1)

        accelerator.print("training complete")
