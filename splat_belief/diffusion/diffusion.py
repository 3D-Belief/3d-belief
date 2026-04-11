# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch

import math
from pathlib import Path
import glob
from PIL import ImageDraw, ImageFont
from collections import OrderedDict
from omegaconf import OmegaConf
import numpy as np
from functools import partial
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam, AdamW
from torch.amp import autocast
from torchvision import utils

from einops import rearrange, reduce, repeat

from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from torchvision.utils import make_grid

import sys
import os
import time
import wandb
import imageio
from copy import deepcopy
from accelerate import DistributedDataParallelKwargs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from splat_belief.utils.vision_utils import *
from splat_belief.splat.layers import *
import lpips

from scipy import interpolate
import json as _json
from splat_belief.utils.vision_utils import JsonLogger
from splat_belief.splat.splat_belief import SplatBelief
from splat_belief.splat.alignment import VGGTAlignmentLoss
from splat_belief.splat.pose_estimator import VGGTPoseEstimator

# constants
ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "pred_x_start_high_res"]
)

def safe_normalize_tensor(in_feat, eps=1e-12):
    # ensure float32 accumulation + sanitize
    in_feat = torch.nan_to_num(in_feat, nan=0.0, posinf=1e4, neginf=-1e4)
    sq = torch.sum(in_feat.float() * in_feat.float(), dim=1, keepdim=True)
    norm = torch.sqrt(torch.clamp_min(sq, eps))
    return in_feat / norm

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

class Diffusion(nn.Module):
    def __init__(
        self,
        model: SplatBelief,
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
        predict_depth_mask=False,
        use_object_binary_mask=False,
        background_weight=0.1,
        use_semantic=False,
        use_reg_model=False,
        use_vggt_alignment=False,
        clip_semantic_loss_weight=0.0,
        cfg=None,
    ):
        super().__init__()
        assert not (type(self) == Diffusion and model.channels != model.out_dim)

        self.cfg = cfg
        self.model = model
        self.channels = self.model.channels
        self.temperature = temperature
        self.clip_semantic_loss_weight = clip_semantic_loss_weight

        self.image_size = image_size
        self.use_guidance = use_guidance
        self.guidance_scale = guidance_scale
        self.objective = objective

        self.clean_target = clean_target
        self.use_depth_supervision = use_depth_supervision
        self.use_semantic = use_semantic
        self.use_depth_mask = use_depth_mask
        self.predict_depth_mask = predict_depth_mask
        self.use_reg_model = use_reg_model
        self.use_object_binary_mask = use_object_binary_mask
        self.background_weight = background_weight
        self.use_vggt_alignment = use_vggt_alignment
        if self.use_vggt_alignment:
            print("using vggt alignment loss")
            self.build_alignment_module(cfg)

        # Pose estimation via VGGT CameraHead
        self.pose_source = getattr(cfg, "pose_source", "gt") if cfg is not None else "gt"
        self.pose_estimator = None
        if self.pose_source != "gt" and self.use_vggt_alignment:
            self.build_pose_estimator(cfg)

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
        lpips.normalize_tensor = safe_normalize_tensor

    def build_alignment_module(self, cfg):
        # load a naive vggt model 
        alignment_config = cfg.alignment 
        # alignment_config is DictConfig
        alignment_dict = OmegaConf.to_container(alignment_config, resolve=True)

        self.vggt_layer = None
        self.vggt_alignment_loss = VGGTAlignmentLoss(**alignment_dict)
        self.vggt_layer = alignment_dict["latents_info"]
        if self.vggt_layer is not None: 
            print(f"Using vggt layer {self.vggt_layer} for alignment loss")
        else:
            print(f"Using default vggt layer for alignment loss: [-3:]")

    def build_pose_estimator(self, cfg):
        """Build VGGTPoseEstimator reusing the VGGT model from alignment loss."""
        assert hasattr(self, "vggt_alignment_loss"), (
            "build_pose_estimator requires build_alignment_module to be called first "
            "(use_vggt_alignment must be True)"
        )
        pe_cfg = cfg.pose_estimator if hasattr(cfg, "pose_estimator") else {}
        freeze = pe_cfg.get("freeze_camera_head", True) if isinstance(pe_cfg, dict) else getattr(pe_cfg, "freeze_camera_head", True)
        num_iters = pe_cfg.get("num_refinement_iterations", 4) if isinstance(pe_cfg, dict) else getattr(pe_cfg, "num_refinement_iterations", 4)
        camera_head_ckpt = pe_cfg.get("camera_head_ckpt", None) if isinstance(pe_cfg, dict) else getattr(pe_cfg, "camera_head_ckpt", None)

        self.pose_estimator = VGGTPoseEstimator(
            vggt_model=self.vggt_alignment_loss.vggt_model,
            freeze_camera_head=freeze,
            num_refinement_iterations=num_iters,
        )
        # Optionally load finetuned CameraHead weights
        if camera_head_ckpt is not None:
            ckpt = torch.load(camera_head_ckpt, map_location="cpu", weights_only=True)
            self.pose_estimator.vggt_model.camera_head.load_state_dict(ckpt)
            print(f"Loaded finetuned CameraHead from {camera_head_ckpt}")
        print(f"Built VGGTPoseEstimator (freeze={freeze}, iters={num_iters}, pose_source={self.pose_source})")

    def alignment_loss(self, latents_list, context_images):
        """
        latents_list: list of latents, each is [b, 16, c, h, w]
        context_images: [b, 1, c, h, w]
        """
        if isinstance(self.vggt_layer, int):
            vggt_latents_list = [latents_list[self.vggt_layer]]
        elif isinstance(self.vggt_layer, str):
            vggt_latents_list = [latents_list[-1]]
        elif self.vggt_layer is None:
            vggt_latents_list = latents_list[-3:]
            
        return self.vggt_alignment_loss(vggt_latents_list, context_images)

    @torch.no_grad()
    def _replace_poses_with_predicted(self, inp):
        """Replace GT poses in inp dict with VGGT CameraHead predictions.

        Concatenates context + target (+ intermediate) images, runs pose estimation,
        then splits back and overwrites the pose keys in inp.
        """
        num_context = inp["ctxt_rgb"].shape[1]
        num_target = inp["trgt_rgb"].shape[1]
        has_intm = "intm_c2w" in inp

        # Collect all images: [B, S, C, H, W] in [0, 1]
        all_imgs = [self.unnormalize(inp["ctxt_rgb"]), self.unnormalize(inp["trgt_rgb"])]
        if has_intm:
            num_intm = inp["intm_rgb"].shape[1]
            all_imgs.append(self.unnormalize(inp["intm_rgb"]))
        all_images = torch.cat(all_imgs, dim=1)  # [B, S_total, C, H, W]

        # Predict poses for all views
        c2w, intrinsics_norm = self.pose_estimator.predict_from_images(
            all_images, normalize_to_first=True
        )

        # Split back
        idx = 0
        inp["ctxt_c2w"] = c2w[:, idx : idx + num_context]
        idx += num_context
        inp["trgt_c2w"] = c2w[:, idx : idx + num_target]
        idx += num_target
        if has_intm:
            inp["intm_c2w"] = c2w[:, idx : idx + num_intm]

        # Use mean predicted intrinsics (should be consistent across views)
        inp["intrinsics"] = intrinsics_norm[:, 0]  # [B, 3, 3]

        return inp

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
    ):
        x = inp["noisy_trgt_rgb"]
        _, _, _, h, w = inp["ctxt_rgb"].shape

        context_gaussians, target_gaussians = self.model.encoder(
            inp, 
            t=t,
            global_step=global_step, 
            deterministic=False
        )

        # Augmented gaussians
        gaussians = context_gaussians + target_gaussians
        
        output = self.model.decoder.forward(
            gaussians.float(),
            inp["trgt_c2w"],
            inp["intrinsics"].unsqueeze(1),
            inp["near"].float().unsqueeze(1),
            inp["far"].float().unsqueeze(1),
            (h, w),
            depth_mode=self.model.depth_mode,
        )

        model_output = self.normalize(output.color)[:, 0, ...]

        if self.use_guidance and self.guidance_scale > 1.0:
            inpt_uncond = deepcopy(inp)
            inpt_uncond["ctxt_rgb"] = inpt_uncond["ctxt_rgb"] * 0.0
            context_gaussians_uncond, target_gaussians_uncond = self.model.encoder(
                inpt_uncond, 
                t=t,
                global_step=global_step, 
                deterministic=False
            )

            # Augmented gaussians
            gaussians_uncond = context_gaussians_uncond + target_gaussians_uncond

            output_uncond = self.model.decoder.forward(
                gaussians_uncond.float(),
                inp["trgt_c2w"],
                inp["intrinsics"].unsqueeze(1),
                inp["near"].float().unsqueeze(1),
                inp["far"].float().unsqueeze(1),
                (h, w),
                depth_mode=self.model.depth_mode,
            )

            uncond_model_output = self.normalize(output_uncond.color)[:, 0, ...]

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

    def p_mean_variance(self, inp, t, clip_denoised=True):
        x = inp["noisy_trgt_rgb"]
        preds = self.model_predictions(inp, t)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, inp, t: int):
        x = inp["noisy_trgt_rgb"]
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            inp=inp, t=batched_times, clip_denoised=True
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False, inp=None, global_step=100000):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        print("p sample loop")

        x_start = None
        with torch.no_grad():
            # Predict Gaussians for the context
            ctxt_rgb = inp["ctxt_rgb"]
            context_input = {
                "image": ctxt_rgb,
                "extrinsics": inp["ctxt_c2w"],
                "intrinsics": inp["intrinsics"],
                "near": inp["near"],
                "far": inp["far"]
            }

            b, num_context, c, h, w = ctxt_rgb.shape
            ctxt_rgb = rearrange(ctxt_rgb, "b t h w c -> (b t) h w c")
            t = torch.zeros((b, num_context), device=ctxt_rgb.device, dtype=torch.long)

            t_zero = rearrange(t, "b t -> (b t)")

            context_gaussians = self.model.encoder(
                context_input, 
                t=t_zero,
                global_step=global_step, 
                abs_camera_poses=inp["ctxt_abs_camera_poses"],
                lang=inp["lang"],
                deterministic=False
            )

            inp["ctxt_gaussians"] = context_gaussians

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            if self.clean_target:
                inp["noisy_trgt_rgb"] = inp["trgt_rgb"][:, 0, :, :, :]
            else:
                inp["noisy_trgt_rgb"] = img

            img, x_start = self.p_sample(inp, t)
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
    def ddim_sample(self, shape, return_all_timesteps=False, inp=None, global_step=100000):
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
            )
            if time_next < 0:
                img = x_start
                imgs.append(img)

                # render the video
                frames, depth_frames, semantics, render_poses = self.model.render_video(
                    inp,
                    time_cond,
                    n=num_frames_render,
                    render_high_res=False,
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
            "depth_videos": depth_frames,
            "semantic_videos": semantics,
            "inp": inp,
            "render_poses": render_poses,
            "time_cond": time_cond,
        }
        return out_dict

    @torch.no_grad()
    def sample(self, batch_size=2, return_all_timesteps=False, inp=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        return sample_fn(
            (batch_size, channels, image_size, image_size),
            return_all_timesteps=return_all_timesteps,
            inp=inp,
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
        
    @property
    def loss_fn_mask(self):
        return F.binary_cross_entropy_with_logits

    @property
    def loss_fn_depth(self):
        return F.l1_loss
    # def loss_fn_depth(self):
    #     return F.mse_loss

    def loss_fn_repa(self, z, z_tilde):
        proj_loss = 0.
        bs = z.shape[0]
        for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
            z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
            z_j = torch.nn.functional.normalize(z_j, dim=-1) 
            proj_loss += self.mean_flat(1.0-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss = proj_loss / bs
        return proj_loss

    def mean_flat(self, x):
        return torch.mean(x, dim=list(range(1, len(x.size()))))

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

    # ------------------------------------------------------------------
    # CLIP Region-Matching Loss: rendered crops must match layout type
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _init_clip_image_encoder(self):
        """Lazily load frozen CLIP image encoder on first use."""
        if hasattr(self, "_clip_image_model"):
            return
        from transformers import CLIPModel, CLIPProcessor
        model_name = "openai/clip-vit-base-patch32"
        self._clip_processor = CLIPProcessor.from_pretrained(model_name)
        self._clip_image_model = CLIPModel.from_pretrained(model_name).eval()
        # Move to same device as diffusion model
        device = next(self.model.parameters()).device
        self._clip_image_model = self._clip_image_model.to(device)
        for p in self._clip_image_model.parameters():
            p.requires_grad_(False)

    def _compute_clip_semantic_loss(
        self,
        rendered_img: torch.Tensor,     # (B, C, H, W) in [-1, 1]
        layout_cls: torch.Tensor,       # (BT, H_layout, W_layout) long
        clip_text_embeddings: torch.Tensor,  # (N_types, D_clip)
        min_pixels: int = 16,
        max_regions: int = 8,
    ) -> torch.Tensor | None:
        """Compute CLIP region-matching loss between rendered crops and layout types.

        For each unique non-background class in the layout, crops the
        corresponding region from the rendered image, computes a CLIP image
        embedding, and compares it to the CLIP text embedding for that type.

        Returns scalar loss (1 - mean cosine similarity), or None if no
        valid regions are found.
        """
        self._init_clip_image_encoder()
        device = rendered_img.device

        # Handle 5D input (B, T, C, H, W) — squeeze the T dimension
        if rendered_img.ndim == 5:
            rendered_img = rendered_img[:, 0]  # (B, C, H, W)
        B, C, H, W = rendered_img.shape

        # Upsample layout_cls to rendered resolution
        # layout_cls: (BT, H_layout, W_layout) → (B, H, W)
        layout_up = F.interpolate(
            layout_cls[:B].unsqueeze(1).float(),
            size=(H, W), mode="nearest",
        ).squeeze(1).long()  # (B, H, W)

        # Unnormalize rendered image from [-1,1] to [0,1]
        img_01 = (rendered_img.clamp(-1, 1) + 1.0) * 0.5  # (B, C, H, W)

        cos_sims = []
        for b_idx in range(B):
            cls_map = layout_up[b_idx]  # (H, W)
            unique_cls = cls_map.unique()
            unique_cls = unique_cls[unique_cls > 0]  # skip background

            if len(unique_cls) == 0:
                continue

            # Limit to max_regions to control compute
            if len(unique_cls) > max_regions:
                # Pick the largest regions
                areas = [(cls_map == c).sum().item() for c in unique_cls]
                top_idx = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)[:max_regions]
                unique_cls = unique_cls[top_idx]

            for cls_id in unique_cls:
                cls_id_int = cls_id.item()
                if cls_id_int >= clip_text_embeddings.shape[0]:
                    continue

                mask = (cls_map == cls_id)  # (H, W)
                area = mask.sum().item()
                if area < min_pixels:
                    continue

                # Get bounding box of mask
                ys, xs = torch.where(mask)
                y_min, y_max = ys.min().item(), ys.max().item()
                x_min, x_max = xs.min().item(), xs.max().item()

                # Ensure minimum crop size (at least 8x8)
                if (y_max - y_min) < 8 or (x_max - x_min) < 8:
                    continue

                # Crop and resize to CLIP input size
                crop = img_01[b_idx, :, y_min:y_max+1, x_min:x_max+1]  # (C, h, w)
                crop_resized = F.interpolate(
                    crop.unsqueeze(0), size=(224, 224),
                    mode="bilinear", align_corners=False,
                )  # (1, C, 224, 224)

                # Normalize for CLIP (mean/std from CLIP processor)
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
                crop_norm = (crop_resized - mean) / std

                # Get CLIP image embedding
                image_features = self._clip_image_model.get_image_features(pixel_values=crop_norm)
                image_features = F.normalize(image_features, dim=-1)  # (1, D_clip)

                # Get text embedding for this type
                text_features = F.normalize(clip_text_embeddings[cls_id_int].unsqueeze(0), dim=-1)

                # Cosine similarity
                cos_sim = (image_features * text_features.to(device)).sum(dim=-1)
                cos_sims.append(cos_sim)

        if len(cos_sims) == 0:
            return None

        cos_sims = torch.cat(cos_sims)
        loss = 1.0 - cos_sims.mean()
        return loss

    def p_losses(self, inp, t, noise=None, global_step=100000):
        num_target = inp["trgt_rgb"].shape[1]
        num_context = inp["ctxt_rgb"].shape[1]
        x_start = inp["trgt_rgb"][:, 0, ...]
        noise = default(noise, lambda: torch.randn_like(x_start))

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
            if inp["trgt_depth_mask"].ndim == 5:
                depth_mask_gt = inp["trgt_depth_mask"][:, 0, ...]
            else:
                depth_mask_gt = inp["trgt_depth_mask"]

        if self.use_object_binary_mask:
            assert "ctxt_obj_binary_mask" in inp and "trgt_obj_binary_mask" in inp
            ctxt_obj_binary_mask = inp["ctxt_obj_binary_mask"][:, 0, ...]
            trgt_obj_binary_mask = inp["trgt_obj_binary_mask"][:, 0, ...]

        # Replace GT poses with predicted poses if configured
        if self.pose_estimator is not None and self.pose_source != "gt":
            inp = self._replace_poses_with_predicted(inp)

        model_out, depth, misc = self.model(inp, t, global_step=global_step)

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

        repa_pred = misc.get("repa_pred")
        repa_gt = misc.get("repa_gt")
        repa_pred_ctxt = misc.get("repa_pred_ctxt")
        repa_gt_ctxt = misc.get("repa_gt_ctxt")
        if repa_pred is not None and repa_gt is not None:
            repa_loss = self.loss_fn_repa(repa_pred.squeeze(1), repa_gt)
        else:
            repa_loss = None
        
        if repa_pred_ctxt is not None and repa_gt_ctxt is not None:
            repa_loss_ctxt = self.loss_fn_repa(repa_pred_ctxt.squeeze(1), repa_gt_ctxt)
        else:
            repa_loss_ctxt = None
        
        latents = misc.get("latents")
        if latents is not None and self.use_vggt_alignment:
            raw_ctxt = self.unnormalize(inp["ctxt_rgb"])
            raw_trgt = self.unnormalize(inp["trgt_rgb"])
            images = torch.cat([raw_ctxt, raw_trgt], dim=1) ## TODO: check
            alignment_loss = self.alignment_loss(latents, images)
        else:
            alignment_loss = None

        # ---- Layout reconstruction auxiliary loss ----
        layout_recon_loss = None
        layout_recon_cls_logits = misc.get("layout_recon_cls_logits")
        if layout_recon_cls_logits is not None:
            layout_cls_gt = misc["layout_cls_gt"]              # (BT, H, W) long
            layout_depth_gt = misc["layout_depth_gt"]          # (BT, H, W) float
            layout_depth_pred = misc["layout_recon_depth_pred"]  # (BT, 1, H, W)
            # Classification loss (cross-entropy, with background=0 downweighted)
            layout_cls_loss = F.cross_entropy(
                layout_recon_cls_logits, layout_cls_gt, reduction="mean"
            )
            # Depth regression loss (L1, only on non-background pixels)
            valid_depth_mask = (layout_cls_gt > 0).float()    # background has depth=0
            layout_depth_loss = (
                F.l1_loss(layout_depth_pred.squeeze(1), layout_depth_gt, reduction="none")
                * valid_depth_mask
            )
            if valid_depth_mask.sum() > 0:
                layout_depth_loss = layout_depth_loss.sum() / valid_depth_mask.sum()
            else:
                layout_depth_loss = layout_depth_loss.mean()
            layout_recon_loss = layout_cls_loss + layout_depth_loss

        # ---- CLIP region-matching semantic loss ----
        clip_semantic_loss = None
        if self.clip_semantic_loss_weight > 0 and rendered_trgt_img is not None:
            layout_cls_for_clip = misc.get("layout_cls_gt")
            clip_text_embs = misc.get("clip_type_embeddings")
            if layout_cls_for_clip is not None and clip_text_embs is not None:
                clip_semantic_loss = self._compute_clip_semantic_loss(
                    rendered_trgt_img, layout_cls_for_clip, clip_text_embs,
                )

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

        b, v, c, h, w = model_out.shape
        
        # Apply object boolean mask to loss
        if self.use_object_binary_mask:
            trgt_object_mask = inp["trgt_obj_binary_mask"][:, 0, ...]
            trgt_object_mask = trgt_object_mask.view(b, v, 1, h, w)
            # masked is 1, background is background_weight
            factor = trgt_object_mask.float() * 1.0 + (~trgt_object_mask).float() * self.background_weight
            target = target * factor
            model_out = model_out * factor

        loss = self.loss_fn(model_out, target, reduction="none")
        if self.use_depth_mask and depth_mask_gt is not None:
            loss = loss * depth_mask_gt.unsqueeze(2) # from logits to 0-1

        if self.use_depth_mask and depth_mask_gt is not None:
            valid = depth_mask_gt.unsqueeze(2).repeat(1, 1, 3, 1, 1).sum(dim=(2, 3, 4))
            loss = loss.sum(dim=(2, 3, 4)) / valid.clamp_min(1)
        else:
            loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        depth_loss = None
        if self.use_depth_supervision:
            # Apply object boolean mask to loss
            if self.use_object_binary_mask:
                trgt_object_mask = inp["trgt_obj_binary_mask"][:, 0, ...]
                # masked is 1, background is background_weight
                factor = trgt_object_mask.float() * 1.0 + (~trgt_object_mask).float() * self.background_weight
                depth_gt = depth_gt * factor
                depth = depth * factor

            depth_loss = self.loss_fn_depth(depth_gt, depth, reduction="none")
            if self.use_depth_mask and depth_mask_gt is not None:
                depth_loss = depth_loss * depth_mask_gt
            if self.use_depth_mask and depth_mask_gt is not None:
                valid = depth_mask_gt.sum(dim=(2, 3))
                depth_loss = depth_loss.sum(dim=(2, 3)) / valid.clamp_min(1)
            else:
                depth_loss = reduce(depth_loss, "b ... -> b (...)", "mean")
            depth_loss = depth_loss * extract(self.p2_loss_weight, t, depth_loss.shape)
        
        depth_mask_loss = None
        if self.use_depth_mask and self.predict_depth_mask:
            depth_mask_loss = self.loss_fn_mask(rendered_trgt_depth_mask.squeeze(2), depth_mask_gt, reduction="none")
            depth_mask_loss = reduce(depth_mask_loss, "b ... -> b (...)", "mean")
            depth_mask_loss = depth_mask_loss * extract(self.p2_loss_weight, t, depth_mask_loss.shape)

        semantic_loss = None
        semantic_loss_reg = None
        if self.use_semantic:
            semantic_gt = trgt_semantic["clip_embeddings"]
            rendered_trgt_semantic = trgt_semantic["center_features"]
            semantic_gt = semantic_gt.to(rendered_trgt_semantic.dtype)
            if self.use_object_binary_mask:
                trgt_object_mask = trgt_obj_binary_mask.unsqueeze(2)
                # masked is 1, background is background_weight
                factor = trgt_object_mask.float() * 1.0 + (~trgt_object_mask).float() * self.background_weight
                semantic_gt = semantic_gt * factor
                rendered_trgt_semantic = rendered_trgt_semantic * factor
            semantic_loss = self.loss_fn(semantic_gt, rendered_trgt_semantic, reduction="none")
            semantic_loss = reduce(semantic_loss, "b ... -> b (...)", "mean")
            semantic_loss = semantic_loss * extract(self.p2_loss_weight, t, semantic_loss.shape)
            if self.use_reg_model:
                semantic_reg_gt = trgt_semantic["dino_embeddings"]
                rendered_trgt_semantic_reg = trgt_semantic["center_features_reg"]
                semantic_reg_gt = semantic_reg_gt.to(rendered_trgt_semantic_reg.dtype)
                if self.use_object_binary_mask:
                    trgt_object_mask = trgt_obj_binary_mask.unsqueeze(2)
                    # masked is 1, background is background_weight
                    factor = trgt_object_mask.float() * 1.0 + (~trgt_object_mask).float() * self.background_weight
                    semantic_reg_gt = semantic_reg_gt * factor
                    rendered_trgt_semantic_reg = rendered_trgt_semantic_reg * factor
                semantic_loss_reg = self.loss_fn(semantic_reg_gt, rendered_trgt_semantic_reg, reduction="none")
                semantic_loss_reg = reduce(semantic_loss_reg, "b ... -> b (...)", "mean")
                semantic_loss_reg = semantic_loss_reg * extract(self.p2_loss_weight, t, semantic_loss_reg.shape)

        lpips_loss = torch.zeros(1, device=model_out.device)
        depth_smooth_loss = torch.zeros(1, device=model_out.device)

        model_out = model_out.reshape(b * v, c, h, w)
        target = target.reshape(b * v, c, h, w)

        model_out = torch.nan_to_num(model_out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1, 1).contiguous()
        target    = torch.nan_to_num(target,    nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1, 1).contiguous()

        with autocast(device_type="cuda", enabled=False):
            lpips_loss = self.perceptual_loss(model_out.float(), target.float())
            lpips_loss = lpips_loss.mean()
        if not torch.isfinite(lpips_loss):
            lpips_loss = torch.zeros((), device=model_out.device)

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
            rendered_intm_rgb = self.normalize(rendered_intm_rgb)
            if self.use_object_binary_mask:
                intm_obj_binary_mask = inp["intm_obj_binary_mask"].reshape(b * v, 1, h, w)
                factor = intm_obj_binary_mask.float() * 1.0 + (~intm_obj_binary_mask).float() * self.background_weight
                intermediate_gt = intermediate_gt * factor
                rendered_intm_rgb = rendered_intm_rgb * factor

            loss_intermediate = self.loss_fn(
            rendered_intm_rgb, intermediate_gt, reduction="none"
            )
            if self.use_depth_mask and rendered_intm_depth_mask is not None:
                if inp["intm_depth_mask"].ndim == 5:
                    intermediate_depth_mask_gt = inp["intm_depth_mask"]
                else:
                    intermediate_depth_mask_gt = inp["intm_depth_mask"].unsqueeze(2)
                intermediate_depth_mask_gt_rgb = rearrange(intermediate_depth_mask_gt, "b v c h w -> (b v) c h w", b=b, v=v)
                loss_intermediate = loss_intermediate * intermediate_depth_mask_gt_rgb
            if self.use_depth_mask and rendered_intm_depth_mask is not None:
                valid = intermediate_depth_mask_gt.sum(dim=(2, 3, 4))
                loss_intermediate = loss_intermediate.view(b, v, -1)                
                loss_intermediate = loss_intermediate.sum(dim=2) / valid.clamp_min(1)
            else:
                loss_intermediate = reduce(loss_intermediate, "b ... -> b (...)", "mean")
                loss_intermediate = loss_intermediate.view(b, v, -1).mean(dim=1)
            loss_intermediate = loss_intermediate * extract(
                self.p2_loss_weight, t, loss_intermediate.shape
            )

        loss_intermediate_depth = None
        if self.use_depth_supervision and rendered_intm_depth is not None:
            if inp["intm_depth"].ndim == 5:
                intermediate_depth_gt = inp["intm_depth"]
            else:
                intermediate_depth_gt = inp["intm_depth"].unsqueeze(2)
            b, v, c, h, w = intermediate_depth_gt.shape
            _, num_intm, _, _ = rendered_intm_depth.shape
            assert num_intm==v
            intermediate_depth_gt = intermediate_depth_gt.reshape(b * v, h, w)
            rendered_intm_depth = rendered_intm_depth.reshape(b * v, h, w)
            if self.use_object_binary_mask:
                intm_obj_binary_mask = inp["intm_obj_binary_mask"].reshape(b * v, 1, h, w)
                factor = intm_obj_binary_mask.float() * 1.0 + (~intm_obj_binary_mask).float() * self.background_weight
                intermediate_depth_gt = intermediate_depth_gt * factor.squeeze(1)
                rendered_intm_depth = rendered_intm_depth * factor.squeeze(1)
            loss_intermediate_depth = self.loss_fn_depth(
            rendered_intm_depth, intermediate_depth_gt, reduction="none"
            )
            if self.use_depth_mask:
                if inp["intm_depth_mask"].ndim == 5:
                    intermediate_depth_mask_gt = inp["intm_depth_mask"]
                else:
                    intermediate_depth_mask_gt = inp["intm_depth_mask"].unsqueeze(2)
                intermediate_depth_mask_gt_depth = rearrange(intermediate_depth_mask_gt, "b v c h w -> (b v) c h w", b=b, v=v).squeeze(1)
                loss_intermediate_depth = loss_intermediate_depth * intermediate_depth_mask_gt_depth
            if self.use_depth_mask:
                valid = intermediate_depth_mask_gt.sum(dim=(2, 3, 4))
                loss_intermediate_depth = loss_intermediate_depth.view(b, v, -1)
                loss_intermediate_depth = loss_intermediate_depth.sum(dim=2) / valid.clamp_min(1)
            else:
                loss_intermediate_depth = reduce(loss_intermediate_depth, "b ... -> b (...)", "mean")
                loss_intermediate_depth = loss_intermediate_depth.view(b, v, -1).mean(dim=1) 
            loss_intermediate_depth = loss_intermediate_depth * extract(
                self.p2_loss_weight, t, loss_intermediate_depth.shape
            )
        
        loss_intermediate_depth_mask = None
        if self.use_depth_mask and rendered_intm_depth_mask is not None and self.predict_depth_mask:
            if inp["intm_depth_mask"].ndim == 5:
                intermediate_depth_mask_gt = inp["intm_depth_mask"]
            else:
                intermediate_depth_mask_gt = inp["intm_depth_mask"].unsqueeze(2)
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
            if self.use_object_binary_mask:
                intm_obj_binary_mask = inp["intm_obj_binary_mask"].reshape(b * v, 1)
                factor = intm_obj_binary_mask.float() * 1.0 + (~intm_obj_binary_mask).float() * self.background_weight
                intermediate_semantic_gt = intermediate_semantic_gt * factor
                rendered_intm_semantic = rendered_intm_semantic * factor
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
                if self.use_object_binary_mask:
                    intm_obj_binary_mask = inp["intm_obj_binary_mask"].reshape(b * v, 1)
                    factor = intm_obj_binary_mask.float() * 1.0 + (~intm_obj_binary_mask).float() * self.background_weight
                    intermediate_semantic_reg_gt = intermediate_semantic_reg_gt * factor
                    rendered_intm_semantic_reg = rendered_intm_semantic_reg * factor
                loss_intermediate_semantic_reg = self.loss_fn(
                rendered_intm_semantic_reg, intermediate_semantic_reg_gt, reduction="none"
                )
                loss_intermediate_semantic_reg = reduce(loss_intermediate_semantic_reg, "b ... -> b (...)", "mean")
                loss_intermediate_semantic_reg = loss_intermediate_semantic_reg * extract(
                    self.p2_loss_weight, t, loss_intermediate_semantic_reg.shape
                )

        depth_smooth_intermediate_loss = None
        if rendered_intm_depth is not None and rendered_intm_rgb is not None:
            # Clamp/interpolate guidance like elsewhere to avoid NaNs/Inf
            intm_guidance_rgb = torch.nan_to_num(
                intermediate_gt, nan=0.0, posinf=1e4, neginf=-1e4
            ).clamp(-1, 1).contiguous()

            if rendered_intm_depth.ndim == 4:
                intm_depth = rearrange(rendered_intm_depth, "b v h w -> (b v) h w", b=b, v=v)
            else:
                intm_depth = rendered_intm_depth
            
            # Compute per-(b*v) smoothness maps; returns either (b*v, h, w) or scalar per sample
            # depending on your implementation of edge_aware_smoothness. We’ll handle both.
            smooth_map = self.edge_aware_smoothness(
                intm_guidance_rgb, intm_depth
            )

            # If depth masks are enabled, weight out invalid pixels
            if self.use_depth_mask:
                if inp["intm_depth_mask"].ndim == 5:
                    intermediate_depth_mask_gt = inp["intm_depth_mask"]
                else:
                    intermediate_depth_mask_gt = inp["intm_depth_mask"].unsqueeze(2)
                intermediate_depth_mask_gt_depth = rearrange(
                    intermediate_depth_mask_gt, "b v c h w -> (b v) c h w", b=b, v=v
                ).squeeze(1)  # (b*v, h, w)

                # If smooth_map is scalar per sample, turn it into a pixel map to mask and then average.
                if smooth_map.ndim == 1:  # (b*v,)
                    # Expand to per-pixel weights (uniform), mask, then average back
                    smooth_map = smooth_map.view(b * v, 1, 1).expand(-1, h, w)
                smooth_map = smooth_map * intermediate_depth_mask_gt_depth

                # Reduce to per-sample means after masking
                smooth_map = smooth_map.view(b * v, -1).mean(dim=1)  # (b*v,)

            else:
                # No mask case: if we have a pixel map, average to per-sample scalars
                if smooth_map.ndim == 3:  # (b*v, h, w)
                    smooth_map = smooth_map.view(b * v, -1).mean(dim=1)  # (b*v,)
                # else if already (b*v,) keep as is

            # Now (b*v,) -> (b, v) -> mean over v to match your other intermediate terms
            depth_smooth_intermediate_loss = smooth_map.view(b, v).mean(dim=1)  # (b,)

            # p2 weighting shaped to match
            depth_smooth_intermediate_loss = depth_smooth_intermediate_loss * extract(
                self.p2_loss_weight, t, depth_smooth_intermediate_loss.shape
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
                if inp["ctxt_depth_mask"].ndim == 5:
                    depth_mask_gt_ctxt = inp["ctxt_depth_mask"][:, 0, ...]
                else:
                    depth_mask_gt_ctxt = inp["ctxt_depth_mask"]

            t_zero = repeat(t_zero, "b -> (b c)", c=num_context)
            if self.use_object_binary_mask:
                ctxt_obj_binary_mask = inp["ctxt_obj_binary_mask"][:, 0, ...]
                ctxt_obj_binary_mask = ctxt_obj_binary_mask.view(b, v, 1, h, w)
                factor = ctxt_obj_binary_mask.float() * 1.0 + (~ctxt_obj_binary_mask).float() * self.background_weight
                rgb_ctxt = rgb_ctxt * factor
                context = context * factor

            loss_ctxt = self.loss_fn(rgb_ctxt, context, reduction="none")
            if self.use_depth_mask and depth_mask_gt_ctxt is not None:
                loss_ctxt = loss_ctxt * depth_mask_gt_ctxt.unsqueeze(2)
            if self.use_depth_mask and depth_mask_gt_ctxt is not None:
                valid = depth_mask_gt_ctxt.unsqueeze(2).repeat(1, 1, 3, 1, 1).sum(dim=(2, 3, 4))
                loss_ctxt = loss_ctxt.sum(dim=(2, 3, 4)) / valid.clamp_min(1)
            else:
                loss_ctxt = reduce(loss_ctxt, "b ... -> b (...)", "mean")
            loss_ctxt = loss_ctxt * extract(self.p2_loss_weight, t_zero, loss_ctxt.shape)

            depth_loss_ctxt = None
            if self.use_depth_supervision:
                if self.use_object_binary_mask:
                    depth_gt_ctxt = depth_gt_ctxt.view(b, v, h, w) * factor.squeeze(2)
                    rendered_ctxt_depth = rendered_ctxt_depth * factor.squeeze(2)
                depth_loss_ctxt = self.loss_fn_depth(depth_gt_ctxt, rendered_ctxt_depth, reduction="none")
                if self.use_depth_mask and depth_mask_gt_ctxt is not None:
                    depth_loss_ctxt = depth_loss_ctxt * depth_mask_gt_ctxt
                if self.use_depth_mask and depth_mask_gt_ctxt is not None:
                    valid = depth_mask_gt_ctxt.sum(dim=(2, 3))
                    depth_loss_ctxt = depth_loss_ctxt.sum(dim=(2, 3)) / valid.clamp_min(1)
                else:
                    depth_loss_ctxt = reduce(depth_loss_ctxt, "b ... -> b (...)", "mean")
                depth_loss_ctxt = depth_loss_ctxt * extract(self.p2_loss_weight, t_zero, depth_loss_ctxt.shape)
            
            depth_mask_loss_ctxt = None
            if self.use_depth_mask and rendered_ctxt_depth_mask is not None and self.predict_depth_mask:
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
                if self.use_object_binary_mask:
                    ctxt_obj_binary_mask = inp["ctxt_obj_binary_mask"][:, 0, ...]
                    ctxt_obj_binary_mask = ctxt_obj_binary_mask.view(b, v, 1, h, w)
                    factor = ctxt_obj_binary_mask.float() * 1.0 + (~ctxt_obj_binary_mask).float() * self.background_weight
                    semantic_gt_ctxt = semantic_gt_ctxt * factor
                    rendered_ctxt_semantic = rendered_ctxt_semantic * factor
                semantic_loss_ctxt = self.loss_fn(semantic_gt_ctxt, rendered_ctxt_semantic, reduction="none")
                semantic_loss_ctxt = reduce(semantic_loss_ctxt, "b ... -> b (...)", "mean")
                semantic_loss_ctxt = semantic_loss_ctxt * extract(self.p2_loss_weight, t_zero, semantic_loss_ctxt.shape)
                if self.use_reg_model:
                    semantic_reg_gt_ctxt = ctxt_semantic["dino_embeddings"]
                    rendered_ctxt_semantic_reg = ctxt_semantic["center_features_reg"]
                    semantic_reg_gt_ctxt = semantic_reg_gt_ctxt.to(rendered_ctxt_semantic_reg.dtype)
                    if self.use_object_binary_mask:
                        ctxt_obj_binary_mask = inp["ctxt_obj_binary_mask"][:, 0, ...]
                        ctxt_obj_binary_mask = ctxt_obj_binary_mask.view(b, v, 1, h, w)
                        factor = ctxt_obj_binary_mask.float() * 1.0 + (~ctxt_obj_binary_mask).float() * self.background_weight
                        semantic_reg_gt_ctxt = semantic_reg_gt_ctxt * factor
                        rendered_ctxt_semantic_reg = rendered_ctxt_semantic_reg * factor
                    semantic_loss_ctxt_reg = self.loss_fn(rendered_ctxt_semantic_reg, semantic_reg_gt_ctxt, reduction="none")
                    semantic_loss_ctxt_reg = reduce(semantic_loss_ctxt_reg, "b ... -> b (...)", "mean")
                    semantic_loss_ctxt_reg = semantic_loss_ctxt_reg * extract(self.p2_loss_weight, t_zero, semantic_loss_ctxt_reg.shape)

            lpips_loss_ctxt = torch.zeros(1, device=rgb_ctxt.device)
            depth_smooth_loss_ctxt = torch.zeros(1, device=rgb_ctxt.device)

            rgb_ctxt = rgb_ctxt.reshape(b * v, c, h, w)
            context = context.reshape(b * v, c, h, w)

            depth_smooth_loss_ctxt = self.edge_aware_smoothness(context, depth_ctxt.squeeze(1))
            rgb_ctxt = torch.nan_to_num(rgb_ctxt, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1, 1).contiguous()
            context    = torch.nan_to_num(context,    nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1, 1).contiguous()

            with autocast(device_type="cuda", enabled=False):
                lpips_loss_ctxt = self.perceptual_loss(rgb_ctxt.float(), context.float())
                lpips_loss_ctxt = lpips_loss_ctxt.mean()
            if not torch.isfinite(lpips_loss_ctxt):
                lpips_loss_ctxt = torch.zeros((), device=rgb_ctxt.device)

        losses = {
            "rgb_loss": loss.mean(),
            "depth_loss": depth_loss.mean() if depth_loss is not None else None,
            "depth_mask_loss": depth_mask_loss.mean() if depth_mask_loss is not None else None,
            "depth_smooth_loss": depth_smooth_loss.mean(),
            "semantic_loss": semantic_loss.mean() if semantic_loss is not None else None,
            "semantic_loss_reg": semantic_loss_reg.mean() if semantic_loss_reg is not None else None,
            "lpips_loss": lpips_loss,
            "rgb_intermediate_loss": loss_intermediate.mean() if loss_intermediate is not None else None,
            "depth_intermediate_loss": loss_intermediate_depth.mean() if loss_intermediate_depth is not None else None,
            "depth_mask_intermediate_loss": loss_intermediate_depth_mask.mean() if loss_intermediate_depth_mask is not None else None,
            "semantic_intermediate_loss": loss_intermediate_semantic.mean() if loss_intermediate_semantic is not None else None,
            "semantic_intermediate_loss_reg": loss_intermediate_semantic_reg.mean() if loss_intermediate_semantic_reg is not None else None,
            "depth_smooth_intermediate_loss": depth_smooth_intermediate_loss.mean() if depth_smooth_intermediate_loss is not None else None,
            "rgb_loss_ctxt": loss_ctxt.mean(), 
            "depth_loss_ctxt": depth_loss_ctxt.mean() if depth_loss_ctxt is not None else None,
            "depth_mask_loss_ctxt": depth_mask_loss_ctxt.mean() if depth_mask_loss_ctxt is not None else None,
            "semantic_loss_ctxt": semantic_loss_ctxt.mean() if semantic_loss_ctxt is not None else None,
            "semantic_loss_ctxt_reg": semantic_loss_ctxt_reg.mean() if semantic_loss_ctxt_reg is not None else None,
            "lpips_loss_ctxt": lpips_loss_ctxt, 
            "depth_smooth_loss_ctxt": depth_smooth_loss_ctxt.mean(), 
            "repa_loss": repa_loss.mean() if repa_loss is not None else None,
            "repa_loss_ctxt": repa_loss_ctxt.mean() if repa_loss_ctxt is not None else None,
            "alignment_loss": alignment_loss.mean() if alignment_loss is not None else None,
            "layout_recon_loss": layout_recon_loss if layout_recon_loss is not None else None,
            "clip_semantic_loss": clip_semantic_loss if clip_semantic_loss is not None else None,
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

    def forward(self, inp, *args, **kwargs):
        img = inp["trgt_rgb"][:, 0, ...]
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(inp, t, *args, **kwargs)

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
        depth_mask_loss_weight=0.0,
        depth_smooth_loss_weight=0.0,
        semantic_loss_weight=0.0,
        semantic_reg_loss_weight=0.0,
        intermediate_weight=0.0,
        lpips_loss_weight=0.0,
        repa_loss_weight=0.0,
        vggt_alignment_loss_weight=0.0,
        layout_recon_loss_weight=0.0,
        clip_semantic_loss_weight=0.0,
        ctxt_losses_factor=1.0,
        cfg=None,
        num_context=1,
        load_enc=False,
        lock_enc_steps=0,
        use_identity=False,
        load_optimizer=False,
        use_depth_smoothness=False,
        use_lpips_loss=True,
        use_semantic=False,
    ):
        super().__init__()

        self.accelerator = accelerator
        if self.accelerator is None:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            self.accelerator = Accelerator(
                split_batches=True, mixed_precision="no", kwargs_handlers=[ddp_kwargs],
            )

        self.num_context = num_context
        self.model = diffusion_model
        self.load_optimizer = load_optimizer
        self.use_depth_smoothness = use_depth_smoothness
        self.use_lpips_loss = use_lpips_loss
        self.use_semantic = use_semantic

        self.num_samples = num_samples
        self.sample_every = sample_every
        self.save_every = save_every
        self.wandb_every = wandb_every
        self.depth_loss_weight = depth_loss_weight
        self.depth_mask_loss_weight = depth_mask_loss_weight
        self.depth_smooth_loss_weight = depth_smooth_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        self.semantic_reg_loss_weight = semantic_reg_loss_weight
        self.lpips_loss_weight = lpips_loss_weight
        self.repa_loss_weight = repa_loss_weight
        self.intermediate_weight = intermediate_weight
        self.vggt_alignment_loss_weight = vggt_alignment_loss_weight
        self.layout_recon_loss_weight = layout_recon_loss_weight
        self.clip_semantic_loss_weight = clip_semantic_loss_weight
        self.ctxt_losses_factor = ctxt_losses_factor
        assert self.sample_every % self.wandb_every == 0

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
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
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
        # data["model"].pop("model.encoder.backbone.external_cond_embedding.patch_embedder.proj.weight")
        # data["model"].pop("model.encoder.backbone.embed_input.proj.weight")
        # data["model"].pop("model.encoder.backbone.project_output.proj.weight")
        print(model.load_state_dict(data["model"], strict=False))
        
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
            # data["ema"].pop("ema_model.model.encoder.backbone.external_cond_embedding.patch_embedder.proj.weight")
            # data["ema"].pop("ema_model.model.encoder.backbone.embed_input.proj.weight")
            # data["ema"].pop("ema_model.model.encoder.backbone.project_output.proj.weight")
            # data["ema"].pop("online_model.model.encoder.backbone.external_cond_embedding.patch_embedder.proj.weight")
            # data["ema"].pop("online_model.model.encoder.backbone.embed_input.proj.weight")
            # data["ema"].pop("online_model.model.encoder.backbone.project_output.proj.weight")
            print(self.ema.load_state_dict(data["ema"], strict=False))

        if self.load_optimizer and "scaler" in data and exists(self.accelerator.scaler):
            self.accelerator.scaler.load_state_dict(data["scaler"])
        
        if self.load_optimizer and "lr_scheduler" in data and data["lr_scheduler"] is not None:
            self.lr_scheduler.load_state_dict(data["lr_scheduler"])
            print("loaded lr scheduler")

        del data

    def load_enc_weights(self, path: str):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(path, map_location=device, weights_only=False)

        if path.endswith('.ckpt'):
            encoder_weights = {k.replace('diffusion_model.model.', ''): v
                            for k, v in data['state_dict'].items()
                            if (k.startswith('diffusion_model.model.'))}
            encoder_weights.pop("project_output.proj.weight")
            encoder_weights.pop("project_output.proj.bias")

            # # for different patch sizes
            # encoder_weights.pop("embed_input.proj.weight")
            # encoder_weights.pop("embed_input.proj.bias")
            # encoder_weights.pop("external_cond_embedding.patch_embedder.proj.weight")
            # encoder_weights.pop("external_cond_embedding.patch_embedder.proj.bias")
        elif path.endswith('.pt'):
            encoder_weights = {k.replace('model.encoder.backbone.', ''): v
                            for k, v in data['model'].items()
                            if (k.startswith('model.encoder.backbone.'))}
        
        model_ref = self.model
        if accelerator is not None and hasattr(accelerator, "unwrap_model"):
            model_ref = accelerator.unwrap_model(model_ref)
        elif hasattr(model_ref, "module"):  # vanilla torch.nn.parallel.DistributedDataParallel
            model_ref = model_ref.module

        model_encoder_state_dict = model_ref.model.encoder.backbone.state_dict()
        model_encoder_state_dict.update(encoder_weights)
        model_ref.model.encoder.backbone.load_state_dict(model_encoder_state_dict, strict=False)

        print(f"Encoder weights loaded successfully from {path}")

    def lock_encoder(self):
        model = self.accelerator.unwrap_model(self.model)
        # Freeze base backbone but keep layout modules trainable
        layout_prefixes = ("layout_embedding.", "layout_injections.", "layout_recon_head.")
        for name, param in model.model.encoder.backbone.named_parameters():
            if not any(name.startswith(p) for p in layout_prefixes):
                param.requires_grad = False
        print("Encoder weights locked (layout modules kept trainable).")

    def unlock_encoder(self):
        model = self.accelerator.unwrap_model(self.model)
        for param in model.model.encoder.backbone.parameters():
            param.requires_grad = True
        print("Encoder weights unlocked.")

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
                    
                    # Accumulate losses as detached GPU scalars; batch-convert
                    # to CPU floats after the loop to avoid per-loss GPU-CPU syncs.
                    _gpu = {}  # key -> detached 0-D GPU tensor

                    for _ in range(self.gradient_accumulate_every):
                        num_context = np.random.randint(1, self.num_context + 1,)
                        data_full = next(self.dl)
                        if isinstance(data_full, list):
                            data = data_full[0]
                            for k, v in data.items():
                                if k in ["ctxt_rgb", "ctxt_c2w", "ctxt_abs_camera_poses"]:
                                    data[k] = v[:, :num_context]
                            data = to_gpu(data, device)
                        # calculate losses
                        with self.accelerator.autocast():
                            losses, misc = self.model(data, global_step=self.step)

                            def _accum(key, val):
                                """Add a detached scalar to the GPU accumulator dict."""
                                if val is None:
                                    return
                                v = val.detach() / self.gradient_accumulate_every
                                _gpu[key] = _gpu.get(key, 0.0) + v if key in _gpu else v

                            rgb_loss = losses["rgb_loss"]
                            _accum("rgb_loss", rgb_loss)
                            rgb_loss = rgb_loss / self.gradient_accumulate_every

                            depth_loss = losses["depth_loss"]
                            _accum("depth_loss", depth_loss)
                            if depth_loss is not None:
                                depth_loss = depth_loss / self.gradient_accumulate_every
                            
                            depth_mask_loss = losses.get("depth_mask_loss")
                            _accum("depth_mask_loss", depth_mask_loss)
                            if depth_mask_loss is not None:
                                depth_mask_loss = depth_mask_loss / self.gradient_accumulate_every

                            semantic_loss = losses.get("semantic_loss", None)
                            _accum("semantic_loss", semantic_loss)
                            if semantic_loss is not None:
                                semantic_loss = semantic_loss / self.gradient_accumulate_every

                            semantic_loss_reg = losses.get("semantic_loss_reg", None)
                            _accum("semantic_reg_loss", semantic_loss_reg)
                            if semantic_loss_reg is not None:
                                semantic_loss_reg = semantic_loss_reg / self.gradient_accumulate_every

                            depth_smooth_loss = losses.get("depth_smooth_loss", None)
                            _accum("depth_smooth_loss", depth_smooth_loss)
                            if depth_smooth_loss is not None:
                                depth_smooth_loss = (
                                    depth_smooth_loss / self.gradient_accumulate_every
                                )

                            rgb_intermediate_loss = losses.get("rgb_intermediate_loss", None)
                            _accum("rgb_intermediate_loss", rgb_intermediate_loss)
                            if rgb_intermediate_loss is not None:
                                rgb_intermediate_loss = (
                                    rgb_intermediate_loss / self.gradient_accumulate_every
                                )

                            depth_intermediate_loss = losses.get("depth_intermediate_loss", None)
                            _accum("depth_intermediate_loss", depth_intermediate_loss)
                            if depth_intermediate_loss is not None:
                                depth_intermediate_loss = (
                                    depth_intermediate_loss / self.gradient_accumulate_every
                                )

                            depth_mask_intermediate_loss = losses.get("depth_mask_intermediate_loss", None)
                            _accum("depth_mask_intermediate_loss", depth_mask_intermediate_loss)
                            if depth_mask_intermediate_loss is not None:
                                depth_mask_intermediate_loss = (
                                    depth_mask_intermediate_loss / self.gradient_accumulate_every
                                )

                            semantic_intermediate_loss = losses.get("semantic_intermediate_loss")
                            _accum("semantic_intermediate_loss", semantic_intermediate_loss)
                            if semantic_intermediate_loss is not None:
                                semantic_intermediate_loss = (
                                    semantic_intermediate_loss / self.gradient_accumulate_every
                                )
                            
                            semantic_intermediate_reg_loss = losses.get("semantic_intermediate_loss_reg")
                            _accum("semantic_intermediate_reg_loss", semantic_intermediate_reg_loss)
                            if semantic_intermediate_reg_loss is not None:
                                semantic_intermediate_reg_loss = (
                                    semantic_intermediate_reg_loss / self.gradient_accumulate_every
                                )
                            
                            depth_smooth_intermediate_loss = losses.get("depth_smooth_intermediate_loss", None)
                            _accum("depth_smooth_intermediate_loss", depth_smooth_intermediate_loss)
                            if depth_smooth_intermediate_loss is not None:
                                depth_smooth_intermediate_loss = (
                                    depth_smooth_intermediate_loss / self.gradient_accumulate_every
                                )

                            lpips_loss = losses["lpips_loss"]
                            _accum("lpips_loss", lpips_loss)
                            lpips_loss = lpips_loss / self.gradient_accumulate_every

                            repa_loss = losses["repa_loss"]
                            _accum("repa_loss", repa_loss)
                            if repa_loss is not None:
                                repa_loss = repa_loss / self.gradient_accumulate_every
                            
                            alignment_loss = losses.get("alignment_loss", None)
                            _accum("alignment_loss", alignment_loss)
                            if alignment_loss is not None:
                                alignment_loss = alignment_loss / self.gradient_accumulate_every

                            layout_recon_loss = losses.get("layout_recon_loss", None)
                            _accum("layout_recon_loss", layout_recon_loss)
                            if layout_recon_loss is not None:
                                layout_recon_loss = layout_recon_loss / self.gradient_accumulate_every

                            clip_semantic_loss = losses.get("clip_semantic_loss", None)
                            _accum("clip_semantic_loss", clip_semantic_loss)
                            if clip_semantic_loss is not None:
                                clip_semantic_loss = clip_semantic_loss / self.gradient_accumulate_every

                            # identity losses
                            if self.use_identity:
                                rgb_loss_ctxt = losses["rgb_loss_ctxt"]
                                _accum("rgb_loss_ctxt", rgb_loss_ctxt)
                                rgb_loss_ctxt = rgb_loss_ctxt / self.gradient_accumulate_every

                                depth_loss_ctxt = losses["depth_loss_ctxt"]
                                _accum("depth_loss_ctxt", depth_loss_ctxt)
                                if depth_loss_ctxt is not None:
                                    depth_loss_ctxt = depth_loss_ctxt / self.gradient_accumulate_every
                                
                                depth_mask_loss_ctxt = losses.get("depth_mask_loss_ctxt")
                                _accum("depth_mask_loss_ctxt", depth_mask_loss_ctxt)
                                if depth_mask_loss_ctxt is not None:
                                    depth_mask_loss_ctxt = depth_mask_loss_ctxt / self.gradient_accumulate_every

                                semantic_loss_ctxt = losses["semantic_loss_ctxt"]
                                _accum("semantic_loss_ctxt", semantic_loss_ctxt)
                                if semantic_loss_ctxt is not None:
                                    semantic_loss_ctxt = semantic_loss_ctxt / self.gradient_accumulate_every

                                semantic_loss_ctxt_reg = losses.get("semantic_loss_ctxt_reg", None)
                                _accum("semantic_loss_ctxt_reg", semantic_loss_ctxt_reg)
                                if semantic_loss_ctxt_reg is not None:
                                    semantic_loss_ctxt_reg = (
                                        semantic_loss_ctxt_reg / self.gradient_accumulate_every
                                    )

                                depth_smooth_loss_ctxt = losses["depth_smooth_loss_ctxt"]
                                _accum("depth_smooth_loss_ctxt", depth_smooth_loss_ctxt)
                                depth_smooth_loss_ctxt = (
                                    depth_smooth_loss_ctxt / self.gradient_accumulate_every
                                )

                                lpips_loss_ctxt = losses["lpips_loss_ctxt"]
                                _accum("lpips_loss_ctxt", lpips_loss_ctxt)
                                lpips_loss_ctxt = lpips_loss_ctxt / self.gradient_accumulate_every

                                repa_loss_ctxt = losses["repa_loss_ctxt"]
                                _accum("repa_loss_ctxt", repa_loss_ctxt)
                                if repa_loss_ctxt is not None:
                                    repa_loss_ctxt = repa_loss_ctxt / self.gradient_accumulate_every

                            loss = rgb_loss

                            if self.use_lpips_loss:
                                loss += self.lpips_loss_weight * lpips_loss

                            if depth_loss is not None:
                                loss += self.depth_loss_weight * depth_loss
                            
                            if depth_mask_loss is not None:
                                loss += self.depth_mask_loss_weight * depth_mask_loss

                            if semantic_loss is not None:
                                loss += self.semantic_loss_weight * semantic_loss
                            
                            if semantic_loss_reg is not None:
                                loss += self.semantic_reg_loss_weight * semantic_loss_reg

                            if self.use_depth_smoothness:
                                loss += self.depth_smooth_loss_weight * depth_smooth_loss
                            
                            if repa_loss is not None:
                                loss += self.repa_loss_weight * repa_loss
                            
                            if alignment_loss is not None:
                                loss += self.vggt_alignment_loss_weight * alignment_loss

                            if layout_recon_loss is not None:
                                loss += self.layout_recon_loss_weight * layout_recon_loss

                            if clip_semantic_loss is not None:
                                loss += self.clip_semantic_loss_weight * clip_semantic_loss

                            if self.use_identity:
                                loss = (
                                    loss
                                    + rgb_loss_ctxt * self.ctxt_losses_factor
                                )
                                if self.use_lpips_loss:
                                    loss += self.lpips_loss_weight * lpips_loss_ctxt * self.ctxt_losses_factor
                                
                                if depth_loss_ctxt is not None:
                                    loss += self.depth_loss_weight * depth_loss_ctxt * self.ctxt_losses_factor

                                if depth_mask_loss_ctxt is not None:
                                    loss += self.depth_mask_loss_weight * depth_mask_loss_ctxt * self.ctxt_losses_factor

                                if semantic_loss_ctxt is not None:
                                    loss += self.semantic_loss_weight * semantic_loss_ctxt * self.ctxt_losses_factor
                                
                                if semantic_loss_ctxt_reg is not None:
                                    loss += self.semantic_reg_loss_weight * semantic_loss_ctxt_reg * self.ctxt_losses_factor

                                if self.use_depth_smoothness:
                                    loss += self.depth_smooth_loss_weight * depth_smooth_loss_ctxt * self.ctxt_losses_factor
                                
                                if repa_loss_ctxt is not None:
                                    loss += self.repa_loss_weight * repa_loss_ctxt * self.ctxt_losses_factor

                            if rgb_intermediate_loss is not None:
                                loss += self.intermediate_weight * rgb_intermediate_loss

                            if depth_intermediate_loss is not None:
                                loss += self.intermediate_weight * self.depth_loss_weight * depth_intermediate_loss
                            
                            if depth_mask_intermediate_loss is not None:
                                loss += self.intermediate_weight * self.depth_mask_loss_weight * depth_mask_intermediate_loss

                            if semantic_intermediate_loss is not None:
                                loss += self.intermediate_weight * self.semantic_loss_weight * semantic_intermediate_loss
                            
                            if semantic_intermediate_reg_loss is not None:
                                loss += self.intermediate_weight * self.semantic_reg_loss_weight * semantic_intermediate_reg_loss

                            if self.use_depth_smoothness and depth_smooth_intermediate_loss is not None:
                                loss += self.intermediate_weight * self.depth_smooth_loss_weight * depth_smooth_intermediate_loss

                            _accum("total_loss", loss)

                        self.accelerator.backward(loss)
                    
                    # Batch-extract all accumulated loss scalars from GPU → CPU
                    # (single sync point instead of 20+ per step)
                    total_loss = _gpu.get("total_loss", torch.tensor(0.0)).item()
                    total_rgb_loss = _gpu.get("rgb_loss", torch.tensor(0.0)).item()
                    total_depth_loss = _gpu.get("depth_loss", torch.tensor(0.0)).item()
                    total_depth_mask_loss = _gpu.get("depth_mask_loss", torch.tensor(0.0)).item()
                    total_semantic_loss = _gpu.get("semantic_loss", torch.tensor(0.0)).item()
                    total_semantic_reg_loss = _gpu.get("semantic_reg_loss", torch.tensor(0.0)).item()
                    total_depth_smooth_loss = _gpu.get("depth_smooth_loss", torch.tensor(0.0)).item()
                    total_rgb_intermediate_loss = _gpu.get("rgb_intermediate_loss", torch.tensor(0.0)).item()
                    total_depth_intermediate_loss = _gpu.get("depth_intermediate_loss", torch.tensor(0.0)).item()
                    total_depth_mask_intermediate_loss = _gpu.get("depth_mask_intermediate_loss", torch.tensor(0.0)).item()
                    total_semantic_intermediate_loss = _gpu.get("semantic_intermediate_loss", torch.tensor(0.0)).item()
                    total_semantic_intermediate_reg_loss = _gpu.get("semantic_intermediate_reg_loss", torch.tensor(0.0)).item()
                    total_depth_smooth_intermediate_loss = _gpu.get("depth_smooth_intermediate_loss", torch.tensor(0.0)).item()
                    total_lpips_loss = _gpu.get("lpips_loss", torch.tensor(0.0)).item()
                    total_repa_loss = _gpu.get("repa_loss", torch.tensor(0.0)).item()
                    total_repa_loss_ctxt = _gpu.get("repa_loss_ctxt", torch.tensor(0.0)).item()
                    total_alignment_loss = _gpu.get("alignment_loss", torch.tensor(0.0)).item()
                    total_layout_recon_loss = _gpu.get("layout_recon_loss", torch.tensor(0.0)).item()
                    total_clip_semantic_loss = _gpu.get("clip_semantic_loss", torch.tensor(0.0)).item()
                    total_rgb_loss_ctxt = _gpu.get("rgb_loss_ctxt", torch.tensor(0.0)).item()
                    total_depth_loss_ctxt = _gpu.get("depth_loss_ctxt", torch.tensor(0.0)).item()
                    total_depth_mask_loss_ctxt = _gpu.get("depth_mask_loss_ctxt", torch.tensor(0.0)).item()
                    total_depth_smooth_loss_ctxt = _gpu.get("depth_smooth_loss_ctxt", torch.tensor(0.0)).item()
                    total_semantic_loss_ctxt = _gpu.get("semantic_loss_ctxt", torch.tensor(0.0)).item()
                    total_semantic_loss_ctxt_reg = _gpu.get("semantic_loss_ctxt_reg", torch.tensor(0.0)).item()
                    total_lpips_loss_ctxt = _gpu.get("lpips_loss_ctxt", torch.tensor(0.0)).item()
                    del _gpu

                    # log the step
                    elapsed_time_sec = time.time() - start_time
                    if accelerator.is_main_process:
                        step_log = {
                                "step": self.step,
                                "loss": total_loss,
                                "rgb_loss": total_rgb_loss,
                                "depth_loss": total_depth_loss,
                                "depth_mask_loss": total_depth_mask_loss,
                                "repa_loss": total_repa_loss,
                                "alignment_loss": total_alignment_loss,
                                "layout_recon_loss": total_layout_recon_loss,
                                "clip_semantic_loss": total_clip_semantic_loss,
                                "semantic_loss": total_semantic_loss * self.semantic_loss_weight,
                                "semantic_loss_reg": total_semantic_reg_loss * self.semantic_reg_loss_weight,
                                "rgb_intermediate_loss": total_rgb_intermediate_loss,
                                "depth_intermediate_loss": total_depth_intermediate_loss,
                                "depth_mask_intermediate_loss": total_depth_mask_intermediate_loss,
                                "semantic_intermediate_loss": total_semantic_intermediate_loss * self.semantic_loss_weight,
                                "semantic_intermediate_reg_loss": total_semantic_intermediate_reg_loss * self.semantic_reg_loss_weight,
                                "lr": self.lr_scheduler.get_last_lr()[0],
                                "num_context": data["ctxt_rgb"].shape[1],
                                "elapsed_time_sec": elapsed_time_sec,
                        }
                        if self.use_lpips_loss:
                            step_log.update({
                                "lpips_loss": total_lpips_loss,
                            })
                        if self.use_depth_smoothness:
                            step_log.update({
                                "depth_smooth_loss": total_depth_smooth_loss,
                                "depth_smooth_intermediate_loss": total_depth_smooth_intermediate_loss * self.depth_smooth_loss_weight,
                            })
                        if self.use_identity:
                            step_log.update({
                                "rgb_loss_ctxt": total_rgb_loss_ctxt,
                                "depth_loss_ctxt": total_depth_loss_ctxt,
                                "depth_mask_loss_ctxt": total_depth_mask_loss_ctxt,
                                "repa_loss_ctxt": total_repa_loss_ctxt,
                                "semantic_loss_ctxt": total_semantic_loss_ctxt * self.semantic_loss_weight,
                                "semantic_loss_ctxt_reg": total_semantic_loss_ctxt_reg * self.semantic_reg_loss_weight,
                            })
                            if self.use_lpips_loss:
                                step_log.update({
                                    "lpips_loss_ctxt": total_lpips_loss_ctxt,
                                })
                            if self.use_depth_smoothness:
                                step_log.update({
                                    "depth_smooth_loss_ctxt": total_depth_smooth_loss_ctxt,
                                })
                        json_logger.log(step_log)
                        wandb.log(step_log, step=self.step)
                    
                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    pbar.set_description(f"loss: {total_loss:.4f}")

                    accelerator.wait_for_everyone()
                    self.opt.step()
                    self.opt.zero_grad()

                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:  # or not accelerator.is_main_process:
                        self.ema.to(device)
                        self.ema.update()
                    # diffusion sample
                    if accelerator.is_main_process and (self.step != 0 and self.step % self.sample_every == 0):
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            print(f"batch sizes: {batches}")
                            def sample(n):
                                data.update((k, v[:n]) for k, v in data.items())
                                return self.ema.ema_model.sample(batch_size=n, inp=data)

                            output_dict_list = list(map(lambda n: sample(n), batches,))
                            out = output_dict_list[0]
        
                        frames, depth_frames, semantics = prepare_video_viz(out)

                        save_folder_milestone_frames = os.path.join(self.results_folder, f'milestone_{milestone}_rendered_frames')
                        os.makedirs(
                            save_folder_milestone_frames, exist_ok=True,
                        )
                        # save video
                        denoised_f = os.path.join(self.results_folder, f"milestone_{milestone}_rendered_rgb.mp4")
                        imageio.mimwrite(denoised_f, frames, fps=10, quality=10)
                        denoised_f_depth = os.path.join(self.results_folder, f"milestone_{milestone}_rendered_depth.mp4")
                        imageio.mimwrite(denoised_f_depth, depth_frames, fps=10, quality=10)
                        denoised_f_semantic = os.path.join(self.results_folder, f"milestone_{milestone}_rendered_semantic.mp4")
                        imageio.mimwrite(denoised_f_semantic, semantics, fps=10, quality=10)

                        # save all frames
                        for p, frame in enumerate(frames):
                            Image.fromarray(frame).save(
                                os.path.join(save_folder_milestone_frames, f"rendered_{p}.png")
                            )

                        # --- Save GT images and scene graph info ---
                        _save_gt_and_sg(
                            data, save_folder_milestone_frames,
                            self.results_folder, milestone, out=out,
                        )
                        
                        # Log videos to wandb
                        if wandb.run is not None:
                            wandb.log({
                                f"video/milestone_{milestone}_rendered_rgb": wandb.Video(denoised_f, fps=10, format="mp4"),
                                f"video/milestone_{milestone}_rendered_depth": wandb.Video(denoised_f_depth, fps=10, format="mp4"),
                            }, step=self.step)
                            if self.use_semantic:
                                wandb.log({
                                    f"video/milestone_{milestone}_rendered_semantic": wandb.Video(denoised_f_semantic, fps=10, format="mp4")
                                }, step=self.step)

                        print(f"saved denoised video at {milestone} milestones")
                        if (
                            self.step % self.wandb_every == 0
                        ):
                            if self.step != 0 and self.step % self.save_every == 0:
                                milestone = self.step // self.save_every
                                self.save(milestone)
                                print(f"saved model at {milestone} milestones")

                    accelerator.wait_for_everyone()

                    self.step += 1

                    self.lr_scheduler.step()
                    pbar.update(1)

        accelerator.print("training complete")


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


def _unnormalize(img_tensor):
    """Convert from [-1,1] to [0,255] uint8 (C,H,W) -> (H,W,C)."""
    img = (img_tensor.clamp(-1, 1) + 1) * 0.5 * 255
    return img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def _unnormalize_01(img_tensor):
    """Convert from [0,1] to [0,255] uint8 (C,H,W) -> (H,W,C)."""
    img = img_tensor.clamp(0, 1) * 255
    return img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def _project_bboxes_to_image(
    node_positions,  # (M, 3) world-space centres
    node_sizes,      # (M, 3) full AABB dimensions
    node_mask,       # (M,) bool
    camera_c2w,      # (4, 4)
    intrinsics,      # (3, 3)
    image_hw,        # (H, W)
    node_names=None, # list of str or None
    node_is_wall=None,      # (M,) bool – optional
    wall_endpoints=None,    # (M, 2, 3) – optional
    wall_heights=None,      # (M,) – optional
):
    """Project 3D AABBs (and wall quads) into a camera view, return list of 2D boxes sorted far-to-near.

    Returns list of dicts with keys: u_min, v_min, u_max, v_max, depth, name, color.
    """

    M = node_positions.shape[0]
    H, W = image_hw
    w2c = torch.inverse(camera_c2w.float())
    R = w2c[:3, :3]
    t_vec = w2c[:3, 3]
    fx, fy = intrinsics[0, 0].item(), intrinsics[1, 1].item()
    cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()

    # 8 corner offsets for AABBs
    signs = node_positions.new_tensor(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
         [ 1, -1, -1], [ 1, -1, 1], [ 1, 1, -1], [ 1, 1, 1]]
    )

    boxes = []
    # Distinct colors for up to 30 objects
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 100, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 64, 64),
        (64, 255, 64), (64, 64, 255), (200, 200, 0), (200, 0, 200), (0, 200, 200),
        (255, 160, 122), (144, 238, 144), (173, 216, 230), (238, 130, 238),
        (245, 222, 179), (188, 143, 143), (72, 61, 139), (0, 128, 128),
        (255, 215, 0), (219, 112, 147), (46, 139, 87), (210, 105, 30),
        (100, 149, 237), (250, 128, 114), (152, 251, 152),
    ]
    # Dedicated wall color (orange-red)
    wall_color = (255, 100, 50)

    has_walls = (node_is_wall is not None and wall_endpoints is not None
                 and wall_heights is not None)

    # Floor type names to skip in bbox visualization (too large / uninformative)
    _skip_names = {"Floor"}

    for i in range(M):
        if not node_mask[i]:
            continue

        # Skip floor nodes — they produce huge uninformative bboxes
        if node_names is not None and node_names[i] in _skip_names:
            continue

        is_wall = has_walls and node_is_wall[i].item()

        if is_wall:
            # 4 quad corners: bottom p1, bottom p2, top p1, top p2
            p1 = wall_endpoints[i, 0]  # (3,)
            p2 = wall_endpoints[i, 1]  # (3,)
            h = wall_heights[i].item()
            corners = torch.stack([
                p1,
                p2,
                p1 + torch.tensor([0, h, 0], device=p1.device, dtype=p1.dtype),
                p2 + torch.tensor([0, h, 0], device=p1.device, dtype=p1.dtype),
            ])  # (4, 3)
        else:
            half = node_sizes[i] / 2                        # (3,)
            corners = node_positions[i].unsqueeze(0) + half.unsqueeze(0) * signs  # (8, 3)

        # Transform to camera space
        corners_cam = (R @ corners.T).T + t_vec
        z_vals = corners_cam[:, 2]
        NEAR_Z = 0.01
        visible = z_vals > NEAR_Z
        if not visible.any():
            continue
        # Project visible corners
        z_safe = z_vals.clamp(min=NEAR_Z)
        u_px = (fx * (corners_cam[:, 0] / z_safe) + cx) * W    # pixel coords
        v_px = (fy * (corners_cam[:, 1] / z_safe) + cy) * H

        u_vis = u_px[visible]
        v_vis = v_px[visible]
        u_min = u_vis.min().item()
        u_max = u_vis.max().item()
        v_min = v_vis.min().item()
        v_max = v_vis.max().item()

        # Near-plane clipping: extend bbox with edge-plane intersections
        if (~visible).any():
            n_c = corners_cam.shape[0]
            _edges = (
                [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
                if n_c == 8 else [(0,1),(0,2),(1,3),(2,3)]
            )
            for ei0, ei1 in _edges:
                z0, z1 = z_vals[ei0], z_vals[ei1]
                if (z0 > NEAR_Z) == (z1 > NEAR_Z):
                    continue  # both on same side — no crossing
                dz = z1 - z0
                if dz.abs().item() < 1e-8:
                    continue
                t = ((NEAR_Z - z0) / dz).clamp(0.0, 1.0)
                cp = corners_cam[ei0] + t * (corners_cam[ei1] - corners_cam[ei0])
                cz = max(cp[2].item(), NEAR_Z)
                cu = (fx * (cp[0].item() / cz) + cx) * W
                cv = (fy * (cp[1].item() / cz) + cy) * H
                u_min = min(u_min, cu)
                u_max = max(u_max, cu)
                v_min = min(v_min, cv)
                v_max = max(v_max, cv)

        # Skip if completely outside the image
        if u_max < 0 or u_min > W or v_max < 0 or v_min > H:
            continue

        # Center depth for sorting
        center_cam = R @ node_positions[i] + t_vec
        depth = center_cam[2].item()

        name = node_names[i] if node_names is not None else f"obj_{i}"
        color = wall_color if is_wall else palette[i % len(palette)]
        boxes.append({
            "u_min": max(0, u_min), "v_min": max(0, v_min),
            "u_max": min(W, u_max), "v_max": min(H, v_max),
            "depth": depth, "name": name, "color": color,
        })

    # Sort far to near so closer objects are drawn on top
    boxes.sort(key=lambda b: -b["depth"])
    return boxes


def _draw_bboxes_on_image(pil_img, boxes):
    """Draw projected bounding boxes on a PIL image with a legend panel.

    Returns a new image: [original with bbox outlines | legend panel].
    boxes is sorted far-to-near (far drawn first, near on top).
    """

    # --- Draw bbox rectangles (no inline labels) ---
    draw = ImageDraw.Draw(pil_img)
    for box in boxes:
        rect = [box["u_min"], box["v_min"], box["u_max"], box["v_max"]]
        draw.rectangle(rect, outline=box["color"], width=2)

    # --- Build legend panel ---
    # Deduplicate: show each unique (name, color) once, ordered near-to-far
    seen = set()
    legend_items = []
    for box in reversed(boxes):  # near first
        key = (box["name"], box["color"])
        if key not in seen:
            seen.add(key)
            legend_items.append({"name": box["name"], "color": box["color"],
                                 "depth": box["depth"]})

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()

    swatch_size = 12
    row_height = swatch_size + 6
    padding = 6
    # Measure max text width
    tmp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_tw = 0
    for item in legend_items:
        label = f"{item['name']} ({item['depth']:.1f}m)"
        try:
            bb = tmp_draw.textbbox((0, 0), label, font=font)
            tw = bb[2] - bb[0]
        except AttributeError:
            tw, _ = tmp_draw.textsize(label, font=font)
        max_tw = max(max_tw, tw)

    legend_w = padding + swatch_size + 4 + max_tw + padding
    legend_h = pil_img.size[1]

    legend = Image.new("RGB", (legend_w, legend_h), (255, 255, 255))
    ld = ImageDraw.Draw(legend)

    y = padding
    for item in legend_items:
        if y + row_height > legend_h - padding:
            break
        c = item["color"]
        ld.rectangle([padding, y, padding + swatch_size, y + swatch_size],
                     fill=c, outline=(0, 0, 0))
        label = f"{item['name']} ({item['depth']:.1f}m)"
        ld.text((padding + swatch_size + 4, y - 1), label,
                fill=(0, 0, 0), font=font)
        y += row_height

    # --- Combine image + legend ---
    combined = Image.new("RGB", (pil_img.size[0] + legend_w, pil_img.size[1]),
                         (255, 255, 255))
    combined.paste(pil_img, (0, 0))
    combined.paste(legend, (pil_img.size[0], 0))
    return combined


def _layout_cls_to_colormap(layout_cls, img_h, img_w, id_to_type=None):
    """Convert a per-pixel class map to a colour image (deterministic palette).

    Args:
        layout_cls: (H, W) int numpy array — per-pixel class IDs (0 = background).
        img_h, img_w: target output image size (will resize from layout resolution).
        id_to_type: optional list mapping id → type name.

    Returns:
        PIL.Image (img_h, img_w, RGB) with legend-style colour coding.
    """
    h, w = layout_cls.shape
    # Generate a deterministic palette: class_id → RGB
    rng = np.random.RandomState(42)
    n_classes = int(layout_cls.max()) + 1
    palette = rng.randint(60, 230, size=(n_classes, 3)).astype(np.uint8)
    palette[0] = [0, 0, 0]  # background = black

    # Map class IDs → RGB
    color_map = palette[layout_cls]  # (h, w, 3)
    pil = Image.fromarray(color_map, "RGB")
    if (h, w) != (img_h, img_w):
        pil = pil.resize((img_w, img_h), Image.NEAREST)
    return pil


def _save_gt_and_sg(data, frames_folder, results_folder, milestone, out=None):
    """Save GT images and scene graph info alongside rendered outputs.

    Args:
        data: batch dict on GPU (from the last training iteration).
        frames_folder: per-milestone folder for individual frames.
        results_folder: Path object for the top-level results directory.
        milestone: current milestone number.
        out: output dict from sampling (contains 'images' key with denoised target).
    """
    results_folder = str(results_folder)

    # --- GT target images ---
    if "trgt_rgb" in data:
        trgt = data["trgt_rgb"]  # (B, N_trgt, C, H, W) or (B, C, H, W)
        if trgt.dim() == 5:
            trgt = trgt[:, 0]  # take first target view
        for b in range(trgt.shape[0]):
            gt_img = _unnormalize(trgt[b])
            Image.fromarray(gt_img).save(
                os.path.join(frames_folder, f"gt_target_b{b}.png")
            )

    # --- GT context images ---
    if "ctxt_rgb" in data:
        ctxt = data["ctxt_rgb"]  # (B, N_ctxt, C, H, W)
        if ctxt.dim() == 5:
            for b in range(ctxt.shape[0]):
                for v in range(ctxt.shape[1]):
                    ctx_img = _unnormalize(ctxt[b, v])
                    Image.fromarray(ctx_img).save(
                        os.path.join(frames_folder, f"gt_context_b{b}_v{v}.png")
                    )
        elif ctxt.dim() == 4:
            for b in range(ctxt.shape[0]):
                ctx_img = _unnormalize(ctxt[b])
                Image.fromarray(ctx_img).save(
                    os.path.join(frames_folder, f"gt_context_b{b}.png")
                )

    # --- Scene graph info (only if SG keys are present) ---
    if "sg_node_types" not in data:
        return

    node_types = data["sg_node_types"].cpu()    # (B, max_nodes)
    node_mask = data["sg_node_mask"].cpu()       # (B, max_nodes)
    node_pos = data.get("sg_node_positions")     # (B, max_nodes, 3) or None
    node_sizes = data.get("sg_node_sizes")       # (B, max_nodes, 3) or None
    node_rots = data.get("sg_node_rotations")    # (B, max_nodes, 3) or None
    edge_index = data.get("sg_edge_index")       # (B, 2, max_edges) or None
    edge_types = data.get("sg_edge_types")       # (B, max_edges) or None
    edge_mask = data.get("sg_edge_mask")         # (B, max_edges) or None

    edge_type_names = {0: "CONTAINS", 1: "NEAR", 2: "SAME_ROOM"}

    # Try to recover id_to_type from the backbone embeddings table
    id_to_type = None
    try:
        from splat_belief.utils.procthor_utils import load_vocabulary, _clean_display_name
        if wandb.run is not None:
            wc = OmegaConf.to_container(wandb.config, resolve=True) if hasattr(wandb.config, '_metadata') else dict(wandb.config)
            vocab_dir = None
            if isinstance(wc, dict) and "dataset" in wc and isinstance(wc["dataset"], dict):
                vocab_dir = wc["dataset"].get("vocab_dir")
            if vocab_dir:
                _, id_to_type = load_vocabulary(vocab_dir)
    except Exception:
        pass

    for b in range(node_types.shape[0]):
        mask_b = node_mask[b].bool()
        n_valid = mask_b.sum().item()
        types_b = node_types[b][mask_b].tolist()

        sg_info = {
            "num_nodes": n_valid,
            "node_type_ids": types_b,
        }
        if id_to_type is not None:
            sg_info["node_type_names"] = [
                _clean_display_name(id_to_type[t]) if t < len(id_to_type) else f"<unk:{t}>"
                for t in types_b
            ]
        if node_pos is not None:
            sg_info["node_positions"] = node_pos[b][mask_b].cpu().tolist()
        if node_sizes is not None:
            sg_info["node_bbox_sizes"] = node_sizes[b][mask_b].cpu().tolist()
        if node_rots is not None:
            sg_info["node_rotations"] = node_rots[b][mask_b].cpu().tolist()

        if edge_index is not None and edge_mask is not None:
            emask_b = edge_mask[b].cpu().bool()
            n_edges = emask_b.sum().item()
            ei = edge_index[b].cpu()
            et = edge_types[b].cpu()
            edges_list = []
            for e in range(n_edges):
                src, dst = ei[0, e].item(), ei[1, e].item()
                etype = et[e].item()
                edge_entry = {"src": src, "dst": dst, "type_id": etype,
                              "type_name": edge_type_names.get(etype, str(etype))}
                edges_list.append(edge_entry)
            sg_info["num_edges"] = n_edges
            sg_info["edges"] = edges_list

        sg_path = os.path.join(frames_folder, f"scene_graph_b{b}.json")
        with open(sg_path, "w") as f:
            _json.dump(sg_info, f, indent=2)

    # --- Projected bbox overlay on GT target images ---
    has_sg_geom = (node_pos is not None and node_sizes is not None)
    has_camera = ("trgt_abs_camera_poses" in data and "intrinsics" in data)
    if has_sg_geom and has_camera:
        trgt_c2w = data["trgt_abs_camera_poses"].cpu()   # (B, V_t, 4, 4)
        intr = data["intrinsics"].cpu()                   # (B, 3, 3)
        trgt_rgb = data["trgt_rgb"]                       # (B, V_t, C, H, W) or (B, C, H, W)
        if trgt_rgb.dim() == 5:
            trgt_rgb = trgt_rgb[:, 0]                     # (B, C, H, W)
        _, _, H_img, W_img = trgt_rgb.shape

        # Optional wall tensors
        w_is_wall = data.get("sg_node_is_wall")
        w_endpoints = data.get("sg_wall_endpoints")
        w_heights = data.get("sg_wall_heights")

        # Build per-node name list from id_to_type
        for b in range(trgt_c2w.shape[0]):
            mask_b = node_mask[b].bool()
            if id_to_type is not None:
                names = [
                    _clean_display_name(id_to_type[t]) if t < len(id_to_type) else f"obj_{t}"
                    for t in node_types[b].tolist()
                ]
            else:
                names = [f"obj_{i}" for i in range(node_types.shape[1])]

            # Target view (first target)
            c2w_b = trgt_c2w[b, 0] if trgt_c2w.dim() == 4 else trgt_c2w[b]
            boxes = _project_bboxes_to_image(
                node_pos[b].cpu(), node_sizes[b].cpu(), mask_b,
                c2w_b, intr[b], (H_img, W_img), node_names=names,
                node_is_wall=w_is_wall[b].cpu() if w_is_wall is not None else None,
                wall_endpoints=w_endpoints[b].cpu() if w_endpoints is not None else None,
                wall_heights=w_heights[b].cpu() if w_heights is not None else None,
            )
            if boxes:
                gt_img = _unnormalize(trgt_rgb[b])
                overlay = Image.fromarray(gt_img.copy())
                overlay = _draw_bboxes_on_image(overlay, boxes)
                overlay.save(os.path.join(frames_folder, f"gt_target_bbox_b{b}.png"))

    # --- Side-by-side GT target vs denoised/rendered target (first batch element) ---
    gt_path = os.path.join(frames_folder, "gt_target_b0.png")
    # Use denoised target from diffusion output if available
    rendered_target_path = None
    if out is not None and "images" in out:
        denoised = out["images"]  # (B, C, H, W) in [0, 1]
        if denoised.dim() == 4 and denoised.shape[0] > 0:
            rd_img = _unnormalize_01(denoised[0])
            rendered_target_path = os.path.join(frames_folder, "denoised_target_b0.png")
            Image.fromarray(rd_img).save(rendered_target_path)
    # Fallback to last rendered frame (target viewpoint) if no denoised output
    if rendered_target_path is None:
        rendered_frames = sorted(glob.glob(os.path.join(frames_folder, "rendered_*.png")))
        if rendered_frames:
            rendered_target_path = rendered_frames[-1]

    if os.path.exists(gt_path) and os.path.exists(rendered_target_path):
        gt_im = Image.open(gt_path)
        rd_im = Image.open(rendered_target_path)
        if gt_im.size[1] != rd_im.size[1]:
            h = min(gt_im.size[1], rd_im.size[1])
            gt_im = gt_im.resize((int(gt_im.size[0] * h / gt_im.size[1]), h))
            rd_im = rd_im.resize((int(rd_im.size[0] * h / rd_im.size[1]), h))
        combined = Image.new("RGB", (gt_im.size[0] + rd_im.size[0], gt_im.size[1]))
        combined.paste(gt_im, (0, 0))
        combined.paste(rd_im, (gt_im.size[0], 0))
        cmp_path = os.path.join(results_folder, f"milestone_{milestone}_gt_vs_rendered.png")
        combined.save(cmp_path)

        if wandb.run is not None:
            wandb.log({
                f"comparison/milestone_{milestone}": wandb.Image(cmp_path,
                    caption="Left: GT target | Right: Denoised target"),
            })

    # --- Side-by-side GT target with bbox overlay vs denoised target ---
    bbox_path = os.path.join(frames_folder, "gt_target_bbox_b0.png")
    if os.path.exists(bbox_path) and rendered_target_path is not None and os.path.exists(rendered_target_path):
        gt_bbox_im = Image.open(bbox_path)
        rd_im = Image.open(rendered_target_path)
        if gt_bbox_im.size[1] != rd_im.size[1]:
            h = min(gt_bbox_im.size[1], rd_im.size[1])
            gt_bbox_im = gt_bbox_im.resize((int(gt_bbox_im.size[0] * h / gt_bbox_im.size[1]), h))
            rd_im = rd_im.resize((int(rd_im.size[0] * h / rd_im.size[1]), h))
        combined_bbox = Image.new("RGB", (gt_bbox_im.size[0] + rd_im.size[0], gt_bbox_im.size[1]))
        combined_bbox.paste(gt_bbox_im, (0, 0))
        combined_bbox.paste(rd_im, (gt_bbox_im.size[0], 0))
        cmp_bbox_path = os.path.join(results_folder, f"milestone_{milestone}_gt_bbox_vs_rendered.png")
        combined_bbox.save(cmp_bbox_path)

        if wandb.run is not None:
            wandb.log({
                f"comparison/milestone_{milestone}_bbox": wandb.Image(cmp_bbox_path,
                    caption="Left: GT target + projected bboxes | Right: Denoised target"),
            })

    # --- Dense layout colormap visualization ---
    if has_sg_geom and has_camera and "trgt_rgb" in data:
        try:
            from splat_belief.utils.procthor_utils import rasterize_scene_graph
            trgt_c2w_gpu = data["trgt_abs_camera_poses"]   # (B, V_t, 4, 4)
            intr_gpu = data["intrinsics"]                   # (B, 3, 3)
            trgt_rgb_vis = data["trgt_rgb"]
            if trgt_rgb_vis.dim() == 5:
                trgt_rgb_vis = trgt_rgb_vis[:, 0]           # (B, C, H, W)
            _, _, H_img, W_img = trgt_rgb_vis.shape

            # Use only the first target view for rasterization
            cam_c2w = trgt_c2w_gpu[:, :1] if trgt_c2w_gpu.dim() == 4 else trgt_c2w_gpu.unsqueeze(1)

            with torch.no_grad():
                layout_cls_map, layout_depth_map = rasterize_scene_graph(
                    sg_node_types=data["sg_node_types"],
                    sg_node_positions=data["sg_node_positions"],
                    sg_node_sizes=data["sg_node_sizes"],
                    sg_node_mask=data["sg_node_mask"],
                    camera_c2w=cam_c2w,
                    intrinsics=intr_gpu,
                    output_h=32, output_w=32,
                    sg_wall_endpoints=data.get("sg_wall_endpoints"),
                    sg_wall_heights=data.get("sg_wall_heights"),
                    sg_node_is_wall=data.get("sg_node_is_wall"),
                    sg_node_is_door=data.get("sg_node_is_door"),
                )

            # Visualize first batch element
            cls_np = layout_cls_map[0].cpu().numpy()     # (32, 32)
            colormap_img = _layout_cls_to_colormap(cls_np, H_img, W_img, id_to_type=id_to_type)
            colormap_img.save(os.path.join(frames_folder, "layout_colormap_b0.png"))

            # Alpha-blend overlay on GT target
            gt_img_pil = Image.fromarray(_unnormalize(trgt_rgb_vis[0]))
            overlay = Image.blend(gt_img_pil, colormap_img.resize(gt_img_pil.size), alpha=0.4)
            overlay.save(os.path.join(frames_folder, "layout_overlay_b0.png"))

            if wandb.run is not None:
                wandb.log({
                    f"layout/milestone_{milestone}_colormap": wandb.Image(
                        os.path.join(frames_folder, "layout_colormap_b0.png"),
                        caption="Dense layout class colormap (32x32 rasterized)"),
                    f"layout/milestone_{milestone}_overlay": wandb.Image(
                        os.path.join(frames_folder, "layout_overlay_b0.png"),
                        caption="GT target + layout overlay (alpha=0.4)"),
                })
        except Exception as e:
            import traceback
            traceback.print_exc()