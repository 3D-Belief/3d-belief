# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
import sys
import os
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion.diffusion import (
    Diffusion,
    Trainer,
)
import data_io

from splat import SplatBelief
from splat.decoder import get_decoder
from splat.encoder import get_encoder
from embodied.semantic_mapper import SemanticMapper
import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from utils.model_utils import build_2d_model, load_repa_encoder

import clip
import open_clip


@hydra.main(
    version_base=None, config_path="../config/", config_name="config",
)
def train(cfg: DictConfig):
    train_settings = get_train_settings(cfg.setting_name, cfg.ngpus)
    cfg.num_context = train_settings["num_context"]
    cfg.num_target = train_settings["num_target"]

    # initialize the accelerator at the beginning
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        split_batches=True, mixed_precision="no", kwargs_handlers=[ddp_kwargs],
    )


    dataset = data_io.get_dataset(cfg)
    accelerator.print(f"length dataset {len(dataset)}")

    dl = DataLoader(
        dataset,
        batch_size=train_settings["batch_size"] // accelerator.num_processes,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        worker_init_fn=lambda id: np.random.seed(id * 4),
    )
    depth_mode = cfg.depth_mode
    extended_visualization = cfg.extended_visualization
    use_depth_supervision = cfg.use_depth_supervision

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    use_semantic = cfg.model.encoder.use_semantic
    use_reg_model = cfg.model.encoder.use_reg_model
    reg_model_name = cfg.model.encoder.reg_model_name
    use_depth_mask = cfg.model.encoder.use_depth_mask
    predict_depth_mask = cfg.model.encoder.predict_depth_mask
    semantic_config = cfg.semantic_config
    semantic_mode = cfg.semantic_mode
    semantic_viz = cfg.semantic_viz
    use_object_binary_mask = cfg.use_object_binary_mask
    background_weight = cfg.background_weight
    repa_encoder_resolution = cfg.repa_encoder_resolution
    use_vggt_alignment = cfg.model.encoder.backbone.get("use_vggt_alignment", False)

    repa_encoder = None
    if cfg.model.encoder.backbone.use_repa:
        repa_encoder, _, _ = load_repa_encoder(
            enc_name='dinov2-vit-b',
            device=accelerator.device,
            resolution=repa_encoder_resolution,
        )

    clip_model = None
    if use_semantic:
        # clip_model, _ = clip.load("ViT-B/16", device="cuda")
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            precision="fp16",
        )
        clip_model = clip_model.to("cuda")

        clip_model.eval()

    dino_model = None
    if use_semantic and use_reg_model: 
        dino_model = build_2d_model(model_name=reg_model_name)

    text_encoder = None
    text_tokenizer = None
    if use_semantic and semantic_mode == "embed":
        text_encoder = clip_model.encode_text
        # text_tokenizer = clip.tokenize
        text_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    semantic_mapper = None
    if use_semantic:
        semantic_mapper = SemanticMapper(
            config_path=semantic_config, 
            mode=semantic_mode, 
            text_encoder=text_encoder, 
            text_tokenizer=text_tokenizer, 
            semantic_viz=semantic_viz
        )
    
    model = SplatBelief(
        encoder,
        encoder_visualizer,
        decoder,
        semantic_mapper=semantic_mapper,
        depth_mode=depth_mode,
        extended_visualization=extended_visualization,
        use_semantic=use_semantic,
        semantic_mode=semantic_mode,
        inference_mode=cfg.model.encoder.inference_mode,
        use_depth_mask=use_depth_mask,
        clip_model=clip_model,
        dino_model=dino_model,
        repa_encoder=repa_encoder,
        repa_encoder_name='dinov2-vit-b',
        repa_encoder_resolution=repa_encoder_resolution,
    ).cuda()

    diffusion = Diffusion(
        model,
        image_size=dataset.image_size,
        timesteps=1000,  # number of steps
        sampling_timesteps=10,
        loss_type="l2",  # L1 or L2
        objective="pred_x0",
        beta_schedule="cosine",
        use_guidance=cfg.use_guidance,
        guidance_scale=1.0,
        clean_target=cfg.clean_target,
        use_depth_supervision=use_depth_supervision,
        use_semantic=use_semantic,
        use_reg_model=use_reg_model,
        use_depth_mask=use_depth_mask,
        predict_depth_mask=predict_depth_mask,
        use_object_binary_mask=use_object_binary_mask,
        background_weight=background_weight,
        use_vggt_alignment=use_vggt_alignment,
        cfg=cfg,
    ).cuda()

    print(f"using settings {train_settings}")
    warmup_period = cfg.warmup_period

    print(f"using lr {cfg.lr}")
    trainer = Trainer(
        diffusion,
        accelerator=accelerator,
        dataloader=dl,
        train_batch_size=train_settings["batch_size"],
        train_lr=cfg.lr,
        train_num_steps=cfg.train_num_steps,  # total training steps
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=cfg.ema_decay,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        sample_every=2000,
        wandb_every=2000,
        save_every=2000,
        results_folder=cfg.results_folder,
        num_samples=1,
        warmup_period=warmup_period,
        checkpoint_path=cfg.checkpoint_path,
        wandb_config=cfg.wandb,
        run_name=cfg.name,
        depth_loss_weight=cfg.dataset.depth_loss_weight,
        depth_mask_loss_weight=cfg.dataset.depth_mask_loss_weight,
        depth_smooth_loss_weight=cfg.dataset.depth_smooth_loss_weight,
        semantic_loss_weight=cfg.dataset.semantic_loss_weight,
        semantic_reg_loss_weight=cfg.dataset.semantic_reg_loss_weight,
        lpips_loss_weight=cfg.dataset.lpips_loss_weight,
        repa_loss_weight=cfg.dataset.repa_loss_weight,
        intermediate_weight=cfg.dataset.intermediate_weight,
        vggt_alignment_loss_weight=cfg.dataset.vggt_alignment_loss_weight,
        ctxt_losses_factor=cfg.ctxt_losses_factor,
        # rgb_loss_weight=cfg.dataset.rgb_loss_weight,
        cfg=cfg,
        num_context=train_settings["num_context"],
        load_enc=cfg.load_enc,
        lock_enc_steps=cfg.lock_enc_steps,
        use_identity=cfg.use_identity,
        load_optimizer=cfg.load_optimizer,
        use_depth_smoothness=cfg.use_depth_smoothness,
        use_lpips_loss=cfg.use_lpips_loss,
        use_semantic=use_semantic,
    )
    trainer.train()


def get_train_settings(name, ngpus):
    if name == "re":
        return {
            "n_coarse": 64,
            "n_fine": 64,
            "n_coarse_coarse": 32,
            "n_coarse_fine": 0,
            "num_pixels": int(24 ** 2),
            "batch_size": 3 * ngpus,
            "num_context": 1,
            "num_target": 2,
            "n_feats_out": 64,
            "use_viewdir": False,
            "sampling": "patch",
            "self_condition": False,
            "lindisp": False,
        }
    elif name == "hab":
        return {
            "n_coarse": 64,
            "n_fine": 64,
            "n_coarse_coarse": 32,
            "n_coarse_fine": 0,
            "num_pixels": int(24 ** 2),
            "batch_size": 3 * ngpus,
            "num_context": 1,
            "num_target": 1,
            "n_feats_out": 64,
            "use_viewdir": False,
            "sampling": "patch",
            "self_condition": False,
            "lindisp": False,
        }
    elif name == "pixelsplat_h100":
        return {
            "n_coarse": 64,
            "n_fine": 64,
            "n_coarse_coarse": 32,
            "n_coarse_fine": 0,
            "num_pixels": int(24 ** 2),
            "batch_size": 3 * ngpus,
            "num_context": 1,
            "num_target": 1,
            "n_feats_out": 64,
            "use_viewdir": False,
            "sampling": "patch",
            "self_condition": False,
            "lindisp": False,
        }
    elif name == "pixelsplat":
        return {
            "batch_size": ngpus,
            "num_context": 1,
            "num_target": 1,
            "use_viewdir": False,
            "sampling": "patch",
            "lindisp": False,
        }
    elif name == "debug":
        return {
            "batch_size": 1 * ngpus,
            "num_context": 1,
            "num_target": 1,
        }
    else:
        raise ValueError(f"unknown setting {name}")


if __name__ == "__main__":
    print(f"running {__file__}")
    train()
