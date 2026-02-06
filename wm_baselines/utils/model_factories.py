from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import open_clip
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from splat import PixelSplatEpiState
from splat.decoder import get_decoder
from splat.encoder import get_encoder
from data_io import get_dataset
from model_utils import build_2d_model
from splat_belief.embodied.semantic_mapper import SemanticMapper
from diffusion.denoising_diffusion_pixelsplat_epi_temporal import GaussianDiffusionPixelEpiTemporal

def build_splat_model(
    encoder_visual_pair: Dict[str, Any],
    decoder: Any,
    semantic: Optional[Dict[str, Any]] = None,
    depth_mode: str = "metric",
    extended_visualization: bool = False,
    use_history: bool = False,
    use_semantic: Optional[bool] = None,
    inference_mode: Optional[str] = None,
) -> Any:

    enc = encoder_visual_pair["encoder"]
    viz = encoder_visual_pair["visualizer"]

    clip_model = dino_model = semantic_mapper = None
    if semantic:
        clip_model = semantic.get("clip_model")
        dino_model = semantic.get("dino_model")
        semantic_mapper = semantic.get("semantic_mapper")
    return PixelSplatEpiState(
        enc,
        viz,
        decoder,
        semantic_mapper=semantic_mapper,
        depth_mode=depth_mode,
        extended_visualization=extended_visualization,
        use_history=use_history,
        use_semantic=bool(use_semantic) if use_semantic is not None else (semantic is not None),
        inference_mode=inference_mode,
        clip_model=clip_model,
        dino_model=dino_model,
    ).cuda()

def get_encoder_pair(cfg) -> DictConfig:
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    enc, viz = get_encoder(cfg)
    pair = {"encoder": enc, "visualizer": viz}

    return OmegaConf.create(pair, flags={"allow_objects": True})

def build_decoder(cfg: Union[Dict[str, Any], DictConfig]) -> DictConfig:
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    decoder = get_decoder(cfg)
    return decoder

def build_semantic_bundle(
    use_semantic: bool,
    semantic_mode: str,
    semantic_viz: str,
    semantic_config: Optional[str],
    clip: Dict[str, Any],
    reg_model: Dict[str, Any],
):
    if not use_semantic:
        return None
    clip_model, _, _ = open_clip.create_model_and_transforms(
        clip["model_name"],
        pretrained=clip["pretrained"],
        precision=clip.get("precision", "fp16"),
    )
    clip_model = clip_model.to(clip.get("device", "cuda")).eval()
    text_encoder = clip_model.encode_text if semantic_mode == "embed" else None
    text_tokenizer = open_clip.get_tokenizer(clip["model_name"]) if semantic_mode == "embed" else None

    dino_model = None
    if reg_model.get("enabled", False):
        dino_model = build_2d_model(model_name=reg_model["name"]).to(reg_model.get("device", "cuda"))

    semantic_mapper = SemanticMapper(
        config_path=semantic_config,
        mode=semantic_mode,
        text_encoder=text_encoder,
        text_tokenizer=text_tokenizer,
        semantic_viz=semantic_viz,
    )
    ret_dict = {"clip_model": clip_model, "dino_model": dino_model, "semantic_mapper": semantic_mapper}
    return OmegaConf.create(ret_dict, flags={"allow_objects": True})

def build_diffusion(model: Any, dataset_cfg: Dict[str, Any], **kwargs):
    # probe image_size from dataset_cfg (fresh instance just for size)
    ds = instantiate(
        {"_target_": "data_io.get_dataset", "config": dataset_cfg.config},
        _convert_="none"
    )
    image_size = getattr(ds, "image_size", None)
    if image_size is None:
        raise ValueError("Dataset must expose .image_size for diffusion.")

    # unpack optional semantic bundle into model if provided
    if hasattr(model, "semantic_bundle") and model.semantic_bundle:
        bundle = model.semantic_bundle
        # if PixelSplatEpiState expects these as kwargs:
        model.clip_model = bundle.get("clip_model")
        model.dino_model = bundle.get("dino_model")
        model.semantic_mapper = bundle.get("semantic_mapper")

    return GaussianDiffusionPixelEpiTemporal(
        model=model,
        image_size=image_size,
        timesteps=kwargs.get("timesteps", 1000),
        sampling_timesteps=kwargs.get("sampling_steps", 10),
        loss_type=kwargs.get("loss_type", "l2"),
        objective=kwargs.get("objective", "pred_x0"),
        beta_schedule=kwargs.get("beta_schedule", "cosine"),
        guidance_scale=kwargs.get("guidance_scale", 0.0),
        temperature=kwargs.get("temperature", 0.5),
        clean_target=kwargs.get("clean_target", False),
        use_semantic=getattr(model, "use_semantic", False),
        use_reg_model=getattr(model, "dino_model", None) is not None,
    ).cuda()