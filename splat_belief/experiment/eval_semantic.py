"""Evaluate the semantic head + dense reg head on a few val batches.

Outputs, per sample target view (image_size x image_size):
  - GT trgt RGB
  - rendered RGB
  - one query-similarity heatmap per label in `eval_query_labels`
  - PCA-3 of rendered semantic_reg_dense (the head's prediction)
  - PCA-3 of dino_dense GT (DINOv3 features on the GT trgt RGB)

Run with the same Hydra overrides as finetune_semantic.sh (encoder dims, dataset,
checkpoint path, etc). Extra knobs:
  +eval_query_labels='[bed, sofa, chair, table, tv, kitchen counter, wall, floor]'
  +eval_num_samples=8
"""
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_io  # noqa: E402
import open_clip  # noqa: E402
from accelerate import Accelerator, DistributedDataParallelKwargs  # noqa: E402

from splat import SplatBelief  # noqa: E402
from splat_belief.diffusion.diffusion import Diffusion  # noqa: E402
from splat_belief.embodied.semantic_mapper import SemanticMapper  # noqa: E402
from splat_belief.experiment.train import get_train_settings  # noqa: E402
from splat_belief.utils.vision_utils import to_gpu  # noqa: E402
from splat_belief.splat.decoder import get_decoder  # noqa: E402
from splat_belief.splat.encoder import get_encoder  # noqa: E402
from splat_belief.utils.model_utils import build_2d_model, load_repa_encoder  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _pca_to_rgb(features: torch.Tensor) -> np.ndarray:
    """features: (C, H, W) float -> (H, W, 3) uint8 from top-3 PCA components."""
    c, h, w = features.shape
    flat = features.reshape(c, -1).T.float().cpu().numpy()
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(flat, full_matrices=False)
    pcs = (flat @ vh[:3].T).reshape(h, w, 3)
    pcs = (pcs - pcs.min(axis=(0, 1), keepdims=True)) / (
        pcs.max(axis=(0, 1), keepdims=True) - pcs.min(axis=(0, 1), keepdims=True) + 1e-8
    )
    return (pcs * 255).astype(np.uint8)


def _query_heatmap(sem_dense: torch.Tensor, label_emb: torch.Tensor) -> np.ndarray:
    """sem_dense: (C, H, W); label_emb: (C,) — both unnormalized.
    Returns (H, W, 3) uint8 jet heatmap of cosine similarity."""
    sem = sem_dense.float()
    sem = sem / (sem.norm(dim=0, keepdim=True) + 1e-8)
    le = label_emb.float()
    le = le / (le.norm() + 1e-8)
    sims = (sem * le[:, None, None]).sum(dim=0).cpu().numpy()
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
    return (plt.cm.jet(sims)[..., :3] * 255).astype(np.uint8)


def _save_panel(
    save_path: Path,
    gt_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    sem_dense: torch.Tensor,
    sem_reg: torch.Tensor | None,
    dino_gt: torch.Tensor | None,
    label_embs: torch.Tensor,
    labels: list,
):
    n_q = len(labels)
    cols = 2 + n_q + (2 if sem_reg is not None and dino_gt is not None else 0)
    fig, axes = plt.subplots(1, cols, figsize=(2.5 * cols, 2.5))
    col = 0
    axes[col].imshow(gt_rgb)
    axes[col].set_title("GT trgt")
    axes[col].axis("off")
    col += 1
    axes[col].imshow(rendered_rgb)
    axes[col].set_title("rendered")
    axes[col].axis("off")
    col += 1
    for li, lab in enumerate(labels):
        axes[col].imshow(_query_heatmap(sem_dense, label_embs[li]))
        axes[col].set_title(f"q: {lab}", fontsize=9)
        axes[col].axis("off")
        col += 1
    if sem_reg is not None:
        axes[col].imshow(_pca_to_rgb(sem_reg))
        axes[col].set_title("PCA(reg pred)", fontsize=9)
        axes[col].axis("off")
        col += 1
    if dino_gt is not None:
        axes[col].imshow(_pca_to_rgb(dino_gt))
        axes[col].set_title("PCA(DINO GT)", fontsize=9)
        axes[col].axis("off")
        col += 1
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="../config/", config_name="config")
def evaluate(cfg: DictConfig):
    train_settings = get_train_settings(cfg.setting_name, cfg.ngpus)
    cfg.num_context = train_settings["num_context"]
    cfg.num_target = train_settings["num_target"]

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        split_batches=True, mixed_precision="no", kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device

    dataset = data_io.get_dataset(cfg)
    dl = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)

    use_semantic = cfg.model.encoder.use_semantic
    use_reg_model = cfg.model.encoder.use_reg_model
    use_depth_mask = cfg.model.encoder.use_depth_mask
    semantic_config = cfg.semantic_config
    repa_encoder_resolution = cfg.repa_encoder_resolution

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp16"
    )
    clip_model = clip_model.to(device).eval()
    text_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    dino_model = None
    if use_semantic and use_reg_model:
        dino_model = build_2d_model(
            model_name=cfg.model.encoder.reg_model_name,
            weights_path=cfg.model.encoder.get("reg_model_weights", None),
        ).to(device).eval()

    repa_encoder_name = cfg.repa_encoder_name
    repa_encoder = None
    if cfg.model.encoder.backbone.use_repa:
        repa_encoder, _, _ = load_repa_encoder(
            enc_name=repa_encoder_name,
            device=device,
            resolution=repa_encoder_resolution,
            weights_path=cfg.get("repa_encoder_weights", None),
        )

    semantic_mapper = SemanticMapper(
        config_path=semantic_config,
        mode="embed",
        text_encoder=clip_model.encode_text,
        text_tokenizer=text_tokenizer,
        semantic_viz="query",
    )

    # Load class-text embedding table when the checkpoint was trained with
    # class_text_only supervision.
    semantic_supervision_mode = cfg.get("semantic_supervision_mode", "clip_patch_sampled")
    class_text_table = None
    if use_semantic and semantic_supervision_mode in ("class_text_only", "class_text_image"):
        tt = cfg.get("class_text_table_path", None)
        if not tt:
            raise ValueError(
                f"semantic_supervision_mode={semantic_supervision_mode!r} requires class_text_table_path to be set."
            )
        payload = torch.load(tt, map_location="cpu", weights_only=False)
        class_text_table = payload["embeddings"]

    model = SplatBelief(
        encoder, encoder_visualizer, decoder,
        semantic_mapper=semantic_mapper,
        depth_mode=cfg.depth_mode,
        extended_visualization=cfg.extended_visualization,
        use_semantic=use_semantic,
        semantic_mode="embed",
        inference_mode=cfg.model.encoder.inference_mode,
        use_depth_mask=use_depth_mask,
        clip_model=clip_model,
        dino_model=dino_model,
        repa_encoder=repa_encoder,
        repa_encoder_name=repa_encoder_name,
        repa_encoder_resolution=repa_encoder_resolution,
        semantic_supervision_mode=semantic_supervision_mode,
        class_text_table=class_text_table,
    ).to(device)

    diffusion = Diffusion(
        model,
        image_size=dataset.image_size,
        timesteps=1000,
        sampling_timesteps=10,
        loss_type="l2",
        objective="pred_x0",
        beta_schedule="cosine",
        use_guidance=cfg.use_guidance,
        guidance_scale=1.0,
        clean_target=cfg.clean_target,
        use_depth_supervision=cfg.use_depth_supervision,
        use_semantic=use_semantic,
        use_reg_model=use_reg_model,
        use_depth_mask=use_depth_mask,
        cfg=cfg,
        semantic_supervision_mode=semantic_supervision_mode,
    ).to(device)

    ckpt_path = cfg.checkpoint_path
    print(f"loading checkpoint: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location="cpu")
    if "ema" in state:
        ema_raw = state["ema"]
        # EMA state has both ema_model.* and online_model.* prefixes;
        # take ema_model.* and strip prefix.
        loadable = {}
        for k, v in ema_raw.items():
            if not k.startswith("ema_model."):
                continue
            kk = k[len("ema_model."):]
            if any(s in kk for s in ("repa_encoder", "dino_model", "clip_model")):
                continue
            loadable[kk] = v
        msg = diffusion.load_state_dict(loadable, strict=False)
        print(f"EMA: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    else:
        msg = diffusion.load_state_dict(state["model"], strict=False)
        print(f"model: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    diffusion.eval()
    for p in diffusion.parameters():
        p.requires_grad = False

    # Diffusion.p_losses returns (losses, viz_tuple), discarding the inner
    # misc dict from SplatBelief.forward. Capture it via a forward hook so we
    # can read trgt_semantic / dino_dense / semantic_dense.
    captured = {}

    def _capture_misc(module, inputs, output):
        captured["misc"] = output[2]

    diffusion.model.register_forward_hook(_capture_misc)

    # Resolve query labels
    eval_query_labels = list(
        cfg.get(
            "eval_query_labels",
            ["bed", "sofa", "chair", "table", "tv", "kitchen counter", "wall", "floor"],
        )
    )
    n_samples = int(cfg.get("eval_num_samples", 8))

    # The trained head lives in the same text-embedding space the supervision
    # used. For class_text_only mode that's the CLIP-text encoder applied to
    # bare class names (matching scripts/training/precompute_clip_text_targets.py).
    # In all cases we encode queries with CLIP-text — same encoder + bare-name
    # template — so head outputs and queries share a space.
    class_text_table_lookup = cfg.get("eval_class_text_table_path", None)
    if class_text_table_lookup:
        # Lookup precomputed embeddings by exact (case-insensitive) name match.
        print(f"loading class text table: {class_text_table_lookup}")
        table = torch.load(str(class_text_table_lookup), map_location="cpu", weights_only=False)
        n2id_lower = {k.lower(): v for k, v in table["name_to_id"].items()}
        embs = table["embeddings"].float()
        kept_labels = []
        kept_vecs = []
        for lab in eval_query_labels:
            cid = n2id_lower.get(lab.lower())
            if cid is None:
                print(f"  WARNING: label '{lab}' not in class text table; dropping")
                continue
            kept_labels.append(lab)
            kept_vecs.append(embs[cid])
        if not kept_vecs:
            raise RuntimeError(
                "no eval_query_labels matched the class text table; "
                f"available examples: {list(table['name_to_id'].keys())[:10]}"
            )
        eval_query_labels = kept_labels
        label_embs = torch.stack(kept_vecs, dim=0).to(device)
    else:
        # On-the-fly: encode queries with the same CLIP-text encoder used at
        # precompute time, with the bare "{name}" template (no "a photo of a").
        with torch.no_grad():
            toks = text_tokenizer(eval_query_labels).to(device)
            label_embs = clip_model.encode_text(toks).float()  # (L, D_clip)

    out_dir = Path(cfg.results_folder) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing eval panels to {out_dir}")

    sample_idx = 0
    it = iter(dl)
    while sample_idx < n_samples:
        try:
            data_full = next(it)
        except StopIteration:
            break
        if isinstance(data_full, (list, tuple)):
            data = data_full[0]
        else:
            data = data_full
        data = to_gpu(data, device)

        with torch.no_grad():
            _ = diffusion(data, global_step=int(1e9))
        misc = captured.get("misc", None)
        if misc is None:
            print("forward hook did not capture misc; skipping batch")
            continue

        ts = misc.get("trgt_semantic", None)
        if ts is None or "semantic_dense" not in ts:
            print("no trgt_semantic.semantic_dense found; skipping batch")
            continue

        sem_dense = ts["semantic_dense"]                  # (B, V, C_sem, H, W)
        sem_reg = ts.get("semantic_reg_dense", None)
        dino_gt = ts.get("dino_dense", None)
        rendered_rgb = misc["rendered_trgt_rgb"]          # (B, V, 3, H, W) in [0,1]
        gt_trgt_rgb = data["trgt_rgb"]                    # (B, V, 3, H, W) normalized [-1, 1]
        gt_trgt_rgb = (gt_trgt_rgb + 1.0) / 2.0           # back to [0,1]

        b_size, n_views = sem_dense.shape[:2]
        for b in range(b_size):
            for v in range(n_views):
                if sample_idx >= n_samples:
                    break
                rendered_np = (
                    rendered_rgb[b, v].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                gt_np = (
                    gt_trgt_rgb[b, v].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                _save_panel(
                    out_dir / f"sample_{sample_idx:03d}.png",
                    gt_np, rendered_np,
                    sem_dense[b, v],
                    sem_reg[b, v] if sem_reg is not None else None,
                    dino_gt[b, v] if dino_gt is not None else None,
                    label_embs.detach(), eval_query_labels,
                )
                sample_idx += 1

    print(f"done. wrote {sample_idx} panels to {out_dir}")


if __name__ == "__main__":
    evaluate()
