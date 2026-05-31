"""Side-by-side A/B comparison of diffusion samplers on a fixed scene set.

Builds the model once and runs each scene twice (or more) — once per sampler
configuration in `cfg.compare_samplers`. Per-scene timings (model.sample wall
clock) are written to `<run_dir>/<sampler_label>/visuals_*/timing.json`, and a
top-level summary is written to `<run_dir>/comparison_summary.json`.

Default comparison: ddim @ 50 steps  vs  dpm_solver_pp @ 15 steps. Pick the set
of scenes via `splat_belief/config/inference/temporal_indices.py`.

Usage: see scripts/inference/compare_samplers.sh.
"""
import copy
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import hydra
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_io  # noqa: E402
from config.inference import temporal_indices  # noqa: E402
from splat import SplatBeliefState  # noqa: E402
from splat_belief.diffusion.diffusion_temporal import (  # noqa: E402
    DiffusionTemporal,
    Trainer,
)
from splat_belief.embodied.semantic_mapper import SemanticMapper  # noqa: E402
from splat_belief.experiment.temporal_inference import (  # noqa: E402
    create_all_timesteps_visualization,
    create_timestep_visualization,
    prepare_video_viz,
    sample_video_idx_from_dataset,
)
from splat_belief.splat.decoder import get_decoder  # noqa: E402
from splat_belief.splat.encoder import get_encoder  # noqa: E402
from splat_belief.splat.gaussian_refiner import GaussianRefinerCfg  # noqa: E402
from splat_belief.splat.ply_export import export_gaussians_to_ply  # noqa: E402
from splat_belief.utils.model_utils import build_2d_model  # noqa: E402
from splat_belief.utils.vision_utils import (  # noqa: E402,F401
    normalize_to_neg_one_to_one,
    rotation_angle,
    to_gpu,
)


def _set_sampler(diffusion_obj, sampler_name: str, sampling_steps: int):
    """Mutate a DiffusionTemporal in-place to use the given sampler/steps."""
    diffusion_obj.sampler = sampler_name
    diffusion_obj.sampling_timesteps = int(sampling_steps)
    diffusion_obj.is_ddim_sampling = sampling_steps < diffusion_obj.num_timesteps


def _make_autocast_ctx(dtype_str: str):
    """Return a CUDA autocast context for the requested dtype string."""
    if dtype_str in (None, "", "fp32", "float32"):
        return nullcontext()
    if dtype_str in ("bf16", "bfloat16"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype_str in ("fp16", "float16", "half"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unknown dtype={dtype_str!r}; expected fp32 / bf16 / fp16.")


def _run_single_scene(
    trainer,
    cfg,
    *,
    scene_descriptor,
    save_root,
    use_depth_mask,
    inference_save_scene,
    adjacent_angle,
    adjacent_distance,
    dtype_str: str = "fp32",
):
    """Run the autoregressive imagination rollout for a single scene.
    Returns dict with elapsed_sec + frames_written counts.

    `save_root` is the per-sampler results directory; this function further
    creates a `visuals_<video>_<start>_<end>_<seed>` subfolder under it.
    """
    normalize = normalize_to_neg_one_to_one
    dataset = trainer.dataloader.dataset
    seed = cfg.seed
    if seed == -1:
        seed = (hash((scene_descriptor, 42)) % (2 ** 32)) + (time.time_ns() % (2 ** 32))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    video_idx = scene_descriptor[0]
    if len(scene_descriptor) == 2:
        num_frames = scene_descriptor[1]
        data = dataset.data_for_temporal(video_idx=video_idx, frames_render=num_frames)
    elif len(scene_descriptor) == 3:
        start_end = [scene_descriptor[1], scene_descriptor[2]]
        data = dataset.data_for_temporal(video_idx=video_idx, frames_render=start_end)
        num_frames = scene_descriptor[2] - scene_descriptor[1] + 1
    else:
        raise ValueError("scene descriptor must be (video_idx, n_frames) or (video_idx, start, end)")

    rgb_frames = data[1]
    rgbs = [
        (frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        for frame in rgb_frames
    ]
    start_idx = data[2]
    end_idx = data[3]

    save_folder_sample = os.path.join(
        save_root, f"visuals_{video_idx}_{start_idx}_{end_idx}_{seed}"
    )
    os.makedirs(save_folder_sample, exist_ok=True)
    save_folder_gt_frames = os.path.join(save_folder_sample, "GT_frames")
    os.makedirs(save_folder_gt_frames, exist_ok=True)
    imageio.mimwrite(
        os.path.join(save_folder_sample, "GT_exploration.mp4"),
        rgbs, fps=10, quality=10,
    )
    for p, frame in enumerate(rgbs):
        Image.fromarray(frame).save(os.path.join(save_folder_gt_frames, f"GT_{p}.png"))

    video_dict = data[0]
    data_rgbs = video_dict["rgbs"]
    render_poses = video_dict["render_poses"]
    abs_camera_poses = video_dict["abs_camera_poses"]
    intrinsics = video_dict["intrinsics"]
    image_shape = video_dict["image_shape"]
    near = video_dict["near"]
    far = video_dict["far"]
    lang = video_dict["lang"]
    assert len(render_poses) == num_frames

    # Compute key-frame indices based on rotation/translation deltas.
    key_frame_indices = []
    z_start = render_poses[0][0][:, 2][:3]
    t_start = render_poses[0][0][:, 3][:3]
    z_previous = z_start
    t_previous = t_start
    for idx in range(1, num_frames):
        current_pose = render_poses[idx][0]
        z_idx = current_pose[:, 2][:3]
        t_idx = current_pose[:, 3][:3]
        angle = rotation_angle(z_previous, z_idx)
        distance = torch.norm(t_idx - t_previous)
        if angle > adjacent_angle or distance > adjacent_distance or idx == num_frames - 1:
            key_frame_indices.append(idx)
            z_previous = z_idx
            t_previous = t_idx

    trainer.model.model.reset_timestep()
    trainer.ema.ema_model.model.reset_timestep()
    previous_t = 0
    inp = {}
    inp["ctxt_c2w"] = torch.cat(render_poses[:1], dim=0)
    inp["ctxt_rgb"] = torch.cat(data_rgbs[:1], dim=0)
    inp["ctxt_abs_camera_poses"] = torch.cat(abs_camera_poses[:1], dim=0)

    all_generated_frames_history = []
    observed_keyframes = []

    # ---- TIMED BODY ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rollout_start = time.time()

    imagine_model = None
    for state_t, update_t in enumerate(key_frame_indices):
        inp["trgt_c2w"] = render_poses[update_t]
        inp["trgt_rgb"] = data_rgbs[update_t]
        inp["trgt_abs_camera_poses"] = abs_camera_poses[update_t]
        inp["intrinsics"] = intrinsics[0]
        inp["image_shape"] = image_shape
        inp["render_poses"] = torch.cat(render_poses, dim=0)
        inp["near"] = torch.tensor(near)
        inp["far"] = torch.tensor(far)
        inp["lang"] = lang

        inp = to_gpu(inp, "cuda")
        for k in inp.keys():
            if not k == "num_frames_render":
                inp[k] = inp[k].unsqueeze(0)
        inp["num_frames_render"] = num_frames

        with _make_autocast_ctx(dtype_str):
            out = trainer.ema.ema_model.sample(batch_size=1, inp=inp, state_t=state_t)

        if not state_t == len(key_frame_indices) - 1:
            imagine_input = copy.deepcopy(inp)
            imagine_input.update({
                "ctxt_c2w": torch.cat(render_poses[update_t:update_t + 1], dim=0),
                "ctxt_rgb": normalize(out["images"]),
                "ctxt_abs_camera_poses": torch.cat(abs_camera_poses[update_t:update_t + 1], dim=0),
            })
            imagine_model = copy.deepcopy(trainer.ema.ema_model)
            if hasattr(imagine_model.model, "refiner") and imagine_model.model.refiner is not None:
                imagine_model.model.refiner.cfg = GaussianRefinerCfg(enabled=False)
            for imagine_t in range(state_t + 1, len(key_frame_indices)):
                imagine_input["trgt_c2w"] = render_poses[key_frame_indices[imagine_t]]
                imagine_input["trgt_rgb"] = data_rgbs[key_frame_indices[imagine_t]]
                imagine_input["trgt_abs_camera_poses"] = abs_camera_poses[key_frame_indices[imagine_t]]
                imagine_input["intrinsics"] = intrinsics[0]
                imagine_input["image_shape"] = image_shape
                imagine_input["render_poses"] = torch.cat(render_poses, dim=0)
                imagine_input["near"] = torch.tensor(near)
                imagine_input["far"] = torch.tensor(far)
                imagine_input["lang"] = lang
                if not imagine_t == len(key_frame_indices) - 1:
                    imagine_input.pop("render_poses")
                imagine_input = to_gpu(imagine_input, "cuda")
                for k in imagine_input.keys():
                    if not k == "num_frames_render":
                        imagine_input[k] = imagine_input[k].unsqueeze(0)
                inp["num_frames_render"] = num_frames
                with _make_autocast_ctx(dtype_str):
                    out = imagine_model.sample(batch_size=1, inp=imagine_input, state_t=imagine_t)
                if not imagine_t == len(key_frame_indices) - 1:
                    imagine_input["ctxt_c2w"] = render_poses[key_frame_indices[imagine_t]]
                    imagine_input["ctxt_rgb"] = normalize(out["images"])
                    imagine_input["ctxt_abs_camera_poses"] = abs_camera_poses[key_frame_indices[imagine_t]]

        if use_depth_mask:
            frames, depth_frames, semantics, depth_masks = prepare_video_viz(out)
        else:
            frames, depth_frames, semantics = prepare_video_viz(out)

        all_generated_frames_history.append(frames)
        observed_keyframes.append(update_t)

        save_folder_step = os.path.join(save_folder_sample, f"step_{previous_t}")
        os.makedirs(save_folder_step, exist_ok=True)
        save_folder_rendered_frames = os.path.join(save_folder_step, "frames")
        os.makedirs(save_folder_rendered_frames, exist_ok=True)
        for p, (gt_frame, denoised_frame) in enumerate(zip(rgbs, frames)):
            Image.fromarray(denoised_frame).save(
                os.path.join(save_folder_rendered_frames, f"rendered_{p}.png")
            )
        imageio.mimwrite(
            os.path.join(save_folder_step, "denoised_view_circle.mp4"),
            frames, fps=10, quality=10,
        )
        imageio.mimwrite(
            os.path.join(save_folder_step, "denoised_view_circle_depth.mp4"),
            depth_frames, fps=10, quality=10,
        )
        if use_depth_mask:
            imageio.mimwrite(
                os.path.join(save_folder_step, "denoised_view_circle_depth_mask.mp4"),
                depth_masks, fps=10, quality=10,
            )
        imageio.mimwrite(
            os.path.join(save_folder_step, "denoised_view_circle_semantic.mp4"),
            semantics, fps=10, quality=10,
        )
        # GT vs rendered concat
        output_writer = imageio.get_writer(
            os.path.join(save_folder_step, "concatenated_video.mp4"), fps=10, quality=10
        )
        for gt_frame, denoised_frame in zip(rgbs, frames):
            output_writer.append_data(np.vstack((gt_frame, denoised_frame)))
        output_writer.close()

        # PLY export only for the final timestep, mirroring temporal_inference.
        ply_path = Path(f"{save_folder_step}/scene_{video_idx}_{start_idx}_{end_idx}.ply")
        vis_model = imagine_model if not state_t == len(key_frame_indices) - 1 else trainer.ema.ema_model
        gaussians = vis_model.model.augmented_gaussians.float()
        if inference_save_scene:
            export_gaussians_to_ply(gaussians, render_poses[0].to("cuda"), ply_path)

        create_timestep_visualization(
            rgbs=rgbs, frames=frames, key_frame_indices=key_frame_indices,
            previous_t=previous_t, update_t=update_t,
            save_folder_step=save_folder_step, state_t=state_t,
        )
        create_all_timesteps_visualization(
            rgbs=rgbs, frames_history=all_generated_frames_history,
            key_frame_indices=key_frame_indices,
            observed_keyframes=observed_keyframes,
            save_folder_sample=save_folder_step,
            current_state_t=state_t,
        )

        previous_t = update_t
        inp["ctxt_c2w"] = torch.cat(render_poses[previous_t:previous_t + 1], dim=0)
        inp["ctxt_rgb"] = torch.cat(data_rgbs[previous_t:previous_t + 1], dim=0)
        inp["ctxt_abs_camera_poses"] = torch.cat(abs_camera_poses[previous_t:previous_t + 1], dim=0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - rollout_start

    return {
        "video_idx": int(video_idx),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "n_keyframes": len(key_frame_indices),
        "elapsed_sec": elapsed,
        "save_folder": save_folder_sample,
    }


def _build_model_and_trainer(cfg: DictConfig):
    """Mirrors temporal_inference.train()'s setup; returns (trainer, dataset)."""
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        split_batches=True, mixed_precision="no", kwargs_handlers=[ddp_kwargs]
    )

    train_batch_size = cfg.batch_size
    dataset = data_io.get_dataset(cfg)
    dl = DataLoader(
        dataset, batch_size=train_batch_size, shuffle=False, pin_memory=True, num_workers=0
    )

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    use_semantic = cfg.model.encoder.use_semantic
    use_reg_model = cfg.model.encoder.use_reg_model
    reg_model_name = cfg.model.encoder.reg_model_name
    use_depth_mask = cfg.model.encoder.use_depth_mask
    semantic_config = cfg.semantic_config
    semantic_mode = cfg.semantic_mode
    semantic_viz = cfg.semantic_viz

    clip_model = None
    if use_semantic:
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp16"
        )
        clip_model = clip_model.to("cuda").eval()

    dino_model = None
    if use_semantic and use_reg_model:
        dino_model = build_2d_model(
            model_name=reg_model_name,
            weights_path=cfg.model.encoder.get("reg_model_weights", None),
        )

    text_encoder = None
    text_tokenizer = None
    if use_semantic and semantic_mode == "embed":
        text_encoder = clip_model.encode_text
        text_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    semantic_mapper = None
    if use_semantic:
        semantic_mapper = SemanticMapper(
            config_path=semantic_config, mode=semantic_mode,
            text_encoder=text_encoder, text_tokenizer=text_tokenizer,
            semantic_viz=semantic_viz,
        )

    refiner_cfg_dict = getattr(cfg, "refiner", {})
    if isinstance(refiner_cfg_dict, dict):
        refiner_cfg = GaussianRefinerCfg(**refiner_cfg_dict) if refiner_cfg_dict else GaussianRefinerCfg()
    else:
        refiner_cfg = GaussianRefinerCfg(**dict(refiner_cfg_dict))

    model = SplatBeliefState(
        encoder, encoder_visualizer, decoder,
        semantic_mapper=semantic_mapper,
        depth_mode=cfg.depth_mode,
        extended_visualization=cfg.extended_visualization,
        use_history=cfg.use_history,
        use_semantic=use_semantic,
        semantic_mode=semantic_mode,
        inference_mode=cfg.model.encoder.inference_mode,
        clip_model=clip_model,
        dino_model=dino_model,
        use_depth_mask=use_depth_mask,
        refiner_cfg=refiner_cfg,
    ).cuda()

    diffusion = DiffusionTemporal(
        model,
        image_size=dataset.image_size,
        timesteps=1000,
        sampling_timesteps=cfg.sampling_steps,
        loss_type="l2",
        objective="pred_x0",
        beta_schedule="cosine",
        use_guidance=cfg.use_guidance,
        guidance_scale=cfg.guidance_scale,
        temperature=cfg.temperature,
        clean_target=cfg.clean_target,
        use_semantic=use_semantic,
        use_reg_model=use_reg_model,
        use_depth_mask=use_depth_mask,
        sampler=cfg.get("sampler", "ddim"),
    ).cuda()

    trainer = Trainer(
        diffusion,
        dataloader=dl,
        train_batch_size=train_batch_size,
        train_lr=cfg.lr,
        train_num_steps=700000,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=False,
        sample_every=1000,
        wandb_every=50,
        save_every=5000,
        num_samples=1,
        warmup_period=1000,
        checkpoint_path=cfg.checkpoint_path,
        wandb_config=cfg.wandb,
        run_name=cfg.name,
        accelerator=accelerator,
        cfg=cfg,
        use_semantic=use_semantic,
    )
    return trainer, dataset


@hydra.main(version_base=None, config_path="../config/", config_name="config")
def main(cfg: DictConfig):
    run_dir = Path(cfg.results_folder)
    run_dir.mkdir(parents=True, exist_ok=True)
    use_depth_mask = cfg.model.encoder.use_depth_mask
    inference_save_scene = cfg.inference_save_scene
    adjacent_angle = cfg.adjacent_angle
    adjacent_distance = cfg.adjacent_distance

    trainer, dataset = _build_model_and_trainer(cfg)

    if cfg.inference_sample_from_dataset:
        interesting_indices = sample_video_idx_from_dataset(
            dataset,
            num_samples=cfg.inference_num_samples,
            min_frames=cfg.inference_min_frames,
            max_frames=cfg.inference_max_frames,
        )
    else:
        interesting_indices = temporal_indices.interesting_indices

    samplers_cfg = list(cfg.get("compare_samplers", [
        {"name": "ddim", "steps": 50, "dtype": "fp32"},
        {"name": "ddim", "steps": 50, "dtype": "bf16"},
        {"name": "dpm_solver_pp", "steps": 15, "dtype": "fp32"},
        {"name": "dpm_solver_pp", "steps": 15, "dtype": "bf16"},
    ]))
    print(
        f"comparing {len(samplers_cfg)} configs on {len(interesting_indices)} scenes "
        f"-> {len(samplers_cfg) * len(interesting_indices)} runs total"
    )

    summary = []
    for s_idx, s_cfg in enumerate(samplers_cfg):
        sampler_name = s_cfg["name"]
        sampling_steps = int(s_cfg["steps"])
        dtype_str = str(s_cfg.get("dtype", "fp32"))
        label = f"{sampler_name}_{sampling_steps}_{dtype_str}"
        sampler_dir = run_dir / label
        sampler_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== config {s_idx + 1}/{len(samplers_cfg)}: {label} ===")

        # Apply sampler config to BOTH the online model and the EMA model.
        for d in (trainer.model, trainer.ema.ema_model):
            _set_sampler(d, sampler_name, sampling_steps)

        for scene_idx, scene_descriptor in enumerate(interesting_indices):
            print(f"  scene {scene_idx + 1}/{len(interesting_indices)}: {scene_descriptor}")
            with torch.no_grad():
                result = _run_single_scene(
                    trainer, cfg,
                    scene_descriptor=scene_descriptor,
                    save_root=str(sampler_dir),
                    use_depth_mask=use_depth_mask,
                    inference_save_scene=inference_save_scene,
                    adjacent_angle=adjacent_angle,
                    adjacent_distance=adjacent_distance,
                    dtype_str=dtype_str,
                )
            entry = {
                "sampler": sampler_name,
                "sampling_steps": sampling_steps,
                "dtype": dtype_str,
                "scene_descriptor": list(scene_descriptor),
                **result,
            }
            print(
                f"    -> {entry['elapsed_sec']:.2f}s ({entry['n_keyframes']} keyframes), "
                f"saved to {entry['save_folder']}"
            )

            # Per-scene timing.json next to the visuals folder.
            timing_path = os.path.join(entry["save_folder"], "timing.json")
            with open(timing_path, "w") as f:
                json.dump(entry, f, indent=2)
            summary.append(entry)

    # Top-level summary + a nicely formatted comparison table.
    summary_path = run_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    table_path = run_dir / "comparison_summary.txt"
    with open(table_path, "w") as f:
        f.write(
            f"{'sampler':<16} {'steps':<6} {'dtype':<6} {'scene':<24} "
            f"{'keyframes':<10} {'time_s':<10}\n"
        )
        f.write("-" * 80 + "\n")
        for e in summary:
            f.write(
                f"{e['sampler']:<16} {e['sampling_steps']:<6d} "
                f"{e.get('dtype', 'fp32'):<6} "
                f"{str(e['scene_descriptor']):<24} {e['n_keyframes']:<10d} "
                f"{e['elapsed_sec']:<10.2f}\n"
            )

        # Per-scene speedup matrix vs the first config (typically ddim/fp32).
        if len(samplers_cfg) >= 2:
            ref = samplers_cfg[0]
            ref_label = f"{ref['name']}_{int(ref['steps'])}_{ref.get('dtype', 'fp32')}"
            f.write(f"\nSpeedup vs {ref_label}:\n")
            unique_scenes = sorted({tuple(e["scene_descriptor"]) for e in summary})
            for sc in unique_scenes:
                f.write(f"  scene {list(sc)}:\n")
                base_t = next(
                    (e["elapsed_sec"] for e in summary
                     if tuple(e["scene_descriptor"]) == sc
                     and e["sampler"] == ref["name"]
                     and e["sampling_steps"] == int(ref["steps"])
                     and e.get("dtype", "fp32") == ref.get("dtype", "fp32")),
                    None,
                )
                if base_t is None:
                    continue
                for s_cfg in samplers_cfg:
                    label = f"{s_cfg['name']}_{int(s_cfg['steps'])}_{s_cfg.get('dtype', 'fp32')}"
                    t = next(
                        (e["elapsed_sec"] for e in summary
                         if tuple(e["scene_descriptor"]) == sc
                         and e["sampler"] == s_cfg["name"]
                         and e["sampling_steps"] == int(s_cfg["steps"])
                         and e.get("dtype", "fp32") == s_cfg.get("dtype", "fp32")),
                        None,
                    )
                    if t is None:
                        continue
                    speedup = base_t / max(t, 1e-9)
                    f.write(f"    {label}: {t:.2f}s  ({speedup:.2f}x)\n")
    print(f"\nWrote {summary_path} and {table_path}")


if __name__ == "__main__":
    main()
