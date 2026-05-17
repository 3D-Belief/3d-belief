#!/usr/bin/env python
"""Run RE10K temporal vision predictions for 3D-Belief, DFoT, and Gen3C."""
from __future__ import annotations

import argparse
import gc
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    DEFAULT_OUTPUT_ROOT,
    EpisodeSpec,
    add_project_paths,
    compute_key_frame_indices,
    env_snapshot,
    frame_sets_for_episode,
    parse_episode_token,
    patch_numpy_legacy_aliases,
    pose_array,
    require_cuda,
    resize_uint8,
    save_indexed_frames,
    save_video,
    tensor_to_uint8_image,
    triplet_is_monotonic,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RE10K_ROOT = Path("/home/ubuntu/tianmin-neurips/datasets")
DEFAULT_BELIEF_CKPT = Path("/home/ubuntu/tianmin-neurips/yyin34/codebase/3d-belief/checkpoints/model-44.pt")
DEFAULT_DFOT_CKPT = Path("/home/ubuntu/tianmin-neurips/yyin34/codebase/3d-belief/checkpoints/DFoT_RE10K.ckpt")
DEFAULT_DFOT_REPO = REPO_ROOT / "third_party" / "dfot"

# Match 3D-Belief RE10K training thresholds.
DEFAULT_ADJACENT_ANGLE = 0.523
DEFAULT_ADJACENT_DISTANCE = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vision metrics on RE10K for 3D-Belief, DFoT, and Gen3C."
    )
    parser.add_argument("--models", default="3d_belief,dfot,gen3c")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_RE10K_ROOT,
                        help="Directory containing <stage>/ frame folders and <stage>.mat poses.")
    parser.add_argument("--stage", default="test", choices=("test", "train", "unit"))
    parser.add_argument("--image-size", type=int, default=128,
                        help="3D-Belief input image size (must match training; train_re10k_unfrozen.sh uses 128).")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Cap RE10K scenes loaded (faster startup).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--episode", action="append", default=[],
                        help="Override episode: scene,start,kf1,kf2 (repeatable).")
    parser.add_argument("--episode-file", type=Path, default=None,
                        help="Text file with one scene,start,kf1,kf2 episode token per line.")
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--scan-max-frames", type=int, default=60)
    parser.add_argument("--max-start-candidates", type=int, default=8)
    parser.add_argument("--one-episode-per-scene", action="store_true",
                        help="Accept at most one episode per scene; move to the next scene as "
                             "soon as a viable episode is found.")
    parser.add_argument("--adjacent-angle", type=float, default=DEFAULT_ADJACENT_ANGLE)
    parser.add_argument("--adjacent-distance", type=float, default=DEFAULT_ADJACENT_DISTANCE)
    parser.add_argument("--allow-nonmonotonic", action="store_true")

    parser.add_argument("--belief-checkpoint", type=Path, default=DEFAULT_BELIEF_CKPT)
    parser.add_argument(
        "--belief-config-profile",
        default="re10k_128_vggt",
        choices=("re10k_128_vggt", "re10k_256_from_128"),
        help="3D-Belief architecture/training override profile to use for inference.",
    )
    parser.add_argument("--belief-sampling-steps", type=int, default=50)
    parser.add_argument("--belief-sampler", default="ddim")
    parser.add_argument("--belief-temperature", type=float, default=0.85)
    parser.add_argument("--belief-inference-dtype", default="fp32",
                        choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--belief-obj-permanence-mode", default="none",
                        choices=("none", "opacity", "dps"),
                        help="Object permanence guidance mode for 3D-Belief inference.")
    parser.add_argument("--belief-obj-permanence-state-t-min", type=int, default=0,
                        help="Minimum temporal state_t where object permanence guidance activates.")
    parser.add_argument("--belief-obj-permanence-mask-blur", type=int, default=0)
    parser.add_argument("--belief-obj-permanence-mask-threshold", type=float, default=0.5)
    parser.add_argument("--belief-obj-permanence-erode-kernel", type=int, default=21)
    parser.add_argument(
        "--belief-obj-permanence-mask-binarize",
        action="store_true",
        help="Binarize the history coverage mask after optional blur using the mask threshold.",
    )
    parser.add_argument("--belief-dps-guidance-scale", type=float, default=1.0)
    parser.add_argument("--belief-dps-pos-weight", type=float, default=1.0)
    parser.add_argument("--belief-dps-opacity-weight", type=float, default=0.5)
    parser.add_argument("--belief-refiner-enabled",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--belief-refiner-num-iterations", type=int, default=None)
    parser.add_argument("--belief-refiner-prior-weight", type=float, default=None)
    parser.add_argument("--belief-refiner-depth-consistency-weight", type=float, default=None)
    parser.add_argument("--belief-refiner-position-update-mode",
                        choices=("free", "ray", "ray_tangent"), default=None)
    parser.add_argument("--belief-refiner-ray-tangent-weight", type=float, default=None)
    parser.add_argument("--belief-refiner-ray-min-depth", type=float, default=None)
    parser.add_argument(
        "--belief-save-gaussians",
        action="store_true",
        help="Export 3D-Belief Gaussian PLY assets for each prediction split.",
    )
    parser.add_argument(
        "--belief-gaussian-layers",
        default="full",
        help="Comma-separated Gaussian layers to export: full,history,belief,incremental.",
    )
    parser.add_argument(
        "--belief-gaussian-opacity-thresh",
        type=float,
        default=0.0,
        help="Viewer-export only: drop Gaussians with opacity below this threshold.",
    )
    parser.add_argument(
        "--belief-gaussian-max-scale",
        type=float,
        default=0.0,
        help="Viewer-export only: drop Gaussians whose largest covariance stddev exceeds this value. <=0 disables.",
    )
    parser.add_argument(
        "--belief-gaussian-max-count",
        type=int,
        default=0,
        help="Viewer-export only: keep at most this many Gaussians by opacity after other filters. <=0 disables.",
    )
    parser.add_argument(
        "--belief-gaussian-voxel-size",
        type=float,
        default=0.0,
        help="Viewer-export only: voxel size for a cheap isolation filter. <=0 disables.",
    )
    parser.add_argument(
        "--belief-gaussian-min-voxel-count",
        type=int,
        default=1,
        help="Viewer-export only: with --belief-gaussian-voxel-size, keep voxels containing at least this many Gaussians.",
    )

    parser.add_argument("--dfot-checkpoint", type=Path, default=DEFAULT_DFOT_CKPT)
    parser.add_argument("--dfot-repo", type=Path, default=DEFAULT_DFOT_REPO)
    parser.add_argument("--dfot-resolution", type=int, default=256)
    parser.add_argument("--dfot-include-kf1-to-kf2",
                        action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--gen3c-python",
        default=os.environ.get("GEN3C_PYTHON", "/home/ubuntu/tianmin-neurips/miniconda3/envs/cosmos-predict1/bin/python"),
    )
    parser.add_argument("--gen3c-repo", type=Path, default=Path("/home/ubuntu/tianmin-neurips/yyin34/codebase/GEN3C"))
    parser.add_argument("--gen3c-checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--gen3c-cuda-home",
        type=Path,
        default=Path(os.environ.get("GEN3C_CUDA_HOME", "/home/ubuntu/tianmin-neurips/miniconda3/envs/cosmos-predict1")),
    )
    parser.add_argument("--gen3c-height", type=int, default=704)
    parser.add_argument("--gen3c-width", type=int, default=1280)
    parser.add_argument("--gen3c-fps", type=int, default=10)
    parser.add_argument("--gen3c-seed", type=int, default=1)
    parser.add_argument("--gen3c-guidance", type=float, default=1.0)
    parser.add_argument("--gen3c-num-steps", type=int, default=35)
    parser.add_argument("--gen3c-num-gpus", type=int, default=1)
    parser.add_argument("--gen3c-strategy", default="keyframes", choices=("keyframes", "history"))
    parser.add_argument("--gen3c-filter-points-threshold", type=float, default=0.05)
    parser.add_argument("--gen3c-no-foreground-masking", action="store_true")
    parser.add_argument("--gen3c-offload", action="store_true", help="Enable Gen3C offload flags for lower-memory GPUs.")
    parser.add_argument(
        "--gen3c-enable-prompt-encoder",
        action="store_true",
        help="Use Gen3C's T5 prompt encoder. Defaults to dummy prompt embeddings for vision-only evaluation.",
    )
    parser.add_argument(
        "--gen3c-missing-depth-policy",
        default="moge",
        choices=("error", "moge", "unit"),
        help="How Gen3C should resolve missing RE10K seed depth. Use 'moge' for fair RGB-only evaluation.",
    )
    return parser.parse_args()


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def make_autocast_ctx(dtype_str: str):
    if dtype_str in (None, "", "fp32", "float32"):
        return nullcontext()
    if dtype_str in ("bf16", "bfloat16"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype_str in ("fp16", "float16", "half"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(dtype_str)


def normalize_models(raw: str) -> List[str]:
    aliases = {
        "all": ["3d_belief", "dfot", "gen3c"],
        "3d-belief": ["3d_belief"],
        "3d_belief": ["3d_belief"],
        "belief": ["3d_belief"],
        "dfot": ["dfot"],
        "gen3c": ["gen3c"],
        "gen-3c": ["gen3c"],
    }
    out: List[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in aliases:
            raise ValueError(f"Unknown model {token!r}; choose 3d_belief,dfot,gen3c,all")
        for m in aliases[token]:
            if m not in out:
                out.append(m)
    return out


def build_dataset(dataset_root: Path, stage: str, image_size: int,
                  adjacent_angle: float, adjacent_distance: float,
                  max_scenes: int | None):
    add_project_paths()
    from splat_belief.data_io.realestate import RealEstate10kDatasetOM
    from splat_belief.splat.layers import T5Encoder

    language_encoder = T5Encoder()
    dataset = RealEstate10kDatasetOM(
        root=dataset_root,
        num_context=1,
        num_target=1,
        context_min_distance=15,
        context_max_distance=16,
        stage=stage,
        image_size=image_size,
        adjacent_angle=adjacent_angle,
        adjacent_distance=adjacent_distance,
        language_encoder=language_encoder,
        max_scenes=max_scenes,
        use_depth_supervision=False,
    )
    # Sample-episodes loop expects `num_frames_per_scene`; derive from rgb files.
    dataset.num_frames_per_scene = [len(rgbs) for rgbs in dataset.all_rgb_files]
    return dataset


def candidate_starts(num_frames: int, scan_max_frames: int, max_candidates: int,
                     rng: random.Random) -> List[int]:
    if num_frames < 3:
        return []
    max_start = max(0, num_frames - 3)
    anchors = [0]
    if max_start > 0:
        usable = list(range(1, max_start + 1))
        rng.shuffle(usable)
        anchors.extend(usable[: max(0, max_candidates - 1)])
    return [start for start in anchors if start + 2 < num_frames and start < num_frames - 1]


def sample_episodes(dataset: Any, args: argparse.Namespace) -> List[EpisodeSpec]:
    episode_tokens = list(args.episode)
    if args.episode_file is not None:
        for line in args.episode_file.read_text().splitlines():
            token = line.strip()
            if token and not token.startswith("#"):
                episode_tokens.append(token)
    if episode_tokens:
        return [parse_episode_token(token) for token in episode_tokens]

    rng = random.Random(args.seed)
    scene_indices = list(range(len(dataset.scene_path_list)))
    rng.shuffle(scene_indices)
    episodes: List[EpisodeSpec] = []

    for scene_idx in scene_indices:
        num_frames = int(dataset.num_frames_per_scene[scene_idx])
        for start_idx in candidate_starts(num_frames, args.scan_max_frames,
                                          args.max_start_candidates, rng):
            end_scan = min(num_frames - 1, start_idx + args.scan_max_frames - 1)
            try:
                video_dict, _rgb, actual_start, _actual_end = dataset.data_for_temporal(
                    video_idx=scene_idx,
                    frames_render=[start_idx, end_scan],
                )
            except Exception as exc:
                print(f"[episode-select] skipping scene={scene_idx} start={start_idx}: {exc}")
                continue

            key_frame_indices = compute_key_frame_indices(
                video_dict["render_poses"],
                adjacent_angle=args.adjacent_angle,
                adjacent_distance=args.adjacent_distance,
            )
            if len(key_frame_indices) < 2:
                continue
            kf1_local, kf2_local = int(key_frame_indices[0]), int(key_frame_indices[1])
            distance_ok, angle_ok = triplet_is_monotonic(
                video_dict["render_poses"], kf1_local, kf2_local
            )
            if not args.allow_nonmonotonic and not (distance_ok and angle_ok):
                continue

            episode = EpisodeSpec(
                scene_idx=int(scene_idx),
                start_idx=int(actual_start),
                end_idx=int(actual_start + kf2_local),
                kf0_idx=int(actual_start),
                kf1_idx=int(actual_start + kf1_local),
                kf2_idx=int(actual_start + kf2_local),
                key_frame_indices=tuple(int(i) for i in key_frame_indices[:2]),
                monotonic_distance=bool(distance_ok),
                monotonic_angle=bool(angle_ok),
            )
            episodes.append(episode)
            print(f"[episode-select] {episode.name} keyframes={episode.key_frame_indices}")
            if len(episodes) >= args.num_episodes:
                return episodes
            if args.one_episode_per_scene:
                break
    return episodes


def load_episode_sample(dataset: Any, episode: EpisodeSpec) -> dict:
    video_dict, rgb_frames, start_idx, end_idx = dataset.data_for_temporal(
        video_idx=episode.scene_idx,
        frames_render=[episode.start_idx, episode.end_idx],
    )
    gt_frames = {idx: tensor_to_uint8_image(frame) for idx, frame in enumerate(rgb_frames)}
    render_poses = [pose.detach().cpu() for pose in video_dict["render_poses"]]
    abs_poses = [pose.detach().cpu() for pose in video_dict["abs_camera_poses"]]
    return {
        "episode": episode,
        "video_dict": video_dict,
        "gt_frames": gt_frames,
        "render_poses": render_poses,
        "abs_poses": abs_poses,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "height_width": next(iter(gt_frames.values())).shape[:2],
    }


def write_ground_truth(dataset: Any, episodes: Sequence[EpisodeSpec],
                       output_root: Path) -> None:
    manifest = []
    for episode in episodes:
        sample = load_episode_sample(dataset, episode)
        episode_root = output_root / "ground_truth" / episode.name
        save_indexed_frames(sample["gt_frames"], episode_root / "frames")
        save_video(episode_root / "gt.mp4",
                   [sample["gt_frames"][i] for i in sorted(sample["gt_frames"])])
        write_json(episode_root / "frame_sets.json", frame_sets_for_episode(episode))
        manifest.append({**episode.to_dict(), "gt_dir": str(episode_root / "frames")})
    write_json(output_root / "ground_truth" / "manifest.json", {"episodes": manifest})


def write_selected_episode_files(episodes: Sequence[EpisodeSpec], output_root: Path) -> None:
    tokens = [
        f"{episode.scene_idx},{episode.kf0_idx},{episode.kf1_idx},{episode.kf2_idx}"
        for episode in episodes
    ]
    (output_root / "selected_episode_tokens.txt").write_text("\n".join(tokens) + "\n")
    (output_root / "selected_episode_names.txt").write_text("\n".join(episode.name for episode in episodes) + "\n")
    write_json(
        output_root / "selected_episodes.json",
        {
            "episodes": [episode.to_dict() for episode in episodes],
            "episode_tokens": tokens,
        },
    )


class BeliefTemporalRunnerRE10K:
    def __init__(self, args: argparse.Namespace):
        add_project_paths()
        import hydra
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        if args.belief_config_profile == "re10k_256_from_128":
            belief_profile_overrides = [
                "setting_name=debug",
                "dataset.vggt_alignment_loss_weight=2.0",
                "dataset.intermediate_weight=5.0",
                "dataset.depth_loss_weight=0.5",
                "dataset.depth_smooth_loss_weight=0.1",
                "ctxt_losses_factor=0.8",
                "model.encoder.backbone.use_vggt_alignment=false",
                "repa_encoder_name=dinov3-vit-b",
                (
                    "repa_encoder_weights="
                    f"{Path(__file__).resolve().parents[2] / 'checkpoints' / 'dinov3_vitb16_pretrain_lvd1689m.pth'}"
                ),
                "use_depth_smoothness=true",
            ]
        else:
            belief_profile_overrides = [
                "setting_name=pixelsplat_h100",
                "dataset.vggt_alignment_loss_weight=2.0",
                "dataset.depth_loss_weight=1.0",
                "dataset.depth_smooth_loss_weight=0.1",
                "ctxt_losses_factor=0.7",
                "model.encoder.backbone.use_vggt_alignment=true",
                "use_depth_smoothness=true",
            ]

        # Match the selected RE10K 3D-Belief training architecture.
        overrides = [
            "dataset=realestate",
            f"dataset.root_dir={args.dataset_root}",
            f"stage={args.stage}",
            "batch_size=1",
            "num_target=1",
            "num_context=1",
            "ctxt_min=50",
            "ctxt_max=150",
            "intermediate=true",
            "num_intermediate=15",
            "use_depth_supervision=false",
            "use_identity=true",
            "clean_target=false",
            "use_history=false",
            "ctxt_losses_factor=0.7",
            "repa_encoder_resolution=512",
            "alignment.latents_info=-1",
            "model/encoder=uvitmvsplat",
            "model.encoder.use_image_condition=true",
            "model.encoder.depth_predictor_time_embed=true",
            "model.encoder.use_camera_pose=true",
            "model.encoder.use_semantic=false",
            "model.encoder.use_reg_model=false",
            "model.encoder.d_semantic=512",
            "model.encoder.d_semantic_reg=384",
            "model.encoder.gaussians_per_pixel=1",
            "model.encoder.evolve_ctxt=false",
            "model.encoder.use_depth_mask=false",
            "model.encoder.freeze_depth_predictor=false",
            "model.encoder.inference_mode=false",
            "model.encoder.grid_sample_disable_cudnn=true",
            "model/encoder/backbone=u_vit3d_pose",
            "model.encoder.backbone.use_repa=true",
            f"model.encoder.backbone.input_size=[{args.image_size},{args.image_size}]",
            "model_type=uvit_pose",
            "semantic_mode=embed",
            "semantic_viz=query",
            f"temperature={args.belief_temperature}",
            f"sampling_steps={args.belief_sampling_steps}",
            f"sampler={args.belief_sampler}",
            "name=vision_metrics_re10k_belief",
            f"image_size={args.image_size}",
            f"adjacent_angle={args.adjacent_angle}",
            f"adjacent_distance={args.adjacent_distance}",
            f"semantic_config={Path('splat_belief/config/semantic/onehot.yaml')}",
            f"checkpoint_path={args.belief_checkpoint}",
            f"results_folder={DEFAULT_OUTPUT_ROOT / '_tmp_belief_re10k'}",
        ]
        overrides.extend(belief_profile_overrides)
        refiner_overrides = {
            "refiner.enabled": getattr(args, "belief_refiner_enabled", None),
            "refiner.num_iterations": getattr(args, "belief_refiner_num_iterations", None),
            "refiner.prior_weight": getattr(args, "belief_refiner_prior_weight", None),
            "refiner.depth_consistency_weight": getattr(args, "belief_refiner_depth_consistency_weight", None),
            "refiner.position_update_mode": getattr(args, "belief_refiner_position_update_mode", None),
            "refiner.ray_tangent_weight": getattr(args, "belief_refiner_ray_tangent_weight", None),
            "refiner.ray_min_depth": getattr(args, "belief_refiner_ray_min_depth", None),
        }
        for key, value in refiner_overrides.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = str(value).lower()
            overrides.append(f"{key}={value}")
        config_dir = str(Path(__file__).resolve().parents[2] / "splat_belief" / "config")
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = hydra.compose(config_name="config", overrides=overrides)

        from splat_belief.experiment.compare_samplers import _build_model_and_trainer

        self.cfg = cfg
        self.trainer, _ = _build_model_and_trainer(cfg)
        self.dtype = args.belief_inference_dtype
        self.obj_permanence_config = self._configure_obj_permanence(args)
        self.save_gaussians = bool(getattr(args, "belief_save_gaussians", False))
        self.gaussian_layers = {
            item.strip()
            for item in str(getattr(args, "belief_gaussian_layers", "full")).split(",")
            if item.strip()
        }
        self.gaussian_export_filters = {
            "opacity_thresh": float(getattr(args, "belief_gaussian_opacity_thresh", 0.0)),
            "max_scale": float(getattr(args, "belief_gaussian_max_scale", 0.0)),
            "max_count": int(getattr(args, "belief_gaussian_max_count", 0)),
            "voxel_size": float(getattr(args, "belief_gaussian_voxel_size", 0.0)),
            "min_voxel_count": int(getattr(args, "belief_gaussian_min_voxel_count", 1)),
        }

    def _belief_modules(self) -> dict[str, Any]:
        return {
            "online": self.trainer.model.model,
            "ema": self.trainer.ema.ema_model.model,
        }

    def _configure_obj_permanence(self, args: argparse.Namespace) -> dict:
        config = {
            "requested_mode": str(args.belief_obj_permanence_mode),
            "requested_state_t_min": int(args.belief_obj_permanence_state_t_min),
            "requested_mask_blur": int(args.belief_obj_permanence_mask_blur),
            "requested_mask_threshold": float(args.belief_obj_permanence_mask_threshold),
            "requested_erode_kernel": int(args.belief_obj_permanence_erode_kernel),
            "requested_mask_binarize_after_blur": bool(args.belief_obj_permanence_mask_binarize),
            "requested_dps_guidance_scale": float(args.belief_dps_guidance_scale),
            "requested_dps_pos_weight": float(args.belief_dps_pos_weight),
            "requested_dps_opacity_weight": float(args.belief_dps_opacity_weight),
            "resolved": {},
        }
        for name, module in self._belief_modules().items():
            if not hasattr(module, "obj_permanence_mode"):
                raise AttributeError(
                    f"3D-Belief {name} module has no obj_permanence_mode attribute; "
                    "this checkout cannot run object-permanence inference."
                )
            module.obj_permanence_mode = str(args.belief_obj_permanence_mode)
            if hasattr(module, "obj_permanence_state_t_min"):
                module.obj_permanence_state_t_min = int(args.belief_obj_permanence_state_t_min)
            if hasattr(module, "obj_permanence_mask_blur"):
                module.obj_permanence_mask_blur = int(args.belief_obj_permanence_mask_blur)
            if hasattr(module, "obj_permanence_mask_threshold"):
                module.obj_permanence_mask_threshold = float(args.belief_obj_permanence_mask_threshold)
            if hasattr(module, "obj_permanence_erode_kernel"):
                module.obj_permanence_erode_kernel = int(args.belief_obj_permanence_erode_kernel)
            module.obj_permanence_mask_binarize_after_blur = bool(args.belief_obj_permanence_mask_binarize)
            if hasattr(module, "dps_guidance_scale"):
                module.dps_guidance_scale = float(args.belief_dps_guidance_scale)
            if hasattr(module, "dps_pos_weight"):
                module.dps_pos_weight = float(args.belief_dps_pos_weight)
            if hasattr(module, "dps_opacity_weight"):
                module.dps_opacity_weight = float(args.belief_dps_opacity_weight)
            config["resolved"][name] = {
                "obj_permanence_mode": getattr(module, "obj_permanence_mode", None),
                "obj_permanence_state_t_min": getattr(module, "obj_permanence_state_t_min", None),
                "obj_permanence_mask_blur": getattr(module, "obj_permanence_mask_blur", None),
                "obj_permanence_mask_threshold": getattr(module, "obj_permanence_mask_threshold", None),
                "obj_permanence_erode_kernel": getattr(module, "obj_permanence_erode_kernel", None),
                "obj_permanence_mask_binarize_after_blur": getattr(module, "obj_permanence_mask_binarize_after_blur", None),
                "dps_guidance_scale": getattr(module, "dps_guidance_scale", None),
                "dps_pos_weight": getattr(module, "dps_pos_weight", None),
                "dps_opacity_weight": getattr(module, "dps_opacity_weight", None),
            }
        print(f"[3d_belief] object permanence config: {config}")
        return config

    def _make_input(self, sample: Mapping[str, Any], ctxt_idx: int, trgt_idx: int,
                    render_indices: Sequence[int]) -> dict:
        from splat_belief.utils.vision_utils import to_gpu

        video_dict = sample["video_dict"]
        render_poses = video_dict["render_poses"]
        rgbs = video_dict["rgbs"]
        abs_camera_poses = video_dict["abs_camera_poses"]

        inp = {
            "ctxt_c2w": render_poses[ctxt_idx],
            "ctxt_rgb": rgbs[ctxt_idx],
            "ctxt_abs_camera_poses": abs_camera_poses[ctxt_idx],
            "trgt_c2w": render_poses[trgt_idx],
            "trgt_rgb": rgbs[trgt_idx],
            "trgt_abs_camera_poses": abs_camera_poses[trgt_idx],
            "intrinsics": video_dict["intrinsics"][0],
            "image_shape": video_dict["image_shape"],
            "render_poses": torch.cat([render_poses[i] for i in render_indices], dim=0),
            "near": torch.tensor(video_dict["near"]),
            "far": torch.tensor(video_dict["far"]),
            "lang": video_dict["lang"],
        }
        inp = to_gpu(inp, "cuda")
        for key in list(inp.keys()):
            if key != "num_frames_render" and torch.is_tensor(inp[key]):
                inp[key] = inp[key].unsqueeze(0)
        inp["num_frames_render"] = len(render_indices)
        return inp

    def _sample(self, sample: Mapping[str, Any], ctxt_idx: int, trgt_idx: int,
                render_indices: Sequence[int], state_t: int,
                clean_target: bool = False):
        inp = self._make_input(sample, ctxt_idx, trgt_idx, render_indices)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.time()
        with self._temporary_clean_target(clean_target), torch.no_grad(), make_autocast_ctx(self.dtype):
            out = self.trainer.ema.ema_model.sample(batch_size=1, inp=inp, state_t=state_t)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out, time.time() - started

    @contextmanager
    def _temporary_clean_target(self, enabled: bool):
        diffusion_modules = [
            getattr(self.trainer, "model", None),
            getattr(getattr(self.trainer, "ema", None), "ema_model", None),
        ]
        previous_values = []
        for module in diffusion_modules:
            if module is not None and hasattr(module, "clean_target"):
                previous_values.append((module, module.clean_target))
                module.clean_target = bool(enabled)
        try:
            yield
        finally:
            for module, previous in previous_values:
                module.clean_target = previous

    @staticmethod
    def _frames_from_output(out: Mapping[str, Any]) -> List[np.ndarray]:
        from splat_belief.experiment.temporal_inference import prepare_video_viz

        result = prepare_video_viz(out)
        return list(result[0])

    @staticmethod
    def _write_split(episode_dir: Path, split: str,
                     frames: Mapping[int, np.ndarray],
                     requested_indices: Sequence[int],
                     gaussian_assets: Mapping[str, str] | None = None) -> None:
        split_dir = episode_dir / split
        save_indexed_frames(frames, split_dir / "frames")
        save_video(split_dir / "prediction.mp4", [frames[i] for i in sorted(frames)])
        manifest = {
            "split": split,
            "requested_frame_indices": list(requested_indices),
            "saved_frame_indices": sorted(frames),
        }
        if gaussian_assets:
            manifest["gaussian_assets"] = dict(gaussian_assets)
        write_json(split_dir / "manifest.json", manifest)

    def _export_gaussian_assets(self, sample: Mapping[str, Any], episode_dir: Path,
                                split: str, reference_frame_idx: int) -> dict[str, str]:
        if not self.save_gaussians:
            return {}
        from splat_belief.splat.types import Gaussians
        from splat_belief.splat.ply_export import compute_ply_normalization, export_gaussians_to_ply

        model = self.trainer.ema.ema_model.model
        gaussians_by_name = {
            "full": getattr(model, "augmented_gaussians", None),
            "history": getattr(model, "history_gaussians", None),
            "belief": getattr(model, "belief_gaussians", None),
            "incremental": getattr(model, "incremental_gaussians", None),
        }
        reference_pose = sample["video_dict"]["render_poses"][reference_frame_idx]
        if torch.is_tensor(reference_pose):
            reference_pose = reference_pose.to("cuda")
            if reference_pose.ndim == 2:
                reference_pose = reference_pose.unsqueeze(0)

        gaussian_dir = episode_dir / split / "gaussians"
        gaussian_dir.mkdir(parents=True, exist_ok=True)
        assets = {}
        filter_stats = {}
        shared_normalization = None
        normalization_manifest = None
        normalization_source = getattr(model, "augmented_gaussians", None)
        if normalization_source is not None:
            normalization_gaussians, normalization_stats = self._filter_gaussians_for_export(
                normalization_source.detach().float(), Gaussians,
            )
            if normalization_gaussians is not None:
                shared_normalization = compute_ply_normalization(normalization_gaussians.means[0])
                center, scale_factor = shared_normalization
                normalization_manifest = {
                    "source": "filtered_augmented_gaussians",
                    "source_filter_stats": normalization_stats,
                    "center": [float(v) for v in center.detach().cpu().tolist()],
                    "scale_factor": float(scale_factor.detach().cpu().item()),
                }
        for name, gaussians in gaussians_by_name.items():
            if name not in self.gaussian_layers:
                continue
            if gaussians is None:
                continue
            gaussians, stats = self._filter_gaussians_for_export(gaussians.detach().float(), Gaussians)
            filter_stats[name] = stats
            if gaussians is None:
                print(f"[3d_belief] skipping empty Gaussian export for {split}/{name}: {stats}")
                continue
            ply_path = gaussian_dir / f"{name}.ply"
            export_gaussians_to_ply(
                gaussians, reference_pose, ply_path,
                normalization=shared_normalization,
            )
            assets[name] = str(ply_path.relative_to(episode_dir))
        if assets:
            write_json(gaussian_dir / "manifest.json", {
                "split": split,
                "reference_frame_idx": int(reference_frame_idx),
                "filters": self.gaussian_export_filters,
                "filter_stats": filter_stats,
                "shared_normalization": normalization_manifest,
                "assets": assets,
            })
        return assets

    def _filter_gaussians_for_export(self, gaussians, gaussians_cls):
        filters = self.gaussian_export_filters
        means = gaussians.means
        covs = gaussians.covariances
        harmonics = gaussians.harmonics
        opacities = gaussians.opacities
        features = gaussians.features

        if means.ndim != 3 or means.shape[0] != 1:
            raise ValueError(
                "Viewer Gaussian export filtering currently expects batch size 1; "
                f"got means shape {tuple(means.shape)}"
            )

        keep = (
            torch.isfinite(means).all(dim=-1)
            & torch.isfinite(covs).reshape(*covs.shape[:2], -1).all(dim=-1)
            & torch.isfinite(harmonics).reshape(*harmonics.shape[:2], -1).all(dim=-1)
            & torch.isfinite(opacities)
        )

        opacity_thresh = float(filters["opacity_thresh"])
        if opacity_thresh > 0:
            keep = keep & (opacities >= opacity_thresh)

        max_scale = float(filters["max_scale"])
        if max_scale > 0 and keep.any():
            covs_sym = 0.5 * (covs + covs.transpose(-1, -2))
            try:
                evals = torch.linalg.eigvalsh(covs_sym)
            except Exception:
                evals = torch.linalg.eigvalsh(covs_sym.detach().cpu()).to(covs.device)
            gaussian_scales = evals.clamp_min(0).sqrt().max(dim=-1).values
            keep = keep & (gaussian_scales <= max_scale)

        voxel_size = float(filters["voxel_size"])
        min_voxel_count = int(filters["min_voxel_count"])
        if voxel_size > 0 and min_voxel_count > 1 and keep.any():
            voxel_keep = torch.zeros_like(keep)
            kept_idx = torch.nonzero(keep[0], as_tuple=False).squeeze(-1)
            coords = torch.floor(means[0, kept_idx] / voxel_size).to(torch.int64)
            _, inverse, counts = torch.unique(coords, dim=0, return_inverse=True, return_counts=True)
            local_keep = counts[inverse] >= min_voxel_count
            voxel_keep[0, kept_idx[local_keep]] = True
            keep = voxel_keep

        max_count = int(filters["max_count"])
        if max_count > 0 and int(keep.sum().item()) > max_count:
            kept_idx = torch.nonzero(keep[0], as_tuple=False).squeeze(-1)
            scores = opacities[0, kept_idx]
            top_idx = torch.topk(scores, k=max_count, largest=True, sorted=False).indices
            limited_keep = torch.zeros_like(keep)
            limited_keep[0, kept_idx[top_idx]] = True
            keep = limited_keep

        input_count = int(means.shape[1])
        output_count = int(keep.sum().item())
        stats = {
            "input_count": input_count,
            "output_count": output_count,
            "removed_count": input_count - output_count,
            "kept_fraction": float(output_count / max(input_count, 1)),
        }
        if output_count == 0:
            return None, stats

        m = keep[0]
        filtered_features = features[:, m].contiguous() if features is not None else None
        return gaussians_cls(
            means=means[:, m].contiguous(),
            covariances=covs[:, m].contiguous(),
            harmonics=harmonics[:, m].contiguous(),
            opacities=opacities[:, m].contiguous(),
            features=filtered_features,
        ), stats

    def predict_episode(self, sample: Mapping[str, Any], episode_dir: Path) -> dict:
        episode = sample["episode"]
        frame_sets = frame_sets_for_episode(episode)
        kf1 = episode.local_kf1
        kf2 = episode.local_kf2

        self.trainer.model.model.reset_timestep()
        self.trainer.ema.ema_model.model.reset_timestep()

        trace = {
            "model": "3d_belief",
            "episode": episode.to_dict(),
            "obj_permanence": self.obj_permanence_config,
            "splits": {},
            "total_diffusion_calls": 0,
            "total_prediction_seconds": 0.0,
            "notes": [
                "RE10K: 3D-Belief uses temporal_inference stack (no semantic/reg model, no image cond).",
                "`imagined_kf0_to_kf1` is the state_t=0 prediction (only kf0 observed).",
                "`observed` is a separate clean-target state_t=0 prediction from kf0 to kf1.",
            ],
        }

        # State 0: condition on kf0, predict kf1.
        render_01 = list(range(0, kf1 + 1))
        out_01, elapsed_01 = self._sample(sample, ctxt_idx=0, trgt_idx=kf1,
                                          render_indices=render_01, state_t=0)
        frames_01 = self._frames_from_output(out_01)
        frame_map_01 = {idx: resize_uint8(frame, sample["height_width"])
                        for idx, frame in zip(render_01, frames_01)}
        trace["total_diffusion_calls"] += 1
        trace["total_prediction_seconds"] += elapsed_01

        imagined_01_frames = {idx: frame_map_01[idx]
                              for idx in frame_sets["imagined_kf0_to_kf1"]
                              if idx in frame_map_01}
        gaussian_assets_01 = self._export_gaussian_assets(
            sample, episode_dir, "imagined_kf0_to_kf1", reference_frame_idx=0,
        )
        self._write_split(episode_dir, "imagined_kf0_to_kf1",
                          imagined_01_frames, frame_sets["imagined_kf0_to_kf1"],
                          gaussian_assets=gaussian_assets_01)
        trace["splits"]["imagined_kf0_to_kf1"] = {
            "mode": "state_t0_context_kf0_only_predict_kf1",
            "frame_indices": frame_sets["imagined_kf0_to_kf1"],
            "elapsed_seconds": elapsed_01,
            "diffusion_calls": 1,
            "gaussian_assets": gaussian_assets_01,
        }

        # State 1: condition on kf1, predict kf2 (state carries kf0).
        render_12 = list(range(0, kf2 + 1))
        out_12, elapsed_12 = self._sample(sample, ctxt_idx=kf1, trgt_idx=kf2,
                                          render_indices=render_12, state_t=1)
        frames_12 = self._frames_from_output(out_12)
        frame_map_12 = {idx: resize_uint8(frame, sample["height_width"])
                        for idx, frame in zip(render_12, frames_12)}
        imagined_12_frames = {idx: frame_map_12[idx]
                              for idx in frame_sets["imagined_kf1_to_kf2"]
                              if idx in frame_map_12}
        gaussian_assets_12 = self._export_gaussian_assets(
            sample, episode_dir, "imagined_kf1_to_kf2", reference_frame_idx=kf1,
        )
        self._write_split(episode_dir, "imagined_kf1_to_kf2",
                          imagined_12_frames, frame_sets["imagined_kf1_to_kf2"],
                          gaussian_assets=gaussian_assets_12)
        trace["splits"]["imagined_kf1_to_kf2"] = {
            "mode": "state_t1_context_kf1_target_kf2",
            "frame_indices": frame_sets["imagined_kf1_to_kf2"],
            "elapsed_seconds": elapsed_12,
            "diffusion_calls": 1,
            "gaussian_assets": gaussian_assets_12,
        }
        trace["total_diffusion_calls"] += 1
        trace["total_prediction_seconds"] += elapsed_12

        # Keep scene-memory evaluation independent of the imagination branch.
        self.trainer.model.model.reset_timestep()
        self.trainer.ema.ema_model.model.reset_timestep()
        out_observed, elapsed_observed = self._sample(
            sample,
            ctxt_idx=0,
            trgt_idx=kf1,
            render_indices=render_01,
            state_t=0,
            clean_target=True,
        )
        frames_observed = self._frames_from_output(out_observed)
        frame_map_observed = {idx: resize_uint8(frame, sample["height_width"])
                              for idx, frame in zip(render_01, frames_observed)}
        observed_frames = {idx: frame_map_observed[idx]
                           for idx in frame_sets["observed"]
                           if idx in frame_map_observed}
        gaussian_assets_observed = self._export_gaussian_assets(
            sample, episode_dir, "observed", reference_frame_idx=0,
        )
        self._write_split(episode_dir, "observed", observed_frames,
                          frame_sets["observed"], gaussian_assets=gaussian_assets_observed)
        trace["splits"]["observed"] = {
            "mode": "state_t0_context_kf0_clean_target_kf1_scene_memory",
            "frame_indices": frame_sets["observed"],
            "elapsed_seconds": elapsed_observed,
            "diffusion_calls": 1,
            "clean_target": True,
            "gaussian_assets": gaussian_assets_observed,
        }
        trace["total_diffusion_calls"] += 1
        trace["total_prediction_seconds"] += elapsed_observed
        return trace


@contextmanager
def count_sample_sequences(algo) -> Iterable[dict]:
    counter = {"calls": 0}
    original = algo._sample_sequence

    def wrapped(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    algo._sample_sequence = wrapped
    try:
        yield counter
    finally:
        algo._sample_sequence = original


class DFoTRunnerRE10K:
    def __init__(self, args: argparse.Namespace):
        patch_numpy_legacy_aliases()
        add_project_paths(args.dfot_repo)

        import hydra as _hydra
        from hydra.core.global_hydra import GlobalHydra as _GlobalHydra
        if _GlobalHydra.instance().is_initialized():
            _GlobalHydra.instance().clear()

        # Match RE10K checkpoint config and avoid compile prefixes.
        cfg_overrides = [
            "+name=vision_metrics_re10k_dfot",
            "dataset=realestate10k_mini",
            "algorithm=dfot_video_pose",
            "experiment=video_generation",
            ("++algorithm={diffusion:{is_continuous:True,precond_scale:0.125},"
             "backbone:{use_fourier_noise_embedding:True},compile:False}"),
        ]
        config_dir = str((args.dfot_repo / "configurations").resolve())
        with _hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = _hydra.compose(config_name="config", overrides=cfg_overrides)
        from model_wrapper import ModelWrapper as _ModelWrapper

        wrapper = _ModelWrapper(cfg, str(args.dfot_checkpoint))
        self.algo = wrapper.algo.to("cuda").eval()
        self.resolution = int(args.dfot_resolution)
        self.include_kf1_to_kf2 = bool(args.dfot_include_kf1_to_kf2)

    def _video_tensor(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        arr = np.stack([tensor_to_uint8_image(frame) for frame in frames], axis=0)
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(self.resolution, self.resolution),
                          mode="bilinear", antialias=True)
        x = x.to(self.algo.device)
        return self.algo._normalize_x(x)

    def _conds(self, sample: Mapping[str, Any], indices: Sequence[int],
               pad_to: int | None = None) -> torch.Tensor:
        # DFoT_RE10K conds: normalized intrinsics plus OpenCV w2c.
        intr_list = sample["video_dict"]["intrinsics"]
        rows = []
        for idx in indices:
            intr = intr_list[idx]
            if torch.is_tensor(intr):
                intr = intr.detach().cpu().numpy()
            intr = np.asarray(intr, dtype=np.float64).reshape(3, 3)
            fx = float(intr[0, 0])
            fy = float(intr[1, 1])
            cx = float(intr[0, 2])
            cy = float(intr[1, 2])
            c2w = pose_array(sample["render_poses"][idx])
            w2c = np.linalg.inv(c2w)
            rows.append(np.concatenate([[fx, fy, cx, cy], w2c[:3, :].reshape(-1)], axis=0))
        if pad_to is not None and len(rows) < pad_to:
            rows.extend([rows[-1].copy() for _ in range(pad_to - len(rows))])
        return torch.tensor(np.stack(rows, axis=0), dtype=torch.float32,
                            device=self.algo.device).unsqueeze(0)

    def _uint8_frames(self, video_norm: torch.Tensor, indices: Sequence[int],
                      size_hw: tuple[int, int]) -> dict:
        video = self.algo._unnormalize_x(video_norm).clamp(0, 1)
        if video.ndim == 5:
            video = video[0]
        frames = {}
        for out_idx, frame_idx in enumerate(indices):
            frame = tensor_to_uint8_image(video[out_idx])
            frames[int(frame_idx)] = resize_uint8(frame, size_hw)
        return frames

    def _interpolate(self, sample: Mapping[str, Any], indices: Sequence[int],
                     known_indices: Sequence[int]):
        gt_frames = sample["gt_frames"]
        input_frames = [gt_frames[known_indices[0]]] * len(indices)
        video_norm = self._video_tensor(input_frames).unsqueeze(0)
        index_to_pos = {idx: pos for pos, idx in enumerate(indices)}
        for known_idx in known_indices:
            video_norm[:, index_to_pos[known_idx]] = self._video_tensor([gt_frames[known_idx]])[0]
        mask = torch.zeros((1, len(indices)), dtype=torch.bool, device=self.algo.device)
        for known_idx in known_indices:
            mask[:, index_to_pos[known_idx]] = True
        conds = self._conds(sample, indices, pad_to=max(len(indices), self.algo.max_tokens))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.time()
        with torch.no_grad(), count_sample_sequences(self.algo) as counter:
            pred_norm = self.algo._interpolate_videos(
                video_norm, context_mask=mask, conditions=conds[:, : len(indices)]
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (self._uint8_frames(pred_norm, indices, sample["height_width"]),
                time.time() - started, int(counter["calls"]))

    def _predict(self, sample: Mapping[str, Any], context_indices: Sequence[int],
                 output_indices: Sequence[int]):
        gt_frames = sample["gt_frames"]
        all_indices = list(context_indices) + [idx for idx in output_indices
                                                if idx not in context_indices]
        context_frames = [gt_frames[idx] for idx in context_indices]
        context = self._video_tensor(context_frames).unsqueeze(0)
        conds = self._conds(sample, all_indices,
                            pad_to=len(all_indices) + self.algo.max_tokens)
        sliding_context_len = (
            self.algo.cfg.tasks.prediction.sliding_context_len
            or self.algo.max_tokens // 2
        )
        sliding_context_len = max(int(sliding_context_len), len(context_indices))
        if sliding_context_len >= self.algo.max_tokens:
            sliding_context_len = self.algo.max_tokens - 1
        if len(context_indices) > sliding_context_len:
            raise ValueError(
                f"DFoT initial context has {len(context_indices)} frames, "
                f"but max usable context is {sliding_context_len}"
            )
        from algorithms.dfot.history_guidance import HistoryGuidance

        history_guidance = HistoryGuidance.from_config(
            config=self.algo.cfg.tasks.prediction.history_guidance,
            timesteps=self.algo.timesteps,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.time()
        with torch.no_grad(), count_sample_sequences(self.algo) as counter:
            pred_norm, _ = self.algo._predict_sequence(
                context,
                length=len(all_indices),
                conditions=conds,
                history_guidance=history_guidance,
                reconstruction_guidance=self.algo.cfg.diffusion.reconstruction_guidance,
                sliding_context_len=sliding_context_len,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        all_frames = self._uint8_frames(pred_norm, all_indices, sample["height_width"])
        return ({idx: all_frames[idx] for idx in output_indices if idx in all_frames},
                time.time() - started, int(counter["calls"]),
                f"sliding_context_len={sliding_context_len}")

    def predict_episode(self, sample: Mapping[str, Any], episode_dir: Path) -> dict:
        episode = sample["episode"]
        frame_sets = frame_sets_for_episode(episode)
        kf1 = episode.local_kf1
        kf2 = episode.local_kf2
        trace = {
            "model": "dfot",
            "episode": episode.to_dict(),
            "splits": {},
            "total_prediction_seconds": 0.0,
            "total_diffusion_calls": 0,
            "notes": ["RE10K: DFoT_RE10K.ckpt; observed split uses native interpolation between kf0 and kf1."],
        }

        observed_indices = list(range(0, kf1 + 1))
        observed_frames, elapsed, calls = self._interpolate(
            sample, observed_indices, known_indices=[0, kf1]
        )
        observed_save = {idx: observed_frames[idx] for idx in frame_sets["observed"]
                         if idx in observed_frames}
        BeliefTemporalRunnerRE10K._write_split(
            episode_dir, "observed", observed_save, frame_sets["observed"]
        )
        trace["splits"]["observed"] = {
            "mode": "dfot_native_interpolation_kf0_kf1_fixed",
            "frame_indices": frame_sets["observed"],
            "elapsed_seconds": elapsed,
            "diffusion_calls": calls,
        }
        trace["total_prediction_seconds"] += elapsed
        trace["total_diffusion_calls"] += calls

        imagined_01, elapsed, calls, context_note = self._predict(
            sample, [0], frame_sets["imagined_kf0_to_kf1"]
        )
        BeliefTemporalRunnerRE10K._write_split(
            episode_dir, "imagined_kf0_to_kf1", imagined_01, frame_sets["imagined_kf0_to_kf1"]
        )
        trace["splits"]["imagined_kf0_to_kf1"] = {
            "mode": "dfot_prediction_context_kf0",
            "frame_indices": frame_sets["imagined_kf0_to_kf1"],
            "elapsed_seconds": elapsed,
            "diffusion_calls": calls,
            "context_note": context_note,
        }
        trace["total_prediction_seconds"] += elapsed
        trace["total_diffusion_calls"] += calls

        if self.include_kf1_to_kf2:
            context_indices = list(range(0, kf1 + 1))
            if len(context_indices) <= self.algo.max_tokens - 1:
                imagined_12, elapsed, calls, context_note = self._predict(
                    sample, context_indices, frame_sets["imagined_kf1_to_kf2"]
                )
                BeliefTemporalRunnerRE10K._write_split(
                    episode_dir, "imagined_kf1_to_kf2",
                    imagined_12, frame_sets["imagined_kf1_to_kf2"]
                )
                trace["splits"]["imagined_kf1_to_kf2"] = {
                    "mode": "dfot_prediction_all_observed_frames_kf0_to_kf1_as_context",
                    "frame_indices": frame_sets["imagined_kf1_to_kf2"],
                    "elapsed_seconds": elapsed,
                    "diffusion_calls": calls,
                    "context_note": context_note,
                }
                trace["total_prediction_seconds"] += elapsed
                trace["total_diffusion_calls"] += calls
            else:
                trace["splits"]["imagined_kf1_to_kf2"] = {
                    "mode": "skipped",
                    "reason": (f"DFoT max initial context is {self.algo.max_tokens - 1}; "
                               f"observed context has {len(context_indices)} frames."),
                    "frame_indices": frame_sets["imagined_kf1_to_kf2"],
                    "elapsed_seconds": 0.0,
                    "diffusion_calls": 0,
                }
        return trace


def build_runner(model_name: str, args: argparse.Namespace):
    if model_name == "3d_belief":
        return BeliefTemporalRunnerRE10K(args)
    if model_name == "dfot":
        return DFoTRunnerRE10K(args)
    raise ValueError(model_name)


def run_gen3c_external_re10k(output_root: Path, args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    export_cmd = [
        sys.executable,
        str(script_dir / "export_gen3c_inputs_re10k.py"),
        "--run-dir",
        str(output_root),
        "--dataset-root",
        str(args.dataset_root),
        "--stage",
        str(args.stage),
        "--image-size",
        str(args.image_size),
        "--adjacent-angle",
        str(args.adjacent_angle),
        "--adjacent-distance",
        str(args.adjacent_distance),
    ]
    if args.max_scenes is not None:
        export_cmd.extend(["--max-scenes", str(args.max_scenes)])
    if args.skip_existing:
        export_cmd.append("--skip-existing")
    else:
        export_cmd.append("--no-skip-existing")
    print("[gen3c] exporting RE10K RGB/camera packs")
    subprocess.run(export_cmd, check=True)

    gen3c_cmd = [
        args.gen3c_python,
        str(script_dir / "run_gen3c_predictions.py"),
        "--run-dir",
        str(output_root),
        "--gen3c-repo",
        str(args.gen3c_repo),
        "--height",
        str(args.gen3c_height),
        "--width",
        str(args.gen3c_width),
        "--fps",
        str(args.gen3c_fps),
        "--seed",
        str(args.gen3c_seed),
        "--guidance",
        str(args.gen3c_guidance),
        "--num-steps",
        str(args.gen3c_num_steps),
        "--num-gpus",
        str(args.gen3c_num_gpus),
        "--strategy",
        str(args.gen3c_strategy),
        "--filter-points-threshold",
        str(args.gen3c_filter_points_threshold),
        "--missing-depth-policy",
        str(args.gen3c_missing_depth_policy),
    ]
    if args.gen3c_checkpoint_dir is not None:
        gen3c_cmd.extend(["--checkpoint-dir", str(args.gen3c_checkpoint_dir)])
    if args.skip_existing:
        gen3c_cmd.append("--skip-existing")
    else:
        gen3c_cmd.append("--no-skip-existing")
    if args.gen3c_no_foreground_masking:
        gen3c_cmd.append("--no-foreground-masking")
    if args.gen3c_enable_prompt_encoder:
        gen3c_cmd.append("--enable-prompt-encoder")
    if args.gen3c_offload:
        gen3c_cmd.extend(
            [
                "--offload-diffusion-transformer",
                "--offload-tokenizer",
                "--offload-text-encoder-model",
                "--offload-prompt-upsampler",
                "--offload-guardrail-models",
            ]
        )

    gen3c_env = os.environ.copy()
    gen3c_env["PYTHONNOUSERSITE"] = "1"
    gen3c_env["NVTE_FRAMEWORK"] = "pytorch"
    gen3c_env["CUDA_HOME"] = str(args.gen3c_cuda_home.resolve())
    gen3c_env["CONDA_PREFIX"] = str(args.gen3c_cuda_home.resolve())
    gen3c_env["PYTHONPATH"] = f"{args.gen3c_repo.resolve()}:{gen3c_env.get('PYTHONPATH', '')}"
    print(f"[gen3c] running external RE10K adapter with {args.gen3c_python}")
    subprocess.run(gen3c_cmd, check=True, env=gen3c_env)


def run_predictions_for_model(model_name: str, dataset: Any,
                              episodes: Sequence[EpisodeSpec],
                              output_root: Path, args: argparse.Namespace) -> None:
    model_root = output_root / "models" / model_name
    if args.skip_existing and model_root.exists():
        print(f"[{model_name}] skipping existing model folder: {model_root}")
        return
    print(f"[{model_name}] loading")
    started = time.time()
    runner = None
    release_cuda_memory()
    try:
        runner = build_runner(model_name, args)
        load_seconds = time.time() - started
        print(f"[{model_name}] loaded in {load_seconds:.2f}s")
        for episode in episodes:
            episode_dir = model_root / episode.name
            if args.skip_existing and (episode_dir / "trace.json").exists():
                print(f"[{model_name}] skipping {episode.name}")
                continue
            print(f"[{model_name}] episode {episode.name}")
            sample = load_episode_sample(dataset, episode)
            trace = runner.predict_episode(sample, episode_dir)
            trace["load_seconds"] = load_seconds
            write_json(episode_dir / "trace.json", trace)
            print(
                f"[{model_name}] {episode.name}: "
                f"{trace['total_prediction_seconds']:.2f}s, "
                f"calls={trace['total_diffusion_calls']}"
            )
    finally:
        del runner
        release_cuda_memory()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    add_project_paths()
    args.dataset_root = args.dataset_root.resolve()

    models = normalize_models(args.models)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.run_name or time.strftime("re10k_temporal_vision_%Y%m%d_%H%M%S")
    output_root = (args.output_root or (DEFAULT_OUTPUT_ROOT / run_name)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        args.dataset_root, args.stage, args.image_size,
        args.adjacent_angle, args.adjacent_distance,
        args.max_scenes,
    )
    episodes = sample_episodes(dataset, args)
    if not episodes:
        raise RuntimeError("No evaluation episodes were selected.")

    manifest = {
        "run_name": run_name,
        "dataset": "realestate10k",
        "dataset_root": str(args.dataset_root),
        "stage": args.stage,
        "image_size": args.image_size,
        "max_scenes": args.max_scenes,
        "adjacent_angle": args.adjacent_angle,
        "adjacent_distance": args.adjacent_distance,
        "episodes": [episode.to_dict() for episode in episodes],
        "models": models,
        "checkpoints": {
            "3d_belief": str(args.belief_checkpoint.resolve()),
            "dfot": str(args.dfot_checkpoint.resolve()),
            "gen3c": str((args.gen3c_checkpoint_dir or (args.gen3c_repo / "checkpoints")).resolve()),
        },
        "belief_config_profile": args.belief_config_profile,
        "belief_obj_permanence": {
            "mode": args.belief_obj_permanence_mode,
            "state_t_min": args.belief_obj_permanence_state_t_min,
            "mask_blur": args.belief_obj_permanence_mask_blur,
            "mask_threshold": args.belief_obj_permanence_mask_threshold,
            "erode_kernel": args.belief_obj_permanence_erode_kernel,
            "mask_binarize_after_blur": args.belief_obj_permanence_mask_binarize,
            "dps_guidance_scale": args.belief_dps_guidance_scale,
            "dps_pos_weight": args.belief_dps_pos_weight,
            "dps_opacity_weight": args.belief_dps_opacity_weight,
        },
        "belief_refiner": {
            "enabled": args.belief_refiner_enabled,
            "num_iterations": args.belief_refiner_num_iterations,
            "prior_weight": args.belief_refiner_prior_weight,
            "depth_consistency_weight": args.belief_refiner_depth_consistency_weight,
            "position_update_mode": args.belief_refiner_position_update_mode,
            "ray_tangent_weight": args.belief_refiner_ray_tangent_weight,
            "ray_min_depth": args.belief_refiner_ray_min_depth,
        },
        "belief_gaussian_export_filters": {
            "opacity_thresh": args.belief_gaussian_opacity_thresh,
            "max_scale": args.belief_gaussian_max_scale,
            "max_count": args.belief_gaussian_max_count,
            "voxel_size": args.belief_gaussian_voxel_size,
            "min_voxel_count": args.belief_gaussian_min_voxel_count,
        },
        "dry_run": bool(args.dry_run),
        "env": env_snapshot(),
    }
    write_json(output_root / "manifest.json", manifest)
    gt_manifest = output_root / "ground_truth" / "manifest.json"
    if args.skip_existing and gt_manifest.exists():
        print(f"[ground-truth] reusing existing materialization at {gt_manifest.parent}")
    else:
        write_ground_truth(dataset, episodes, output_root)
    write_selected_episode_files(episodes, output_root)
    print(f"Wrote manifest and ground truth to {output_root}")

    if args.dry_run:
        print("Dry run complete.")
        return

    prediction_models = [model for model in models if model != "gen3c"]
    if prediction_models:
        require_cuda(prediction_models)
    for model_name in prediction_models:
        run_predictions_for_model(model_name, dataset, episodes, output_root, args)
    if "gen3c" in models:
        run_gen3c_external_re10k(output_root, args)


if __name__ == "__main__":
    main()
