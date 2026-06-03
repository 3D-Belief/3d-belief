#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from common import (
    DEFAULT_ADJACENT_ANGLE,
    DEFAULT_ADJACENT_DISTANCE,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    EpisodeSpec,
    add_project_paths,
    compute_key_frame_indices,
    env_snapshot,
    normalize_models,
    parse_episode_token,
    patch_numpy_legacy_aliases,
    pose_array,
    require_cuda,
    resize_uint8,
    resolve_stage,
    save_indexed_frames,
    save_video,
    tensor_to_uint8_image,
    triplet_is_monotonic,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporal vision predictions for 3D-Belief, DFoT, and NWM on common SPOC episodes."
    )
    parser.add_argument("--models", default="3d_belief,nwm,dfot")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--stage", default="test")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--episode", action="append", default=[], help="Override episode: scene,start,kf1,kf2")
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--scan-max-frames", type=int, default=80)
    parser.add_argument("--max-start-candidates", type=int, default=8)
    parser.add_argument("--adjacent-angle", type=float, default=DEFAULT_ADJACENT_ANGLE)
    parser.add_argument("--adjacent-distance", type=float, default=DEFAULT_ADJACENT_DISTANCE)
    parser.add_argument("--allow-nonmonotonic", action="store_true")

    parser.add_argument(
        "--belief-checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT / "3d_belief_spoc.pt",
    )
    parser.add_argument("--belief-sampling-steps", type=int, default=50)
    parser.add_argument("--belief-sampler", default="ddim")
    parser.add_argument("--belief-temperature", type=float, default=0.85)
    parser.add_argument(
        "--belief-obj-permanence-mode",
        default="none",
        choices=("none", "opacity", "dps"),
    )
    parser.add_argument(
        "--belief-obj-permanence-observed-mode",
        default="live",
        choices=("none", "live"),
    )
    parser.add_argument("--belief-d-semantic", type=int, default=512)
    parser.add_argument("--belief-d-semantic-reg", type=int, default=768)
    parser.add_argument("--belief-reg-model-name", default="dinov3_base")
    parser.add_argument("--belief-inference-dtype", default="fp32", choices=("fp32", "bf16", "fp16"))
    parser.add_argument(
        "--dinov3-weights",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT / "dinov3_vitb16_pretrain_lvd1689m.pth",
    )

    parser.add_argument("--dfot-checkpoint", type=Path, default=DEFAULT_CHECKPOINT_ROOT / "dfot_finetune_spoc.ckpt")
    parser.add_argument("--dfot-repo", type=Path, default=REPO_ROOT / "third_party" / "dfot")
    parser.add_argument("--dfot-resolution", type=int, default=256)
    parser.add_argument("--dfot-include-kf1-to-kf2", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--nwm-checkpoint", type=Path, default=DEFAULT_CHECKPOINT_ROOT / "nwm_finetune_spoc.pth.tar")
    parser.add_argument("--nwm-repo", type=Path, default=REPO_ROOT / "third_party" / "nwm")
    parser.add_argument("--nwm-num-timesteps", type=int, default=4)
    parser.add_argument(
        "--nwm-observed-mode",
        default="diffusion_guidance",
        choices=("autoregressive", "multi_horizon", "diffusion_guidance"),
        help="How NWM produces the observed split (between kf0 and kf1).",
    )
    parser.add_argument(
        "--nwm-imagined-mode",
        default="multi_horizon",
        choices=("autoregressive", "multi_horizon"),
        help="How NWM produces imagined splits (kf0->kf1 and kf1->kf2).",
    )

    parser.add_argument(
        "--gen3c-python",
        default=os.environ.get("GEN3C_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--gen3c-repo",
        type=Path,
        default=Path(os.environ.get("GEN3C_REPO", str(REPO_ROOT / "third_party" / "GEN3C"))),
    )
    parser.add_argument("--gen3c-checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--gen3c-cuda-home",
        type=Path,
        default=Path(os.environ.get("GEN3C_CUDA_HOME", os.environ.get("CUDA_HOME", sys.prefix))),
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
    return parser.parse_args()


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_dataset(dataset_root: Path, stage: str, image_size: int, adjacent_angle: float, adjacent_distance: float):
    add_project_paths()
    from splat_belief.data_io.spoc_seq import SPOCDatasetSeq

    return SPOCDatasetSeq(
        root=dataset_root,
        num_context=1,
        num_target=1,
        context_min_distance=15,
        context_max_distance=16,
        stage=stage,
        image_size=image_size,
        adjacent_angle=adjacent_angle,
        adjacent_distance=adjacent_distance,
        language_encoder=None,
        use_depth_supervision=True,
    )


def candidate_starts(num_frames: int, scan_max_frames: int, max_candidates: int, rng: random.Random) -> list[int]:
    if num_frames < 3:
        return []
    max_start = max(0, num_frames - 3)
    anchors = [0]
    if max_start > 0:
        usable = list(range(1, max_start + 1))
        rng.shuffle(usable)
        anchors.extend(usable[: max(0, max_candidates - 1)])
    return [start for start in anchors if start + 2 < num_frames and start < num_frames - 1]


def sample_episodes(dataset: Any, args: argparse.Namespace) -> list[EpisodeSpec]:
    if args.episode:
        return [parse_episode_token(token) for token in args.episode]

    rng = random.Random(args.seed)
    scene_indices = list(range(len(dataset.scene_path_list)))
    rng.shuffle(scene_indices)
    episodes: list[EpisodeSpec] = []

    for scene_idx in scene_indices:
        num_frames = int(dataset.num_frames_per_scene[scene_idx])
        for start_idx in candidate_starts(num_frames, args.scan_max_frames, args.max_start_candidates, rng):
            end_scan = min(num_frames - 1, start_idx + args.scan_max_frames - 1)
            try:
                video_dict, _rgb_frames, actual_start, _actual_end = dataset.data_for_temporal(
                    video_idx=scene_idx,
                    frames_render=[start_idx, end_scan],
                )
            except Exception as exc:  # noqa: BLE001 - keep sampling robust
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
            distance_ok, angle_ok = triplet_is_monotonic(video_dict["render_poses"], kf1_local, kf2_local)
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


def write_ground_truth(dataset: Any, episodes: Sequence[EpisodeSpec], output_root: Path) -> None:
    manifest = []
    for episode in episodes:
        sample = load_episode_sample(dataset, episode)
        episode_root = output_root / "ground_truth" / episode.name
        save_indexed_frames(sample["gt_frames"], episode_root / "frames")
        save_video(episode_root / "gt.mp4", [sample["gt_frames"][i] for i in sorted(sample["gt_frames"])])
        write_json(episode_root / "frame_sets.json", frame_sets_for_episode(episode))
        manifest.append({**episode.to_dict(), "gt_dir": str(episode_root / "frames")})
    write_json(output_root / "ground_truth" / "manifest.json", {"episodes": manifest})


def frame_sets_for_episode(episode: EpisodeSpec) -> dict:
    observed = list(range(episode.local_kf0 + 1, episode.local_kf1))
    imagined_01 = list(range(episode.local_kf0 + 1, episode.local_kf1 + 1))
    imagined_12 = list(range(episode.local_kf1 + 1, episode.local_kf2 + 1))
    return {
        "observed": observed,
        "imagined_kf0_to_kf1": imagined_01,
        "imagined_kf1_to_kf2": imagined_12,
    }


def make_autocast_ctx(dtype_str: str):
    if dtype_str in (None, "", "fp32", "float32"):
        return nullcontext()
    if dtype_str in ("bf16", "bfloat16"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype_str in ("fp16", "float16", "half"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(dtype_str)


class BeliefTemporalRunner:
    def __init__(self, args: argparse.Namespace):
        add_project_paths()
        import hydra
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        overrides = [
            "dataset=spoc_seq",
            f"dataset.root_dir={args.dataset_root}",
            f"stage={args.stage}",
            "batch_size=1",
            "num_target=1",
            "num_context=1",
            "model/encoder=uvitmvsplat",
            "model.encoder.use_image_condition=true",
            "model.encoder.depth_predictor_time_embed=true",
            "model.encoder.evolve_ctxt=false",
            "model.encoder.use_camera_pose=true",
            "model.encoder.use_semantic=true",
            "model.encoder.use_reg_model=true",
            f"model.encoder.d_semantic={args.belief_d_semantic}",
            f"model.encoder.d_semantic_reg={args.belief_d_semantic_reg}",
            f"model.encoder.reg_model_name={args.belief_reg_model_name}",
            "model.encoder.gaussians_per_pixel=1",
            "model.encoder.inference_mode=false",
            # Avoid depth-mask holes in novel-view renders.
            "model.encoder.use_depth_mask=false",
            "model.encoder.grid_sample_disable_cudnn=true",
            f"model.encoder.reg_model_weights={args.dinov3_weights}",
            "model/encoder/backbone=u_vit3d_pose",
            f"model.encoder.backbone.input_size=[{args.image_size},{args.image_size}]",
            "model_type=uvit_pose",
            "semantic_mode=embed",
            "semantic_viz=query",
            f"temperature={args.belief_temperature}",
            f"sampling_steps={args.belief_sampling_steps}",
            f"sampler={args.belief_sampler}",
            f"obj_permanence_mode={args.belief_obj_permanence_mode}",
            f"obj_permanence_observed_mode={args.belief_obj_permanence_observed_mode}",
            "name=vision_metrics_temporal",
            f"image_size={args.image_size}",
            f"adjacent_angle={args.adjacent_angle}",
            f"adjacent_distance={args.adjacent_distance}",
            "clean_target=false",
            "use_history=false",
            f"semantic_config={Path('splat_belief/config/semantic/onehot.yaml')}",
            f"checkpoint_path={args.belief_checkpoint}",
            f"results_folder={DEFAULT_OUTPUT_ROOT / '_tmp_belief'}",
        ]
        with hydra.initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "splat_belief" / "config"), version_base=None):
            cfg = hydra.compose(config_name="config", overrides=overrides)

        from splat_belief.experiment.compare_samplers import _build_model_and_trainer

        self.cfg = cfg
        self.trainer, _dataset = _build_model_and_trainer(cfg)
        self.use_depth_mask = bool(cfg.model.encoder.use_depth_mask)
        self.dtype = args.belief_inference_dtype

    def _make_input(self, sample: Mapping[str, Any], ctxt_idx: int, trgt_idx: int, render_indices: Sequence[int]) -> dict:
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

    def _sample(self, sample: Mapping[str, Any], ctxt_idx: int, trgt_idx: int, render_indices: Sequence[int], state_t: int) -> tuple[dict, float]:
        inp = self._make_input(sample, ctxt_idx, trgt_idx, render_indices)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.time()
        with torch.no_grad(), make_autocast_ctx(self.dtype):
            out = self.trainer.ema.ema_model.sample(batch_size=1, inp=inp, state_t=state_t)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out, time.time() - started

    @staticmethod
    def _frames_from_output(out: Mapping[str, Any]) -> list[np.ndarray]:
        from splat_belief.experiment.temporal_inference import prepare_video_viz

        result = prepare_video_viz(out)
        return list(result[0])

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
            "splits": {},
            "total_diffusion_calls": 0,
            "total_prediction_seconds": 0.0,
            "notes": [
                "3D-Belief uses the temporal_inference validation stack, not wm_baselines world wrappers.",
                "`imagined_kf0_to_kf1` is the state_t=0 prediction (only kf0 observed).",
                "`observed` is re-rendered at state_t=1 (after kf1 has also been observed),",
                "so frames between kf0 and kf1 reflect both endpoints via the accumulated belief.",
            ],
        }

        # State 0: kf0 only.
        render_01 = list(range(0, kf1 + 1))
        out_01, elapsed_01 = self._sample(sample, ctxt_idx=0, trgt_idx=kf1, render_indices=render_01, state_t=0)
        frames_01 = self._frames_from_output(out_01)
        frame_map_01 = {idx: resize_uint8(frame, sample["height_width"]) for idx, frame in zip(render_01, frames_01)}
        trace["total_diffusion_calls"] += 1
        trace["total_prediction_seconds"] += elapsed_01

        imagined_01_frames = {idx: frame_map_01[idx] for idx in frame_sets["imagined_kf0_to_kf1"] if idx in frame_map_01}
        self._write_split(episode_dir, "imagined_kf0_to_kf1", imagined_01_frames, frame_sets["imagined_kf0_to_kf1"])
        trace["splits"]["imagined_kf0_to_kf1"] = {
            "mode": "state_t0_context_kf0_only_predict_kf1",
            "frame_indices": frame_sets["imagined_kf0_to_kf1"],
            "elapsed_seconds": elapsed_01,
            "diffusion_calls": 1,
            "shared_call_id": "belief_state0_kf0_to_kf1",
        }

        # State 1: kf1 observed, state carries kf0.
        render_12 = list(range(0, kf2 + 1))
        out_12, elapsed_12 = self._sample(sample, ctxt_idx=kf1, trgt_idx=kf2, render_indices=render_12, state_t=1)
        frames_12 = self._frames_from_output(out_12)
        frame_map_12 = {idx: resize_uint8(frame, sample["height_width"]) for idx, frame in zip(render_12, frames_12)}
        observed_frames = {idx: frame_map_12[idx] for idx in frame_sets["observed"] if idx in frame_map_12}
        imagined_12_frames = {idx: frame_map_12[idx] for idx in frame_sets["imagined_kf1_to_kf2"] if idx in frame_map_12}
        self._write_split(episode_dir, "observed", observed_frames, frame_sets["observed"])
        self._write_split(episode_dir, "imagined_kf1_to_kf2", imagined_12_frames, frame_sets["imagined_kf1_to_kf2"])
        trace["splits"]["observed"] = {
            "mode": "state_t1_context_kf1_state_carries_kf0_render_between_endpoints",
            "frame_indices": frame_sets["observed"],
            "elapsed_seconds": elapsed_12,
            "diffusion_calls": 1,
            "shared_call_id": "belief_state1_kf1_to_kf2",
        }
        trace["splits"]["imagined_kf1_to_kf2"] = {
            "mode": "state_t1_context_kf1_target_kf2_after_observing_kf0_to_kf1",
            "frame_indices": frame_sets["imagined_kf1_to_kf2"],
            "elapsed_seconds": elapsed_12,
            "diffusion_calls": 1,
            "shared_call_id": "belief_state1_kf1_to_kf2",
        }
        trace["total_diffusion_calls"] += 1
        trace["total_prediction_seconds"] += elapsed_12
        return trace

    @staticmethod
    def _write_split(episode_dir: Path, split: str, frames: Mapping[int, np.ndarray], requested_indices: Sequence[int]) -> None:
        split_dir = episode_dir / split
        save_indexed_frames(frames, split_dir / "frames")
        save_video(split_dir / "prediction.mp4", [frames[i] for i in sorted(frames)])
        write_json(split_dir / "manifest.json", {"split": split, "requested_frame_indices": list(requested_indices), "saved_frame_indices": sorted(frames)})


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


class DFoTRunner:
    def __init__(self, args: argparse.Namespace):
        patch_numpy_legacy_aliases()
        add_project_paths(args.dfot_repo)
        # Compose the full DFoT config before loading the checkpoint.
        import hydra as _hydra
        from hydra.core.global_hydra import GlobalHydra as _GlobalHydra
        if _GlobalHydra.instance().is_initialized():
            _GlobalHydra.instance().clear()
        cfg_overrides = [
            "+name=vision_metrics_dfot",
            "dataset=spoc",
            "algorithm=dfot_video_pose",
            "experiment=video_generation",
            "++algorithm={diffusion:{is_continuous:True,precond_scale:0.125},backbone:{use_fourier_noise_embedding:True}}",
        ]
        with _hydra.initialize_config_dir(config_dir=str((args.dfot_repo / "configurations").resolve()), version_base=None):
            cfg = _hydra.compose(config_name="config", overrides=cfg_overrides)
        from model_wrapper import ModelWrapper as _ModelWrapper
        wrapper = _ModelWrapper(cfg, str(args.dfot_checkpoint))
        self.algo = wrapper.algo.to("cuda").eval()
        self.resolution = int(args.dfot_resolution)
        self.include_kf1_to_kf2 = bool(args.dfot_include_kf1_to_kf2)

    def _video_tensor(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        arr = np.stack([tensor_to_uint8_image(frame) for frame in frames], axis=0)
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", antialias=True)
        x = x.to(self.algo.device)
        return self.algo._normalize_x(x)

    def _conds(self, sample: Mapping[str, Any], indices: Sequence[int], pad_to: int | None = None) -> torch.Tensor:
        fx = 0.390 * self.resolution
        fy = 0.385 * self.resolution
        cx = 0.5 * self.resolution
        cy = 0.5 * self.resolution
        rows = []
        for idx in indices:
            c2w = pose_array(sample["render_poses"][idx])
            w2c = np.linalg.inv(c2w)
            rows.append(np.concatenate([[fx, fy, cx, cy], w2c[:3, :].reshape(-1)], axis=0))
        if pad_to is not None and len(rows) < pad_to:
            rows.extend([rows[-1].copy() for _ in range(pad_to - len(rows))])
        return torch.tensor(np.stack(rows, axis=0), dtype=torch.float32, device=self.algo.device).unsqueeze(0)

    def _uint8_frames(self, video_norm: torch.Tensor, indices: Sequence[int], size_hw: tuple[int, int]) -> dict[int, np.ndarray]:
        video = self.algo._unnormalize_x(video_norm).clamp(0, 1)
        if video.ndim == 5:
            video = video[0]
        frames = {}
        for out_idx, frame_idx in enumerate(indices):
            frame = tensor_to_uint8_image(video[out_idx])
            frames[int(frame_idx)] = resize_uint8(frame, size_hw)
        return frames

    def _interpolate(self, sample: Mapping[str, Any], indices: Sequence[int], known_indices: Sequence[int]) -> tuple[dict[int, np.ndarray], float, int]:
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
            pred_norm = self.algo._interpolate_videos(video_norm, context_mask=mask, conditions=conds[:, : len(indices)])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return self._uint8_frames(pred_norm, indices, sample["height_width"]), time.time() - started, int(counter["calls"])

    def _predict(self, sample: Mapping[str, Any], context_indices: Sequence[int], output_indices: Sequence[int]) -> tuple[dict[int, np.ndarray], float, int, str]:
        gt_frames = sample["gt_frames"]
        all_indices = list(context_indices) + [idx for idx in output_indices if idx not in context_indices]
        context_frames = [gt_frames[idx] for idx in context_indices]
        context = self._video_tensor(context_frames).unsqueeze(0)
        conds = self._conds(sample, all_indices, pad_to=len(all_indices) + self.algo.max_tokens)
        sliding_context_len = self.algo.cfg.tasks.prediction.sliding_context_len or self.algo.max_tokens // 2
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
            pred_norm, _record = self.algo._predict_sequence(
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
        return {idx: all_frames[idx] for idx in output_indices if idx in all_frames}, time.time() - started, int(counter["calls"]), f"sliding_context_len={sliding_context_len}"

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
            "notes": ["Observed split uses DFoT _interpolate_videos with kf0 and kf1 fixed."],
        }

        observed_indices = list(range(0, kf1 + 1))
        observed_frames, elapsed, calls = self._interpolate(sample, observed_indices, known_indices=[0, kf1])
        observed_save = {idx: observed_frames[idx] for idx in frame_sets["observed"] if idx in observed_frames}
        BeliefTemporalRunner._write_split(episode_dir, "observed", observed_save, frame_sets["observed"])
        trace["splits"]["observed"] = {
            "mode": "dfot_native_interpolation_kf0_kf1_fixed",
            "frame_indices": frame_sets["observed"],
            "elapsed_seconds": elapsed,
            "diffusion_calls": calls,
        }
        trace["total_prediction_seconds"] += elapsed
        trace["total_diffusion_calls"] += calls

        imagined_01, elapsed, calls, context_note = self._predict(sample, [0], frame_sets["imagined_kf0_to_kf1"])
        BeliefTemporalRunner._write_split(episode_dir, "imagined_kf0_to_kf1", imagined_01, frame_sets["imagined_kf0_to_kf1"])
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
                imagined_12, elapsed, calls, context_note = self._predict(sample, context_indices, frame_sets["imagined_kf1_to_kf2"])
                BeliefTemporalRunner._write_split(episode_dir, "imagined_kf1_to_kf2", imagined_12, frame_sets["imagined_kf1_to_kf2"])
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
                    "reason": f"DFoT max initial context is {self.algo.max_tokens - 1}; observed context has {len(context_indices)} frames.",
                    "frame_indices": frame_sets["imagined_kf1_to_kf2"],
                    "elapsed_seconds": 0.0,
                    "diffusion_calls": 0,
                }
        return trace


def get_yaw_from_pose(pose: np.ndarray) -> float:
    raw_z_axis = pose[:3, 2]
    forward = -raw_z_axis
    return float(np.arctan2(forward[0], forward[2]))


def angle_difference(theta1: np.ndarray | float, theta2: np.ndarray | float) -> np.ndarray:
    delta = np.asarray(theta2) - np.asarray(theta1)
    return delta - 2 * np.pi * np.floor((delta + np.pi) / (2 * np.pi))


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float64,
    )


def to_local_coords(positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float) -> np.ndarray:
    return (positions - curr_pos).dot(yaw_rotmat(curr_yaw))


def normalize_action_xy(data: np.ndarray) -> np.ndarray:
    stats_min = np.array([[-2.5, -4.0]], dtype=np.float64)
    stats_max = np.array([[5.0, 4.0]], dtype=np.float64)
    return ((data - stats_min) / (stats_max - stats_min)) * 2.0 - 1.0


def get_delta_np(actions: np.ndarray) -> np.ndarray:
    ex_actions = np.concatenate((np.zeros((1, actions.shape[1]), dtype=actions.dtype), actions), axis=0)
    return ex_actions[1:] - ex_actions[:-1]


class NWMRunner:
    def __init__(self, args: argparse.Namespace):
        add_project_paths(args.nwm_repo)
        import yaml

        config_path = args.nwm_repo / "config" / "eval_config.yaml"
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        config["ckpt_path"] = str(args.nwm_checkpoint)
        from isolated_nwm_infer import ModelWrapper

        self.model = ModelWrapper(config=config, device="cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.model.device
        self.context_size = int(config.get("context_size", 4))
        self.image_size = int(config.get("image_size", 224))
        self.num_timesteps = int(args.nwm_num_timesteps)
        self.observed_mode = str(args.nwm_observed_mode)
        self.imagined_mode = str(args.nwm_imagined_mode)

    def _frame_tensor(self, frame: np.ndarray) -> torch.Tensor:
        arr = tensor_to_uint8_image(frame)
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        x = F.interpolate(x.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", antialias=True)[0]
        return x * 2.0 - 1.0

    def _poses_to_actions(self, sample: Mapping[str, Any], indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cumulative and delta NWM actions for a frame path."""
        # Map SPOC poses to NWM's training convention.
        F = np.diag([1.0, 1.0, -1.0, 1.0])
        poses = [F @ pose_array(sample["render_poses"][idx]) @ F for idx in indices]
        yaws = np.array([get_yaw_from_pose(pose) for pose in poses], dtype=np.float64)
        positions = np.array([[pose[2, 3], pose[0, 3]] for pose in poses], dtype=np.float64)
        waypoints_pos = to_local_coords(positions, positions[0], yaws[0])
        waypoints_yaw = angle_difference(yaws[0], yaws)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)[1:]
        actions[:, :2] /= 0.2  # SPOC metric_waypoint_spacing.
        actions[:, :2] = normalize_action_xy(actions[:, :2])
        delta = get_delta_np(actions)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        delta_t = torch.as_tensor(delta, dtype=torch.float32, device=self.device)
        return actions_t, delta_t

    def _poses_to_delta(self, sample: Mapping[str, Any], indices: Sequence[int]) -> torch.Tensor:
        return self._poses_to_actions(sample, indices)[1]

    def _initial_context(self, sample: Mapping[str, Any], context_indices: Sequence[int]) -> torch.Tensor:
        gt_frames = sample["gt_frames"]
        frames = [self._frame_tensor(gt_frames[idx]) for idx in context_indices]
        if len(frames) == 1:
            frames = frames * self.context_size
        elif len(frames) < self.context_size:
            frames = [frames[0]] * (self.context_size - len(frames)) + frames
        else:
            frames = frames[-self.context_size :]
        return torch.stack(frames, dim=0).unsqueeze(0).to(self.device)

    def _rollout_floats(self, sample: Mapping[str, Any], context_indices: Sequence[int], output_indices: Sequence[int]) -> tuple[dict[int, torch.Tensor], int]:
        """Autoregressive rollout returning float tensors in [-1, 1] at the model's native size."""
        if not output_indices:
            return {}, 0
        path_indices = [context_indices[-1]] + list(output_indices)
        _, deltas = self._poses_to_actions(sample, path_indices)
        curr_obs = self._initial_context(sample, context_indices)
        predictions: dict[int, torch.Tensor] = {}
        calls = 0
        with torch.no_grad():
            for step, frame_idx in enumerate(output_indices):
                curr_delta = deltas[step : step + 1].unsqueeze(0)
                pred = self.model.forward(curr_obs, curr_delta, num_timesteps=self.num_timesteps, progress=False)
                calls += 1
                pred_frame = pred[0].detach().clamp(-1, 1)
                predictions[int(frame_idx)] = pred_frame
                curr_obs = torch.cat([curr_obs[:, 1:], pred_frame.unsqueeze(0).unsqueeze(0)], dim=1)
        return predictions, calls

    def _multi_horizon_floats(self, sample: Mapping[str, Any], context_indices: Sequence[int], output_indices: Sequence[int]) -> tuple[dict[int, torch.Tensor], int]:
        """Direct single-shot NWM prediction for each output frame."""
        if not output_indices:
            return {}, 0
        base_idx = context_indices[-1]
        path_indices = [base_idx] + list(output_indices)
        actions, _ = self._poses_to_actions(sample, path_indices)
        curr_obs = self._initial_context(sample, context_indices)
        predictions: dict[int, torch.Tensor] = {}
        calls = 0
        with torch.no_grad():
            for step, frame_idx in enumerate(output_indices):
                horizon = int(frame_idx) - int(base_idx)
                curr_delta = actions[step : step + 1].unsqueeze(0)  # [1,1,3]
                rel_t = torch.tensor([horizon / 128.0], dtype=torch.float32, device=self.device)
                pred = self.model.forward(
                    curr_obs, curr_delta, num_timesteps=horizon, rel_t=rel_t, progress=False
                )
                calls += 1
                predictions[int(frame_idx)] = pred[0].detach().clamp(-1, 1)
        return predictions, calls

    def _diffusion_guidance_floats(
        self,
        sample: Mapping[str, Any],
        kf_a: int,
        kf_b: int,
        output_indices: Sequence[int],
    ) -> tuple[dict[int, torch.Tensor], int]:
        """Endpoint-conditioned NWM prediction via branch blending."""
        if not output_indices:
            return {}, 0
        if kf_b <= kf_a:
            return self._multi_horizon_floats(sample, [kf_a], output_indices)
        span = float(kf_b - kf_a)
        # Forward/backward action targets.
        fwd_path = [kf_a] + list(output_indices)
        bwd_path = [kf_b] + list(output_indices)  # actions are computed in kf_b's egocentric frame
        fwd_actions, _ = self._poses_to_actions(sample, fwd_path)  # [T, 3]
        bwd_actions, _ = self._poses_to_actions(sample, bwd_path)  # [T, 3]

        # Shared branch conditionings.
        fwd_obs = self._initial_context(sample, [kf_a])
        bwd_obs = self._initial_context(sample, [kf_b])
        diffusion = self.model.diffusion
        cdit = self.model.model
        vae = self.model.vae
        latent_size = self.model.latent_size

        predictions: dict[int, torch.Tensor] = {}
        calls = 0
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                fwd_x = vae.encode(fwd_obs.flatten(0, 1)).latent_dist.sample().mul_(0.18215).unflatten(0, (1, self.context_size))
                bwd_x = vae.encode(bwd_obs.flatten(0, 1)).latent_dist.sample().mul_(0.18215).unflatten(0, (1, self.context_size))
                # Keep CDiT x_cond 5D.
                fwd_x_cond = fwd_x[:, : self.context_size]
                bwd_x_cond = bwd_x[:, : self.context_size]

                for step, frame_idx in enumerate(output_indices):
                    idx = int(frame_idx)
                    alpha = (idx - kf_a) / span
                    horizon_fwd = idx - kf_a
                    horizon_bwd = kf_b - idx
                    fwd_y = fwd_actions[step : step + 1]
                    bwd_y = bwd_actions[step : step + 1]
                    fwd_rel_t = torch.tensor([horizon_fwd / 128.0], dtype=torch.float32, device=self.device)
                    bwd_rel_t = torch.tensor([horizon_bwd / 128.0], dtype=torch.float32, device=self.device)
                    fwd_kwargs = dict(y=fwd_y, x_cond=fwd_x_cond, rel_t=fwd_rel_t)
                    bwd_kwargs = dict(y=bwd_y, x_cond=bwd_x_cond, rel_t=bwd_rel_t)

                    z = torch.randn(1, 4, latent_size, latent_size, device=self.device)
                    img = z
                    for i in reversed(range(diffusion.num_timesteps)):
                        t_tensor = torch.tensor([i], device=self.device, dtype=torch.long)
                        out_f = diffusion.p_mean_variance(
                            cdit.forward, img, t_tensor, clip_denoised=False, model_kwargs=fwd_kwargs
                        )
                        out_b = diffusion.p_mean_variance(
                            cdit.forward, img, t_tensor, clip_denoised=False, model_kwargs=bwd_kwargs
                        )
                        calls += 2
                        mean = (1.0 - alpha) * out_f["mean"] + alpha * out_b["mean"]
                        log_var = out_f["log_variance"]
                        if i > 0:
                            noise = torch.randn_like(img)
                            img = mean + torch.exp(0.5 * log_var) * noise
                        else:
                            img = mean
                    sample_pixels = vae.decode(img / 0.18215).sample
                    predictions[idx] = sample_pixels[0].detach().clamp(-1, 1)
        return predictions, calls

    def _generate(
        self,
        sample: Mapping[str, Any],
        mode: str,
        context_indices: Sequence[int],
        output_indices: Sequence[int],
        kf_a: int | None = None,
        kf_b: int | None = None,
    ) -> tuple[dict[int, np.ndarray], float, int, str]:
        if not output_indices:
            return {}, 0.0, 0, "empty output"
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.time()
        if mode == "autoregressive":
            floats, calls = self._rollout_floats(sample, context_indices, output_indices)
            note_extra = "mode=autoregressive"
        elif mode == "multi_horizon":
            floats, calls = self._multi_horizon_floats(sample, context_indices, output_indices)
            note_extra = "mode=multi_horizon"
        elif mode == "diffusion_guidance":
            assert kf_a is not None and kf_b is not None, "diffusion_guidance requires kf_a and kf_b"
            floats, calls = self._diffusion_guidance_floats(sample, kf_a, kf_b, output_indices)
            note_extra = "mode=diffusion_guidance"
        else:
            raise ValueError(f"unknown nwm mode: {mode}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        predictions = {
            idx: resize_uint8(tensor_to_uint8_image(tensor), sample["height_width"])
            for idx, tensor in floats.items()
        }
        note = f"context_size={self.context_size}, num_timesteps={self.num_timesteps}, {note_extra}"
        return predictions, time.time() - started, calls, note

    def predict_episode(self, sample: Mapping[str, Any], episode_dir: Path) -> dict:
        episode = sample["episode"]
        frame_sets = frame_sets_for_episode(episode)
        trace = {
            "model": "nwm",
            "episode": episode.to_dict(),
            "splits": {},
            "total_prediction_seconds": 0.0,
            "total_diffusion_calls": 0,
            "notes": [
                f"observed_mode={self.observed_mode}",
                f"imagined_mode={self.imagined_mode}",
            ],
        }

        observed, elapsed, calls, note = self._generate(
            sample,
            self.observed_mode,
            context_indices=[episode.local_kf0],
            output_indices=frame_sets["observed"],
            kf_a=episode.local_kf0,
            kf_b=episode.local_kf1,
        )
        BeliefTemporalRunner._write_split(episode_dir, "observed", observed, frame_sets["observed"])
        trace["splits"]["observed"] = {
            "mode": f"nwm_{self.observed_mode}",
            "frame_indices": frame_sets["observed"],
            "elapsed_seconds": elapsed,
            "diffusion_calls": calls,
            "context_note": note,
        }
        trace["total_prediction_seconds"] += elapsed
        trace["total_diffusion_calls"] += calls

        imagined_01, elapsed, calls, note = self._generate(
            sample,
            self.imagined_mode,
            context_indices=[episode.local_kf0],
            output_indices=frame_sets["imagined_kf0_to_kf1"],
        )
        BeliefTemporalRunner._write_split(episode_dir, "imagined_kf0_to_kf1", imagined_01, frame_sets["imagined_kf0_to_kf1"])
        trace["splits"]["imagined_kf0_to_kf1"] = {
            "mode": f"nwm_{self.imagined_mode}_context_kf0",
            "frame_indices": frame_sets["imagined_kf0_to_kf1"],
            "elapsed_seconds": elapsed,
            "diffusion_calls": calls,
            "context_note": note,
        }
        trace["total_prediction_seconds"] += elapsed
        trace["total_diffusion_calls"] += calls

        # imagined_kf1_to_kf2: condition on kf1 (and AR tail if mode is autoregressive).
        if self.imagined_mode == "autoregressive":
            context_indices = list(range(max(0, episode.local_kf1 - self.context_size + 1), episode.local_kf1 + 1))
        else:
            context_indices = [episode.local_kf1]
        imagined_12, elapsed, calls, note = self._generate(
            sample,
            self.imagined_mode,
            context_indices=context_indices,
            output_indices=frame_sets["imagined_kf1_to_kf2"],
        )
        BeliefTemporalRunner._write_split(episode_dir, "imagined_kf1_to_kf2", imagined_12, frame_sets["imagined_kf1_to_kf2"])
        trace["splits"]["imagined_kf1_to_kf2"] = {
            "mode": f"nwm_{self.imagined_mode}_context_kf1",
            "frame_indices": frame_sets["imagined_kf1_to_kf2"],
            "elapsed_seconds": elapsed,
            "diffusion_calls": calls,
            "context_note": note,
        }
        trace["total_prediction_seconds"] += elapsed
        trace["total_diffusion_calls"] += calls
        return trace


def build_runner(model_name: str, args: argparse.Namespace):
    if model_name == "3d_belief":
        return BeliefTemporalRunner(args)
    if model_name == "dfot":
        return DFoTRunner(args)
    if model_name == "nwm":
        return NWMRunner(args)
    raise ValueError(model_name)


def run_gen3c_external(output_root: Path, args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    export_cmd = [
        sys.executable,
        str(script_dir / "export_gen3c_inputs.py"),
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
    if args.skip_existing:
        export_cmd.append("--skip-existing")
    else:
        export_cmd.append("--no-skip-existing")
    print("[gen3c] exporting RGB-D/camera packs")
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
    gen3c_env["CUDA_HOME"] = str(args.gen3c_cuda_home.resolve())
    gen3c_env["CONDA_PREFIX"] = str(args.gen3c_cuda_home.resolve())
    gen3c_env["PYTHONPATH"] = f"{args.gen3c_repo.resolve()}:{gen3c_env.get('PYTHONPATH', '')}"
    print(f"[gen3c] running external Gen3C adapter with {args.gen3c_python}")
    subprocess.run(gen3c_cmd, check=True, env=gen3c_env)


def run_predictions_for_model(model_name: str, dataset: Any, episodes: Sequence[EpisodeSpec], output_root: Path, args: argparse.Namespace) -> None:
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
                f"{trace['total_prediction_seconds']:.2f}s, calls={trace['total_diffusion_calls']}"
            )
    finally:
        del runner
        release_cuda_memory()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    add_project_paths()
    args.dataset_root = args.dataset_root.resolve()
    args.stage = resolve_stage(args.dataset_root, args.stage)
    models = normalize_models(args.models)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.run_name or time.strftime("spoc_temporal_vision_%Y%m%d_%H%M%S")
    output_root = (args.output_root or (DEFAULT_OUTPUT_ROOT / run_name)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args.dataset_root, args.stage, args.image_size, args.adjacent_angle, args.adjacent_distance)
    episodes = sample_episodes(dataset, args)
    if not episodes:
        raise RuntimeError("No evaluation episodes were selected.")

    manifest = {
        "run_name": run_name,
        "dataset_root": str(args.dataset_root),
        "stage": args.stage,
        "image_size": args.image_size,
        "adjacent_angle": args.adjacent_angle,
        "adjacent_distance": args.adjacent_distance,
        "episodes": [episode.to_dict() for episode in episodes],
        "models": models,
        "checkpoints": {
            "3d_belief": str(args.belief_checkpoint.resolve()),
            "dfot": str(args.dfot_checkpoint.resolve()),
            "nwm": str(args.nwm_checkpoint.resolve()),
            "gen3c": str((args.gen3c_checkpoint_dir or (args.gen3c_repo / "checkpoints")).resolve()),
        },
        "dry_run": bool(args.dry_run),
        "env": env_snapshot(),
    }
    write_json(output_root / "manifest.json", manifest)
    write_ground_truth(dataset, episodes, output_root)
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
        run_gen3c_external(output_root, args)


if __name__ == "__main__":
    main()
