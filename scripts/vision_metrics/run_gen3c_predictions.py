#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import episode_from_dict, frame_sets_for_episode, read_json, save_indexed_frames, save_video, write_json


GEN3C_FRAME_COUNT = 121
MODEL_NAME = "gen3c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gen3C predictions for an existing temporal vision metrics run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gen3c-repo", type=Path, default=Path("/home/ubuntu/tianmin-neurips/yyin34/codebase/GEN3C"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--filter-points-threshold", type=float, default=0.05)
    parser.add_argument(
        "--missing-depth-policy",
        default="error",
        choices=("error", "moge", "unit"),
        help=(
            "How to handle Gen3C input packs whose seed depth is missing or NaN. "
            "'moge' estimates seed depth with MoGe, intended for RGB-only RE10K."
        ),
    )
    parser.add_argument("--foreground-masking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-diffusion-transformer", action="store_true")
    parser.add_argument("--offload-tokenizer", action="store_true")
    parser.add_argument("--offload-text-encoder-model", action="store_true")
    parser.add_argument("--offload-prompt-upsampler", action="store_true")
    parser.add_argument("--offload-guardrail-models", action="store_true")
    parser.add_argument(
        "--disable-prompt-encoder",
        dest="disable_prompt_encoder",
        action="store_true",
        default=True,
        help="Use Gen3C's dummy T5 embeddings. This is the default for vision-only evaluation.",
    )
    parser.add_argument(
        "--enable-prompt-encoder",
        dest="disable_prompt_encoder",
        action="store_false",
        help="Load the T5 prompt encoder instead of dummy embeddings.",
    )
    parser.add_argument(
        "--strategy",
        default="keyframes",
        choices=("keyframes", "history"),
        help="Seed strategy for imagined_kf1_to_kf2. keyframes uses kf0+kf1; history uses every real frame through kf1.",
    )
    parser.add_argument("--episode", action="append", default=[], help="Episode folder name to run; default is all.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-buffer", action="store_true", help="Save Gen3C rendered cache buffers next to prediction videos.")
    return parser.parse_args()


def add_gen3c_paths(gen3c_repo: Path) -> None:
    repo = gen3c_repo.resolve()
    if not (repo / "cosmos_predict1").is_dir():
        raise FileNotFoundError(f"Could not find Gen3C cosmos_predict1 package under {repo}")
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def build_pipeline_args(args: argparse.Namespace) -> SimpleNamespace:
    checkpoint_dir = args.checkpoint_dir or (args.gen3c_repo / "checkpoints")
    return SimpleNamespace(
        checkpoint_dir=str(checkpoint_dir.resolve()),
        prompt_upsampler_dir="Pixtral-12B",
        disable_prompt_upsampler=True,
        offload_diffusion_transformer=bool(args.offload_diffusion_transformer),
        offload_tokenizer=bool(args.offload_tokenizer),
        offload_text_encoder_model=bool(args.offload_text_encoder_model),
        offload_prompt_upsampler=bool(args.offload_prompt_upsampler),
        offload_guardrail_models=bool(args.offload_guardrail_models),
        disable_guardrail=True,
        disable_prompt_encoder=bool(args.disable_prompt_encoder),
        guidance=float(args.guidance),
        num_steps=int(args.num_steps),
        height=int(args.height),
        width=int(args.width),
        fps=int(args.fps),
        seed=int(args.seed),
        num_gpus=int(args.num_gpus),
        prompt=str(args.prompt),
        negative_prompt=str(args.negative_prompt),
        video_save_folder=str((args.run_dir / "models" / args.model_name / "_gen3c_raw_videos").resolve()),
        video_save_name="",
        filter_points_threshold=float(args.filter_points_threshold),
        foreground_masking=bool(args.foreground_masking),
        save_buffer=bool(args.save_buffer),
        missing_depth_policy=str(args.missing_depth_policy),
    )


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def preflight_gen3c_runtime(pipeline_args: SimpleNamespace) -> None:
    required_modules = {
        "warp": "warp-lang",
        "megatron.core": "megatron-core",
        "transformer_engine": "transformer-engine[pytorch]",
        "apex.multi_tensor_apply": "NVIDIA/apex with --cpp_ext --cuda_ext",
        "amp_C": "NVIDIA/apex CUDA extensions",
        "mediapy": "mediapy",
    }
    if pipeline_args.missing_depth_policy == "moge":
        required_modules["moge.model.v1"] = "MoGe"
    missing_modules = [f"{module} ({package})" for module, package in required_modules.items() if not module_available(module)]

    checkpoint_dir = Path(pipeline_args.checkpoint_dir)
    required_files = [
        checkpoint_dir / "Gen3C-Cosmos-7B" / "model.pt",
        checkpoint_dir / "Cosmos-Tokenize1-CV8x8x8-720p" / "encoder.jit",
        checkpoint_dir / "Cosmos-Tokenize1-CV8x8x8-720p" / "decoder.jit",
        checkpoint_dir / "Cosmos-Tokenize1-CV8x8x8-720p" / "mean_std.pt",
        checkpoint_dir / "Cosmos-Tokenize1-CV8x8x8-720p" / "image_mean_std.pt",
    ]
    if not pipeline_args.disable_prompt_encoder:
        required_files.append(checkpoint_dir / "google-t5" / "t5-11b" / "config.json")
    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_modules or missing_files:
        parts = []
        if missing_modules:
            parts.append("missing Python modules: " + ", ".join(missing_modules))
        if missing_files:
            parts.append("missing checkpoint files: " + ", ".join(missing_files))
        raise RuntimeError(
            "Gen3C runtime is not ready for inference. "
            + "; ".join(parts)
            + ". Follow GEN3C/INSTALL.md and download checkpoints with "
            "`CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_gen3c_checkpoints.py --checkpoint_dir checkpoints`."
        )


def episode_names_filter(episodes: Sequence[dict], names: Sequence[str], max_episodes: int | None) -> list[dict]:
    if names:
        wanted = set(names)
        episodes = [episode for episode in episodes if episode["name"] in wanted]
        missing = sorted(wanted - {episode["name"] for episode in episodes})
        if missing:
            raise ValueError(f"Requested episodes not present in manifest: {missing}")
    if max_episodes is not None:
        episodes = list(episodes)[: max(0, int(max_episodes))]
    return list(episodes)


def letterbox_rgb(rgb: np.ndarray, height: int, width: int) -> tuple[np.ndarray, dict]:
    src_h, src_w = rgb.shape[:2]
    scale = min(width / float(src_w), height / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    resized = np.asarray(Image.fromarray(rgb).resize((new_w, new_h), Image.BICUBIC))
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, {"src_h": src_h, "src_w": src_w, "scale": scale, "left": left, "top": top, "new_h": new_h, "new_w": new_w}


def letterbox_scalar(values: np.ndarray, meta: Mapping, fill_value: float, mode: str) -> np.ndarray:
    src = torch.from_numpy(values.astype(np.float32))[None, None]
    interp_mode = "nearest" if mode == "nearest" else "bilinear"
    kwargs = {} if interp_mode == "nearest" else {"align_corners": False}
    resized = F.interpolate(src, size=(int(meta["new_h"]), int(meta["new_w"])), mode=interp_mode, **kwargs)[0, 0].numpy()
    out = np.full((int(meta["target_h"]), int(meta["target_w"])), fill_value, dtype=np.float32)
    top = int(meta["top"])
    left = int(meta["left"])
    out[top : top + int(meta["new_h"]), left : left + int(meta["new_w"])] = resized
    return out


def adjust_intrinsics_for_letterbox(k_px: np.ndarray, meta: Mapping) -> np.ndarray:
    k = k_px.astype(np.float32, copy=True)
    k[0, 0] *= float(meta["scale"])
    k[1, 1] *= float(meta["scale"])
    k[0, 2] = k[0, 2] * float(meta["scale"]) + float(meta["left"])
    k[1, 2] = k[1, 2] * float(meta["scale"]) + float(meta["top"])
    k[2] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return k


def crop_from_letterbox(rgb: np.ndarray, meta: Mapping, output_hw: tuple[int, int]) -> np.ndarray:
    top = int(meta["top"])
    left = int(meta["left"])
    cropped = rgb[top : top + int(meta["new_h"]), left : left + int(meta["new_w"])]
    out_h, out_w = output_hw
    if cropped.shape[:2] != (out_h, out_w):
        cropped = np.asarray(Image.fromarray(cropped).resize((out_w, out_h), Image.BICUBIC))
    return np.ascontiguousarray(cropped)


def preprocess_episode_pack(npz_path: Path, height: int, width: int) -> dict:
    data = np.load(npz_path)
    rgb_src = data["rgb_uint8"]
    depth_src = data["depth"]
    mask_src = data["mask"]
    k_src = data["K_px"]
    target_h, target_w = int(height), int(width)

    rgb_frames = []
    depth_frames = []
    mask_frames = []
    k_frames = []
    meta = None
    for idx in range(rgb_src.shape[0]):
        rgb, item_meta = letterbox_rgb(rgb_src[idx], target_h, target_w)
        item_meta["target_h"] = target_h
        item_meta["target_w"] = target_w
        if meta is None:
            meta = item_meta
        rgb_frames.append(rgb)
        depth_frames.append(letterbox_scalar(depth_src[idx], item_meta, fill_value=100.0, mode="bilinear"))
        mask_frames.append(letterbox_scalar(mask_src[idx].astype(np.float32), item_meta, fill_value=0.0, mode="nearest") > 0.5)
        k_frames.append(adjust_intrinsics_for_letterbox(k_src[idx], item_meta))

    return {
        "rgb_uint8": np.stack(rgb_frames, axis=0),
        "depth": np.stack(depth_frames, axis=0).astype(np.float32),
        "mask": np.stack(mask_frames, axis=0).astype(np.bool_),
        "w2c": data["w2c"].astype(np.float32),
        "K": np.stack(k_frames, axis=0).astype(np.float32),
        "letterbox": meta,
        "output_hw": tuple(int(v) for v in rgb_src.shape[1:3]),
    }


def pad_indices(indices: Sequence[int], frame_count: int = GEN3C_FRAME_COUNT) -> list[int]:
    if not indices:
        raise ValueError("Gen3C target index list cannot be empty")
    if len(indices) > frame_count:
        raise ValueError(f"Gen3C supports {frame_count} frames per request, got {len(indices)}")
    return list(indices) + [int(indices[-1])] * (frame_count - len(indices))


def split_specs(episode, strategy: str) -> dict[str, dict]:
    kf0 = episode.local_kf0
    kf1 = episode.local_kf1
    kf2 = episode.local_kf2
    frame_sets = frame_sets_for_episode(episode)
    history_seed = list(range(kf0, kf1 + 1)) if strategy == "history" else [kf0, kf1]
    return {
        "observed": {
            "seed_indices": [kf0, kf1],
            "target_indices": list(range(kf0, kf1 + 1)),
            "save_indices": frame_sets["observed"],
            "condition_index": kf0,
            "mode": "gen3c_sparse_keyframes_kf0_kf1_interpolation",
        },
        "imagined_kf0_to_kf1": {
            "seed_indices": [kf0],
            "target_indices": list(range(kf0, kf1 + 1)),
            "save_indices": frame_sets["imagined_kf0_to_kf1"],
            "condition_index": kf0,
            "mode": "gen3c_single_seed_kf0_exact_camera_path",
        },
        "imagined_kf1_to_kf2": {
            "seed_indices": history_seed,
            "target_indices": list(range(kf1, kf2 + 1)),
            "save_indices": frame_sets["imagined_kf1_to_kf2"],
            "condition_index": kf1,
            "mode": f"gen3c_{strategy}_seed_to_kf1_exact_camera_path",
        },
    }


class Gen3CVisionRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        add_gen3c_paths(args.gen3c_repo)
        self.pipeline_args = build_pipeline_args(args)
        preflight_gen3c_runtime(self.pipeline_args)

        from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_BufferSelector
        from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
        from cosmos_predict1.utils import misc

        misc.set_random_seed(self.pipeline_args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.pipeline_args.num_gpus > 1:
            from megatron.core import parallel_state
            from cosmos_predict1.utils import distributed

            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=self.pipeline_args.num_gpus)
            self.process_group = parallel_state.get_context_parallel_group()
        else:
            self.process_group = None

        self.pipeline = Gen3cPipeline(
            inference_type="video2world",
            checkpoint_dir=self.pipeline_args.checkpoint_dir,
            checkpoint_name="Gen3C-Cosmos-7B",
            prompt_upsampler_dir=self.pipeline_args.prompt_upsampler_dir,
            enable_prompt_upsampler=False,
            offload_network=self.pipeline_args.offload_diffusion_transformer,
            offload_tokenizer=self.pipeline_args.offload_tokenizer,
            offload_text_encoder_model=self.pipeline_args.offload_text_encoder_model,
            offload_prompt_upsampler=self.pipeline_args.offload_prompt_upsampler,
            offload_guardrail_models=self.pipeline_args.offload_guardrail_models,
            disable_guardrail=True,
            disable_prompt_encoder=self.pipeline_args.disable_prompt_encoder,
            guidance=self.pipeline_args.guidance,
            num_steps=self.pipeline_args.num_steps,
            height=self.pipeline_args.height,
            width=self.pipeline_args.width,
            fps=self.pipeline_args.fps,
            num_video_frames=GEN3C_FRAME_COUNT,
            seed=self.pipeline_args.seed,
        )
        if self.pipeline_args.num_gpus > 1:
            self.pipeline.model.net.enable_context_parallel(self.process_group)
        self.cache_cls = Cache3D_BufferSelector
        self.device = device
        self.frame_buffer_max = int(self.pipeline.model.frame_buffer_max)
        self.moge_model = None
        if self.pipeline_args.missing_depth_policy == "moge":
            from moge.model.v1 import MoGeModel

            print("[gen3c] loading MoGe depth model for RGB-only seed depths")
            self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()

    def close(self) -> None:
        if self.pipeline_args.num_gpus > 1:
            from megatron.core import parallel_state
            import torch.distributed as dist

            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()

    def estimate_seed_depths_moge(self, rgb_uint8: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if self.moge_model is None:
            raise RuntimeError("MoGe depth policy requested but MoGe model is not loaded.")
        depths = []
        masks = []
        with torch.no_grad():
            for rgb in rgb_uint8:
                image_chw = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)
                output = self.moge_model.infer(image_chw)
                depth = output["depth"].float()
                mask = output["mask"].bool()
                if depth.shape != image_chw.shape[-2:]:
                    depth = F.interpolate(depth[None, None], size=image_chw.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                    mask = F.interpolate(mask.float()[None, None], size=image_chw.shape[-2:], mode="nearest")[0, 0] > 0.5
                depth = torch.nan_to_num(depth, nan=1e4)
                depth = torch.clamp(depth, min=0.0, max=1e4)
                depth = torch.where(mask, depth, torch.full_like(depth, 1000.0))
                depths.append(depth)
                masks.append(mask.float())
        return torch.stack(depths, dim=0), torch.stack(masks, dim=0)

    def resolve_seed_depths(self, pack: Mapping, seed_indices: Sequence[int], seed_rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        depth_np = pack["depth"][list(seed_indices)].astype(np.float32)
        mask_np = pack["mask"][list(seed_indices)].astype(np.float32)
        valid_depth = np.isfinite(depth_np).all() and bool((mask_np > 0.5).any())
        if valid_depth:
            return (
                torch.from_numpy(depth_np).to(self.device),
                torch.from_numpy(mask_np).to(self.device),
            )
        if self.pipeline_args.missing_depth_policy == "moge":
            return self.estimate_seed_depths_moge(seed_rgb)
        if self.pipeline_args.missing_depth_policy == "unit":
            depth = torch.ones((len(seed_indices), *seed_rgb.shape[1:3]), dtype=torch.float32, device=self.device)
            mask = torch.ones_like(depth)
            return depth, mask
        raise RuntimeError(
            "Gen3C input pack has missing/invalid depth for seed frames. "
            "Use --missing-depth-policy moge for RGB-only datasets such as RE10K."
        )

    def tensor_seed_pack(self, pack: Mapping, seed_indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = pack["rgb_uint8"][list(seed_indices)].astype(np.float32) / 255.0
        depth_t, mask_t = self.resolve_seed_depths(pack, seed_indices, pack["rgb_uint8"][list(seed_indices)])
        w2c = pack["w2c"][list(seed_indices)].astype(np.float32)
        k = pack["K"][list(seed_indices)].astype(np.float32)
        image_bNCHW = torch.from_numpy(rgb).permute(0, 3, 1, 2).unsqueeze(0).to(self.device) * 2.0 - 1.0
        depth_bN1HW = depth_t[:, None].unsqueeze(0)
        mask_bN1HW = mask_t[:, None].unsqueeze(0)
        w2c_bN44 = torch.from_numpy(w2c).unsqueeze(0).to(self.device)
        k_bN33 = torch.from_numpy(k).unsqueeze(0).to(self.device)
        return image_bNCHW, depth_bN1HW, mask_bN1HW, w2c_bN44, k_bN33

    def generate_split(self, pack: Mapping, spec: Mapping) -> tuple[dict[int, np.ndarray], float, dict]:
        seed_indices = list(spec["seed_indices"])
        target_indices = list(spec["target_indices"])
        padded = pad_indices(target_indices)
        image_bNCHW, depth_bN1HW, mask_bN1HW, w2c_bN44, k_bN33 = self.tensor_seed_pack(pack, seed_indices)
        cache = self.cache_cls(
            frame_buffer_max=self.frame_buffer_max,
            input_image=image_bNCHW,
            input_depth=depth_bN1HW,
            input_mask=mask_bN1HW,
            input_w2c=w2c_bN44,
            input_intrinsics=k_bN33,
            filter_points_threshold=self.pipeline_args.filter_points_threshold,
            input_format=["B", "N", "C", "H", "W"],
            foreground_masking=self.pipeline_args.foreground_masking,
        )
        target_w2c = torch.from_numpy(pack["w2c"][padded]).unsqueeze(0).to(self.device)
        target_k = torch.from_numpy(pack["K"][padded]).unsqueeze(0).to(self.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.time()
        with torch.no_grad():
            rendered_warp_images, rendered_warp_masks = cache.render_cache(target_w2c, target_k)
            condition_idx = int(spec["condition_index"])
            condition_rgb = pack["rgb_uint8"][condition_idx].astype(np.float32) / 255.0
            condition = torch.from_numpy(condition_rgb).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(self.device)
            condition = condition * 2.0 - 1.0
            generated = self.pipeline.generate(
                prompt=self.pipeline_args.prompt,
                image_path=condition,
                negative_prompt=self.pipeline_args.negative_prompt,
                rendered_warp_images=rendered_warp_images,
                rendered_warp_masks=rendered_warp_masks,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - started
        if generated is None:
            raise RuntimeError("Gen3C generation was blocked or returned None")
        video, _prompt = generated
        pos_by_idx = {int(frame_idx): pos for pos, frame_idx in enumerate(target_indices)}
        frames = {}
        for frame_idx in spec["save_indices"]:
            if frame_idx not in pos_by_idx:
                continue
            raw = np.asarray(video[pos_by_idx[int(frame_idx)]], dtype=np.uint8)
            frames[int(frame_idx)] = crop_from_letterbox(raw, pack["letterbox"], pack["output_hw"])
        info = {
            "mode": spec["mode"],
            "seed_indices": seed_indices,
            "target_indices": target_indices,
            "padded_num_frames": len(padded),
            "gen3c_generate_calls": 1,
            "gen3c_num_steps": self.pipeline_args.num_steps,
            "gen3c_frame_buffer_max": self.frame_buffer_max,
        }
        return frames, elapsed, info

    def predict_episode(self, episode_payload: Mapping, episode_dir: Path) -> dict:
        episode = episode_from_dict(episode_payload)
        npz_path = self.args.run_dir / "ground_truth" / episode.name / "gen3c_inputs.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing Gen3C input pack: {npz_path}. Run export_gen3c_inputs.py first.")
        pack = preprocess_episode_pack(npz_path, self.pipeline_args.height, self.pipeline_args.width)
        trace = {
            "model": self.args.model_name,
            "episode": episode.to_dict(),
            "strategy": self.args.strategy,
            "splits": {},
            "total_prediction_seconds": 0.0,
            "total_diffusion_calls": 0,
            "notes": [
                "Gen3C is run vision-only from exported RGB/camera packs.",
                f"missing_depth_policy={self.pipeline_args.missing_depth_policy}",
                "Input frames are letterboxed into Gen3C resolution, then cropped back before metric scoring.",
                "diffusion_calls counts Gen3C video generation calls, not internal denoising steps.",
            ],
        }
        for split, spec in split_specs(episode, self.args.strategy).items():
            frames, elapsed, info = self.generate_split(pack, spec)
            split_dir = episode_dir / split
            save_indexed_frames(frames, split_dir / "frames")
            save_video(split_dir / "prediction.mp4", [frames[i] for i in sorted(frames)], fps=self.pipeline_args.fps)
            write_json(
                split_dir / "manifest.json",
                {
                    "split": split,
                    "requested_frame_indices": list(spec["save_indices"]),
                    "saved_frame_indices": sorted(frames),
                    **info,
                },
            )
            trace["splits"][split] = {
                "frame_indices": list(spec["save_indices"]),
                "elapsed_seconds": elapsed,
                "diffusion_calls": 1,
                **info,
            }
            trace["total_prediction_seconds"] += elapsed
            trace["total_diffusion_calls"] += 1
        return trace


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.gen3c_repo = args.gen3c_repo.resolve()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    manifest = read_json(args.run_dir / "manifest.json")
    episodes = episode_names_filter(manifest["episodes"], args.episode, args.max_episodes)
    if not episodes:
        raise RuntimeError("No episodes selected for Gen3C prediction.")
    runner = Gen3CVisionRunner(args)
    model_root = args.run_dir / "models" / args.model_name
    model_root.mkdir(parents=True, exist_ok=True)
    load_seconds = 0.0
    try:
        for episode in episodes:
            episode_dir = model_root / episode["name"]
            if args.skip_existing and (episode_dir / "trace.json").exists():
                print(f"[{args.model_name}] skipping existing {episode['name']}")
                continue
            print(f"[{args.model_name}] episode {episode['name']}")
            started = time.time()
            trace = runner.predict_episode(episode, episode_dir)
            trace["load_seconds"] = load_seconds
            trace["wall_seconds"] = time.time() - started
            write_json(episode_dir / "trace.json", trace)
            print(
                f"[{args.model_name}] {episode['name']}: "
                f"{trace['total_prediction_seconds']:.2f}s, calls={trace['total_diffusion_calls']}"
            )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
