#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from common import (
    DEFAULT_ADJACENT_ANGLE,
    DEFAULT_ADJACENT_DISTANCE,
    DEFAULT_DATASET_ROOT,
    episode_from_dict,
    frame_sets_for_episode,
    pose_array,
    read_json,
    resolve_stage,
    tensor_to_uint8_image,
    write_json,
)
from run_vision_predictions import build_dataset, load_episode_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export compact RGB-D/camera packs used by the Gen3C vision-only adapter."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--adjacent-angle", type=float, default=None)
    parser.add_argument("--adjacent-distance", type=float, default=None)
    parser.add_argument("--episode", action="append", default=[], help="Episode folder name to export; default is all.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def tensor_hw(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().float()
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        arr = arr.numpy()
    else:
        arr = np.asarray(value)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
    return arr.astype(np.float32, copy=False)


def pixel_intrinsics(k_norm: np.ndarray, height: int, width: int) -> np.ndarray:
    k_px = k_norm.astype(np.float32, copy=True)
    k_px[0, :] *= float(width)
    k_px[1, :] *= float(height)
    k_px[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return k_px


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


def export_episode(dataset: Any, episode_payload: Mapping, run_dir: Path, skip_existing: bool) -> dict:
    episode = episode_from_dict(episode_payload)
    episode_dir = run_dir / "ground_truth" / episode.name
    out_path = episode_dir / "gen3c_inputs.npz"
    meta_path = episode_dir / "gen3c_inputs.json"
    if skip_existing and out_path.exists() and meta_path.exists():
        return {"episode": episode.name, "path": str(out_path), "status": "skipped_existing"}

    sample = load_episode_sample(dataset, episode)
    video_dict = sample["video_dict"]
    gt_frames = sample["gt_frames"]
    frame_indices = sorted(gt_frames)
    if frame_indices != list(range(episode.num_frames)):
        raise RuntimeError(f"{episode.name}: expected dense local frame indices, got {frame_indices}")

    rgb_uint8 = np.stack([tensor_to_uint8_image(gt_frames[idx]) for idx in frame_indices], axis=0)
    height, width = rgb_uint8.shape[1:3]

    depths = []
    masks = []
    for idx in frame_indices:
        depths.append(tensor_hw(video_dict["depth"][idx]))
        masks.append(tensor_hw(video_dict["depth_mask"][idx]) > 0.5)
    depth_hw = np.stack(depths, axis=0).astype(np.float32)
    mask_hw = np.stack(masks, axis=0).astype(np.bool_)

    c2w = np.stack([pose_array(pose) for pose in sample["abs_poses"]], axis=0).astype(np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    k_norm = np.stack([pose_array(k) for k in video_dict["intrinsics"]], axis=0).astype(np.float32)
    k_px = np.stack([pixel_intrinsics(k, height, width) for k in k_norm], axis=0).astype(np.float32)
    local_indices = np.asarray(frame_indices, dtype=np.int32)
    abs_indices = np.asarray([episode.start_idx + idx for idx in frame_indices], dtype=np.int32)

    episode_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rgb_uint8=rgb_uint8,
        depth=depth_hw,
        mask=mask_hw,
        c2w=c2w,
        w2c=w2c,
        K_norm=k_norm,
        K_px=k_px,
        local_indices=local_indices,
        absolute_indices=abs_indices,
    )
    meta = {
        "episode": episode.to_dict(),
        "frame_sets": frame_sets_for_episode(episode),
        "path": str(out_path),
        "image_size_hw": [int(height), int(width)],
        "arrays": {
            "rgb_uint8": list(rgb_uint8.shape),
            "depth": list(depth_hw.shape),
            "mask": list(mask_hw.shape),
            "w2c": list(w2c.shape),
            "K_px": list(k_px.shape),
        },
        "notes": [
            "RGB, depth, masks, poses, and intrinsics are stored at the SPOC evaluation image size.",
            "The Gen3C runner letterboxes these arrays into its inference resolution without stretching.",
        ],
    }
    write_json(meta_path, meta)
    return {"episode": episode.name, "path": str(out_path), "status": "exported"}


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest = read_json(run_dir / "manifest.json")
    dataset_root = (args.dataset_root or Path(manifest.get("dataset_root", DEFAULT_DATASET_ROOT))).resolve()
    stage = resolve_stage(dataset_root, args.stage or manifest.get("stage", "test"))
    image_size = int(args.image_size or manifest.get("image_size", 128))
    adjacent_angle = float(args.adjacent_angle if args.adjacent_angle is not None else manifest.get("adjacent_angle", DEFAULT_ADJACENT_ANGLE))
    adjacent_distance = float(
        args.adjacent_distance if args.adjacent_distance is not None else manifest.get("adjacent_distance", DEFAULT_ADJACENT_DISTANCE)
    )
    episodes = episode_names_filter(manifest["episodes"], args.episode, args.max_episodes)
    if not episodes:
        raise RuntimeError("No episodes selected for Gen3C input export.")

    dataset = build_dataset(dataset_root, stage, image_size, adjacent_angle, adjacent_distance)
    results = [export_episode(dataset, episode, run_dir, args.skip_existing) for episode in episodes]
    exported = sum(1 for item in results if item["status"] == "exported")
    skipped = sum(1 for item in results if item["status"] == "skipped_existing")
    write_json(run_dir / "ground_truth" / "gen3c_export_manifest.json", {"run_dir": str(run_dir), "results": results})
    print(f"Gen3C export complete for {run_dir}: exported={exported}, skipped={skipped}")


if __name__ == "__main__":
    main()
