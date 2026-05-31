from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLAT_ROOT = REPO_ROOT / "splat_belief"
DEFAULT_DATASET_ROOT = Path(os.environ.get("SPOC_DATASET_ROOT", str(REPO_ROOT / "data" / "spoc")))
DEFAULT_CHECKPOINT_ROOT = Path(os.environ.get("CHECKPOINT_ROOT", str(REPO_ROOT / "checkpoints")))
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "vision_metrics"

DEFAULT_ADJACENT_ANGLE = 0.523
DEFAULT_ADJACENT_DISTANCE = 1.0


@dataclass(frozen=True)
class EpisodeSpec:
    scene_idx: int
    start_idx: int
    end_idx: int
    kf0_idx: int
    kf1_idx: int
    kf2_idx: int
    key_frame_indices: tuple[int, ...] = field(default_factory=tuple)
    monotonic_distance: bool = True
    monotonic_angle: bool = True

    @property
    def local_kf0(self) -> int:
        return self.kf0_idx - self.start_idx

    @property
    def local_kf1(self) -> int:
        return self.kf1_idx - self.start_idx

    @property
    def local_kf2(self) -> int:
        return self.kf2_idx - self.start_idx

    @property
    def num_frames(self) -> int:
        return self.end_idx - self.start_idx + 1

    @property
    def name(self) -> str:
        return (
            f"episode_{self.scene_idx:06d}_"
            f"{self.kf0_idx:06d}_{self.kf1_idx:06d}_{self.kf2_idx:06d}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scene_idx": self.scene_idx,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "num_frames": self.num_frames,
            "kf0_idx": self.kf0_idx,
            "kf1_idx": self.kf1_idx,
            "kf2_idx": self.kf2_idx,
            "local_kf0": self.local_kf0,
            "local_kf1": self.local_kf1,
            "local_kf2": self.local_kf2,
            "key_frame_indices": list(self.key_frame_indices),
            "monotonic_distance": self.monotonic_distance,
            "monotonic_angle": self.monotonic_angle,
        }


def add_project_paths(*extra_paths: Path | str) -> None:
    """Make repo-local modules importable from standalone scripts."""
    for path in (REPO_ROOT, SPLAT_ROOT, *extra_paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def patch_numpy_legacy_aliases() -> None:
    """Restore NumPy 1.x aliases needed by older third-party model code."""
    if "sctypes" not in np.__dict__:
        np.sctypes = {  # type: ignore[attr-defined]
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.bytes_, np.str_, np.void, np.object_],
        }
    for name, value in {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "float_": np.float64,
        "complex_": np.complex128,
        "int_": np.int64,
        "unicode_": np.str_,
        "string_": np.bytes_,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def resolve_stage(dataset_root: Path, requested_stage: str) -> str:
    if requested_stage != "auto":
        return requested_stage
    for candidate in ("test", "unit", "train"):
        if (dataset_root / candidate).is_dir():
            return candidate
    raise FileNotFoundError(f"No split directory found under {dataset_root}")


def env_snapshot() -> dict:
    return {
        "cwd": os.getcwd(),
        "python": sys.executable,
        "repo_root": str(REPO_ROOT),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def tensor_to_uint8_image(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().float()
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.permute(1, 2, 0)
        arr = arr.numpy()
    else:
        arr = np.asarray(value)

    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        if float(np.nanmax(arr)) <= 1.5 and float(np.nanmin(arr)) >= -0.01:
            arr = arr * 255.0
        elif float(np.nanmin(arr)) >= -1.01 and float(np.nanmax(arr)) <= 1.01:
            arr = (arr + 1.0) * 127.5
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def resize_uint8(frame: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    if frame.shape[:2] == (h, w):
        return np.ascontiguousarray(frame)
    return np.asarray(Image.fromarray(frame).resize((w, h), Image.BILINEAR))


def save_indexed_frames(frames: Mapping[int, np.ndarray], folder: Path, prefix: str = "frame") -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx in sorted(frames):
        path = folder / f"{prefix}_{idx:04d}.png"
        Image.fromarray(tensor_to_uint8_image(frames[idx])).save(path)
        paths.append(str(path))
    return paths


def save_video(path: Path, frames: Sequence[np.ndarray], fps: int = 10) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, [tensor_to_uint8_image(frame) for frame in frames], fps=fps, quality=10)


def pose_array(pose: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(pose, torch.Tensor):
        arr = pose.detach().cpu().numpy()
    else:
        arr = np.asarray(pose)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.float64, copy=False)


def rotation_angle(initial_direction, target_direction) -> float:
    a = np.asarray(initial_direction, dtype=np.float64)
    b = np.asarray(target_direction, dtype=np.float64)
    denom = max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
    cos_angle = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.arccos(cos_angle))


def compute_key_frame_indices(
    render_poses: Sequence[torch.Tensor | np.ndarray],
    adjacent_angle: float,
    adjacent_distance: float,
) -> list[int]:
    """Mirror temporal_inference.py's adjacent key-frame selection."""
    if len(render_poses) < 2:
        return []

    first_pose = pose_array(render_poses[0])
    z_previous = first_pose[:, 2][:3]
    t_previous = first_pose[:, 3][:3]
    key_frame_indices: list[int] = []

    for idx in range(1, len(render_poses)):
        current_pose = pose_array(render_poses[idx])
        z_idx = current_pose[:, 2][:3]
        t_idx = current_pose[:, 3][:3]
        angle = rotation_angle(z_previous, z_idx)
        distance = float(np.linalg.norm(t_idx - t_previous))
        if angle > adjacent_angle or distance > adjacent_distance or idx == len(render_poses) - 1:
            key_frame_indices.append(idx)
            z_previous = z_idx
            t_previous = t_idx
    return key_frame_indices


def triplet_is_monotonic(render_poses: Sequence[torch.Tensor | np.ndarray], kf1: int, kf2: int) -> tuple[bool, bool]:
    pose0 = pose_array(render_poses[0])
    pose1 = pose_array(render_poses[kf1])
    pose2 = pose_array(render_poses[kf2])

    t0 = pose0[:, 3][:3]
    t1 = pose1[:, 3][:3]
    t2 = pose2[:, 3][:3]
    z0 = pose0[:, 2][:3]
    z1 = pose1[:, 2][:3]
    z2 = pose2[:, 2][:3]

    d01 = float(np.linalg.norm(t1 - t0))
    d02 = float(np.linalg.norm(t2 - t0))
    a01 = rotation_angle(z0, z1)
    a02 = rotation_angle(z0, z2)

    distance_ok = d02 + 1e-6 >= d01
    angle_ok = a02 + 1e-6 >= a01
    if d01 > 1e-6 and np.linalg.norm(t2 - t1) > 1e-6:
        distance_ok = distance_ok and float(np.dot(t1 - t0, t2 - t1)) >= -1e-6
    return distance_ok, angle_ok


def parse_episode_token(token: str) -> EpisodeSpec:
    parts = [int(p) for p in token.replace(":", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(
            "Episode token must be scene,start,kf1,kf2; "
            f"got {token!r}"
        )
    scene_idx, start_idx, kf1_idx, kf2_idx = parts
    if not (start_idx < kf1_idx < kf2_idx):
        raise ValueError(f"Expected start < kf1 < kf2, got {token!r}")
    return EpisodeSpec(
        scene_idx=scene_idx,
        start_idx=start_idx,
        end_idx=kf2_idx,
        kf0_idx=start_idx,
        kf1_idx=kf1_idx,
        kf2_idx=kf2_idx,
        key_frame_indices=(kf1_idx - start_idx, kf2_idx - start_idx),
    )


def episode_from_dict(payload: Mapping) -> EpisodeSpec:
    return EpisodeSpec(
        scene_idx=int(payload["scene_idx"]),
        start_idx=int(payload["start_idx"]),
        end_idx=int(payload["end_idx"]),
        kf0_idx=int(payload["kf0_idx"]),
        kf1_idx=int(payload["kf1_idx"]),
        kf2_idx=int(payload["kf2_idx"]),
        key_frame_indices=tuple(int(i) for i in payload.get("key_frame_indices", ())),
        monotonic_distance=bool(payload.get("monotonic_distance", True)),
        monotonic_angle=bool(payload.get("monotonic_angle", True)),
    )


def frame_sets_for_episode(episode: EpisodeSpec) -> dict[str, list[int]]:
    observed = list(range(episode.local_kf0 + 1, episode.local_kf1))
    imagined_01 = list(range(episode.local_kf0 + 1, episode.local_kf1 + 1))
    imagined_12 = list(range(episode.local_kf1 + 1, episode.local_kf2 + 1))
    return {
        "observed": observed,
        "imagined_kf0_to_kf1": imagined_01,
        "imagined_kf1_to_kf2": imagined_12,
    }


def normalize_models(raw: str) -> list[str]:
    aliases = {
        "all": "3d_belief,nwm,dfot",
        "3d-belief": "3d_belief",
        "3d_belief": "3d_belief",
        "belief": "3d_belief",
        "nwm": "nwm",
        "dfot": "dfot",
        "gen3c": "gen3c",
        "gen-3c": "gen3c",
    }
    models: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "all":
            for model in normalize_models(aliases[token]):
                if model not in models:
                    models.append(model)
            continue
        if token not in aliases:
            raise ValueError(f"Unknown model {token!r}; choose 3d_belief,nwm,dfot,gen3c,all")
        model = aliases[token]
        if model not in models:
            models.append(model)
    return models


def require_cuda(model_names: Iterable[str]) -> None:
    if torch.cuda.is_available():
        return
    names = ", ".join(model_names)
    raise RuntimeError(
        f"CUDA is required for model prediction with {names}. "
        "Use --dry-run for manifest/ground-truth plumbing."
    )
