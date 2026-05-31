#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg

from common import REPO_ROOT, patch_numpy_legacy_aliases, read_json, write_json


IMAGINED_SPLITS = ("imagined_kf0_to_kf1", "imagined_kf1_to_kf2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute observed and imagined vision metrics for temporal prediction traces.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fvd-clip-length", type=int, default=8)
    parser.add_argument("--fvd-clip-stride", type=int, default=1)
    parser.add_argument(
        "--fvd-backbone",
        default="dfot_i3d",
        choices=("dfot_i3d", "torchmetrics", "r3d_18_kinetics", "r3d_18_random", "none"),
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_models(run_dir: Path, requested: str) -> list[str]:
    model_root = run_dir / "models"
    if requested == "all":
        return sorted([p.name for p in model_root.iterdir() if p.is_dir()]) if model_root.exists() else []
    return [token.strip() for token in requested.split(",") if token.strip()]


def frame_paths(folder: Path) -> list[Path]:
    return sorted(folder.glob("frame_*.png"))


def frame_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_image(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def resize_like(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == gt.shape[-2:]:
        return pred
    return F.interpolate(pred.unsqueeze(0), size=gt.shape[-2:], mode="bilinear", align_corners=False)[0]


def load_pairs(gt_dir: Path, pred_dir: Path) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    pred_by_name = {path.name: path for path in frame_paths(pred_dir)}
    pairs = []
    for gt_path in frame_paths(gt_dir):
        pred_path = pred_by_name.get(gt_path.name)
        if pred_path is None:
            continue
        gt = load_image(gt_path)
        pred = resize_like(load_image(pred_path), gt).clamp(0, 1)
        pairs.append((frame_index(gt_path), gt, pred))
    return pairs


def compute_observed_pair_metrics(pairs_by_key: dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]], device: torch.device):
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    lpips_metric.eval()

    rows = []
    values = {"psnr": [], "ssim": [], "lpips": []}
    per_episode = {}
    with torch.no_grad():
        for episode_name, pairs in pairs_by_key.items():
            episode_values = {"psnr": [], "ssim": [], "lpips": []}
            for frame_idx, gt, pred in pairs:
                gt_b = gt.unsqueeze(0).to(device)
                pred_b = pred.unsqueeze(0).to(device)
                vals = {
                    "psnr": float(psnr(pred_b, gt_b).detach().cpu()),
                    "ssim": float(ssim(pred_b, gt_b).detach().cpu()),
                    "lpips": float(lpips_metric(pred_b, gt_b).detach().cpu()),
                }
                rows.append({"episode": episode_name, "split": "observed", "frame_idx": frame_idx, **vals})
                for key, value in vals.items():
                    values[key].append(value)
                    episode_values[key].append(value)
            per_episode[episode_name] = {
                f"{key}_mean": float(np.mean(vals)) if vals else math.nan
                for key, vals in episode_values.items()
            }
            per_episode[episode_name]["num_frames"] = len(pairs)
    overall = {
        f"{key}_mean": float(np.mean(vals)) if vals else math.nan
        for key, vals in values.items()
    }
    overall["num_frames"] = int(sum(len(pairs) for pairs in pairs_by_key.values()))
    return rows, per_episode, overall


def update_fid(pairs: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> tuple[float, str | None]:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        count = 0
        for gt, pred in pairs:
            metric.update(gt.unsqueeze(0).to(device), real=True)
            metric.update(pred.unsqueeze(0).to(device), real=False)
            count += 1
        if count < 2:
            return math.nan, "at least two frame pairs are required for FID"
        return float(metric.compute().detach().cpu()), None
    except Exception as exc:  # noqa: BLE001
        return math.nan, repr(exc)


def split_video_clips(video: torch.Tensor, clip_length: int, stride: int) -> list[torch.Tensor]:
    if clip_length <= 0 or video.shape[1] <= clip_length:
        return [video]
    starts = range(0, video.shape[1] - clip_length + 1, max(1, stride))
    clips = [video[:, start : start + clip_length] for start in starts]
    return clips or [video]


def collect_imagined_videos(
    run_dir: Path,
    model: str,
    episodes: Sequence[dict],
    fvd_clip_length: int,
    fvd_clip_stride: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]], dict[str, int]]:
    real_videos = []
    fake_videos = []
    fid_pairs = []
    split_counts = {split: 0 for split in IMAGINED_SPLITS}
    for episode in episodes:
        episode_name = episode["name"]
        gt_dir = run_dir / "ground_truth" / episode_name / "frames"
        for split in IMAGINED_SPLITS:
            pred_dir = run_dir / "models" / model / episode_name / split / "frames"
            pairs = load_pairs(gt_dir, pred_dir)
            if not pairs:
                continue
            split_counts[split] += 1
            fid_pairs.extend([(gt, pred) for _idx, gt, pred in pairs])
            real = torch.stack([gt for _idx, gt, _pred in pairs], dim=1)
            fake = torch.stack([pred for _idx, _gt, pred in pairs], dim=1)
            real_videos.extend(split_video_clips(real, fvd_clip_length, fvd_clip_stride))
            fake_videos.extend(split_video_clips(fake, fvd_clip_length, fvd_clip_stride))
    return real_videos, fake_videos, fid_pairs, split_counts


def frechet_distance(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
        raise ValueError("at least two feature samples are required for a Frechet distance")
    mu1, mu2 = real_features.mean(axis=0), fake_features.mean(axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)
    if sigma1.ndim == 0:
        sigma1 = np.eye(real_features.shape[1]) * float(sigma1)
    if sigma2.ndim == 0:
        sigma2 = np.eye(fake_features.shape[1]) * float(sigma2)
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        covmean = linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps) @ (sigma2 + np.eye(sigma2.shape[0]) * eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_fvd_torchmetrics(real_videos: list[torch.Tensor], fake_videos: list[torch.Tensor], device: torch.device) -> tuple[float, str | None]:
    try:
        from torchmetrics.video.fvd import FrechetVideoDistance

        metric = FrechetVideoDistance().to(device)
        for real, fake in zip(real_videos, fake_videos):
            real_u8 = (real.permute(1, 2, 3, 0).unsqueeze(0) * 255).byte().to(device)
            fake_u8 = (fake.permute(1, 2, 3, 0).unsqueeze(0) * 255).byte().to(device)
            metric.update(real_u8, real=True)
            metric.update(fake_u8, real=False)
        return float(metric.compute().detach().cpu()), None
    except Exception as exc:  # noqa: BLE001
        return math.nan, repr(exc)


def build_r3d(weights_name: str, device: torch.device):
    from torchvision.models.video import R3D_18_Weights, r3d_18

    if weights_name == "r3d_18_kinetics":
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
        mean = torch.tensor(weights.transforms().mean, dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = torch.tensor(weights.transforms().std, dtype=torch.float32).view(1, 3, 1, 1, 1)
        backend = "r3d_18_kinetics400"
    else:
        model = r3d_18(weights=None)
        mean = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(1, 3, 1, 1, 1)
        backend = "r3d_18_random_smoke"
    model.fc = torch.nn.Identity()
    return model.to(device).eval(), mean.to(device), std.to(device), backend


def compute_fvd_r3d(real_videos: list[torch.Tensor], fake_videos: list[torch.Tensor], backbone: str, device: torch.device):
    try:
        model, mean, std, backend = build_r3d(backbone, device)
        real_features = []
        fake_features = []
        with torch.no_grad():
            for real, fake in zip(real_videos, fake_videos):
                real_b = real.unsqueeze(0).to(device)
                fake_b = fake.unsqueeze(0).to(device)
                real_b = F.interpolate(real_b, size=(max(4, real_b.shape[2]), 112, 112), mode="trilinear", align_corners=False)
                fake_b = F.interpolate(fake_b, size=(max(4, fake_b.shape[2]), 112, 112), mode="trilinear", align_corners=False)
                real_features.append(model((real_b - mean) / std).detach().cpu().numpy())
                fake_features.append(model((fake_b - mean) / std).detach().cpu().numpy())
        if not real_features:
            return math.nan, backend, "no videos"
        return frechet_distance(np.concatenate(real_features, axis=0), np.concatenate(fake_features, axis=0)), backend, None
    except Exception as exc:  # noqa: BLE001
        return math.nan, backbone, repr(exc)


def load_dfot_i3d_model(device: torch.device):
    dfot_root = REPO_ROOT / "third_party" / "dfot"
    if str(dfot_root) not in sys.path:
        sys.path.insert(0, str(dfot_root))
    i3d_path = dfot_root / "algorithms" / "common" / "metrics" / "video" / "models" / "i3d.py"
    spec = importlib.util.spec_from_file_location("vision_metrics_dfot_i3d", i3d_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import DFoT I3D model from {i3d_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_pretrained_i3d().to(device).eval()


def i3d_features(model, videos: list[torch.Tensor], device: torch.device) -> np.ndarray:
    features = []
    with torch.no_grad():
        for video in videos:
            x = video.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            if x.shape[1] < 9:
                pad_left = (9 - x.shape[1]) // 2
                pad_right = 9 - x.shape[1] - pad_left
                x = torch.cat(
                    [x[:, 0:1].repeat(1, pad_left, 1, 1, 1), x, x[:, -1:].repeat(1, pad_right, 1, 1, 1)],
                    dim=1,
                )
            x = (2.0 * x - 1.0).clamp(-1.0, 1.0)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            out = model(x, rescale=False, resize=True, return_features=True)
            if isinstance(out, (tuple, list)):
                out = out[0]
            features.append(out.flatten(1).detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fvd_dfot_i3d(real_videos: list[torch.Tensor], fake_videos: list[torch.Tensor], device: torch.device):
    try:
        if not real_videos:
            return math.nan, "dfot_i3d_torchscript", "no videos"
        model = load_dfot_i3d_model(device)
        real_features = i3d_features(model, real_videos, device)
        fake_features = i3d_features(model, fake_videos, device)
        return frechet_distance(real_features, fake_features), "dfot_i3d_torchscript", None
    except Exception as exc:  # noqa: BLE001
        return math.nan, "dfot_i3d_torchscript", repr(exc)


def compute_fvd(real_videos: list[torch.Tensor], fake_videos: list[torch.Tensor], backbone: str, device: torch.device):
    if backbone == "none":
        return math.nan, "none", "disabled"
    if backbone == "dfot_i3d":
        return compute_fvd_dfot_i3d(real_videos, fake_videos, device)
    if backbone == "torchmetrics":
        value, error = compute_fvd_torchmetrics(real_videos, fake_videos, device)
        return value, "torchmetrics", error
    return compute_fvd_r3d(real_videos, fake_videos, backbone, device)


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_timing(run_dir: Path, model: str, episodes: Sequence[dict]) -> dict:
    total_seconds = 0.0
    total_calls = 0
    split_seconds = {}
    split_calls = {}
    traces = 0
    for episode in episodes:
        path = run_dir / "models" / model / episode["name"] / "trace.json"
        if not path.exists():
            continue
        trace = read_json(path)
        traces += 1
        total_seconds += float(trace.get("total_prediction_seconds", 0.0))
        total_calls += int(trace.get("total_diffusion_calls", 0))
        for split, info in trace.get("splits", {}).items():
            split_seconds[split] = split_seconds.get(split, 0.0) + float(info.get("elapsed_seconds", 0.0))
            split_calls[split] = split_calls.get(split, 0) + int(info.get("diffusion_calls", 0))
    return {
        "num_traces": traces,
        "total_prediction_seconds": total_seconds,
        "total_diffusion_calls": total_calls,
        "split_seconds": split_seconds,
        "split_diffusion_calls": split_calls,
    }


def main() -> None:
    args = parse_args()
    patch_numpy_legacy_aliases()
    run_dir = args.run_dir.resolve()
    manifest = read_json(run_dir / "manifest.json")
    episodes = manifest["episodes"]
    models = list_models(run_dir, args.models)
    if not models:
        raise FileNotFoundError(f"No model prediction folders found under {run_dir / 'models'}")

    device = select_device(args.device)
    output_dir = run_dir / "metrics"
    observed_rows = []
    summary_rows = []
    timing_rows = []
    summary_json = {
        "run_dir": str(run_dir),
        "device": str(device),
        "fvd_requested_backend": args.fvd_backbone,
        "fvd_clip_length": args.fvd_clip_length,
        "fvd_clip_stride": args.fvd_clip_stride,
        "models": {},
    }

    for model in models:
        pairs_by_episode = {}
        for episode in episodes:
            episode_name = episode["name"]
            gt_dir = run_dir / "ground_truth" / episode_name / "frames"
            pred_dir = run_dir / "models" / model / episode_name / "observed" / "frames"
            pairs_by_episode[episode_name] = load_pairs(gt_dir, pred_dir)

        frame_rows, per_episode, observed_summary = compute_observed_pair_metrics(pairs_by_episode, device)
        for row in frame_rows:
            row["model"] = model
        observed_rows.extend(frame_rows)

        real_videos, fake_videos, fid_pairs, split_counts = collect_imagined_videos(
            run_dir,
            model,
            episodes,
            args.fvd_clip_length,
            args.fvd_clip_stride,
        )
        fid_value, fid_error = update_fid(fid_pairs, device)
        fvd_value, fvd_backend, fvd_error = compute_fvd(real_videos, fake_videos, args.fvd_backbone, device)
        timing = collect_timing(run_dir, model, episodes)

        imagined_summary = {
            "fid": fid_value,
            "fid_error": fid_error,
            "fvd": fvd_value,
            "fvd_backend": fvd_backend,
            "fvd_error": fvd_error,
            "num_imagined_frame_pairs": len(fid_pairs),
            "num_imagined_videos": len(real_videos),
            "imagined_split_counts": split_counts,
        }
        summary_json["models"][model] = {
            "observed": observed_summary,
            "imagined": imagined_summary,
            "timing": timing,
            "episodes": per_episode,
        }
        summary_rows.append(
            {
                "model": model,
                **observed_summary,
                **{k: v for k, v in imagined_summary.items() if not isinstance(v, dict)},
                "total_prediction_seconds": timing["total_prediction_seconds"],
                "total_diffusion_calls": timing["total_diffusion_calls"],
            }
        )
        timing_rows.append(
            {
                "model": model,
                "num_traces": timing["num_traces"],
                "total_prediction_seconds": timing["total_prediction_seconds"],
                "total_diffusion_calls": timing["total_diffusion_calls"],
                "split_seconds_json": timing["split_seconds"],
                "split_diffusion_calls_json": timing["split_diffusion_calls"],
            }
        )

    write_csv(output_dir / "observed_frame_metrics.csv", observed_rows, ["model", "episode", "split", "frame_idx", "psnr", "ssim", "lpips"])
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "model",
            "num_frames",
            "psnr_mean",
            "ssim_mean",
            "lpips_mean",
            "num_imagined_frame_pairs",
            "num_imagined_videos",
            "fid",
            "fvd",
            "fvd_backend",
            "total_prediction_seconds",
            "total_diffusion_calls",
            "fid_error",
            "fvd_error",
        ],
    )
    write_csv(
        output_dir / "timing_summary.csv",
        timing_rows,
        ["model", "num_traces", "total_prediction_seconds", "total_diffusion_calls", "split_seconds_json", "split_diffusion_calls_json"],
    )
    write_json(output_dir / "summary.json", summary_json)
    print(f"Wrote metrics to {output_dir}")
    for row in summary_rows:
        print(
            f"{row['model']}: observed LPIPS={row['lpips_mean']:.4g} "
            f"PSNR={row['psnr_mean']:.4g} SSIM={row['ssim_mean']:.4g}; "
            f"imagined FVD={row['fvd']:.4g} FID={row['fid']:.4g}; "
            f"calls={row['total_diffusion_calls']}"
        )


if __name__ == "__main__":
    main()
