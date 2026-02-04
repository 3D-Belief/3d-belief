#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze per-episode aggregated metrics (with per-file filtering).

We keep only metrics JSON files that satisfy:
  - siglip_similarity > 0.99
  - lpips_distance < 0.01

Then, for each episode, aggregate metrics across the kept files:
  <metric>_mean, <metric>_std, <metric>_n

Finally, compute global stats across episodes using the per-episode means.

Usage:
  python analyze_episode_level_metrics.py /path/to/root --outdir results
Optional:
  --metrics lpips_distance siglip_similarity   (default)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _load_one_json(p: Path) -> Optional[Dict]:
    try:
        with open(p, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _discover_episode_metric_files(root: Path) -> Dict[str, List[Path]]:
    """
    Returns:
      episode_id -> list of metrics_*.json paths under EPISODE_ID/trace/metrics/
    """
    out: Dict[str, List[Path]] = {}

    for episode_dir in root.iterdir():
        if not episode_dir.is_dir():
            continue

        metrics_dir = episode_dir / "trace" / "metrics"
        if not metrics_dir.is_dir():
            continue

        files = sorted(metrics_dir.glob("metrics_*.json"))
        if not files:
            continue

        out[episode_dir.name] = files

    return out


def _file_passes_thresholds(data: Dict) -> bool:
    """
    Keep only if:
      siglip_similarity > 0.99 and lpips_distance < 0.01
    Missing / non-numeric => fail.
    """
    try:
        sig = float(data.get("siglip_similarity", np.nan))
        lpi = float(data.get("lpips_distance", np.nan))
    except (TypeError, ValueError):
        return False

    if not (np.isfinite(sig) and np.isfinite(lpi)):
        return False

    return (sig > 0.99) and (lpi < 0.01)


def _episode_aggregate(
    episode_id: str,
    files: List[Path],
    metrics_to_keep: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    Aggregate metrics within one episode across *kept* files only.
    Returns a dict with keys:
      episode_id, n_files_total, n_kept_files,
      <metric>_mean, <metric>_std, <metric>_n
    """
    collected: Dict[str, List[float]] = {}

    n_total = 0
    n_kept = 0

    for p in files:
        n_total += 1
        data = _load_one_json(p)
        if not data:
            continue

        # # Per-file filtering condition
        # if not _file_passes_thresholds(data):
        #     continue

        n_kept += 1

        for k, v in data.items():
            if metrics_to_keep is not None and k not in metrics_to_keep:
                continue
            if isinstance(v, (int, float, bool)):
                val = float(v) if isinstance(v, bool) else float(v)
                if np.isfinite(val):
                    collected.setdefault(k, []).append(val)

    # Skip episodes with no kept files (or nothing collected)
    if n_kept == 0 or not collected:
        return None

    row: Dict = {
        "episode_id": episode_id,
        "n_files_total": n_total,
        "n_kept_files": n_kept,
    }

    for m, vals in collected.items():
        arr = np.asarray(vals, dtype=np.float64)
        n = int(arr.size)
        mean = float(np.mean(arr)) if n > 0 else np.nan
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        row[f"{m}_mean"] = mean
        row[f"{m}_std"] = std
        row[f"{m}_n"] = n

    return row


def _global_stats_over_episodes(df: pd.DataFrame, metric_basenames: List[str]) -> pd.DataFrame:
    """
    Global stats computed over per-episode means: <metric>_mean columns.
    """
    rows = []
    for m in metric_basenames:
        col = f"{m}_mean"
        if col not in df.columns:
            continue

        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            rows.append({"metric": m, "mean": np.nan, "std": np.nan, "n": 0})
            continue

        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        rows.append({"metric": m, "mean": mean, "std": std, "n": int(vals.size)})

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Aggregate metrics per episode, with per-file thresholds.")
    ap.add_argument("root", type=str, help="Root directory containing episode folders")
    ap.add_argument("--outdir", type=str, default="analysis_results_episode", help="Output dir")
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=["lpips_distance", "siglip_similarity"],
        help="Metric keys to keep (default: lpips_distance siglip_similarity). Use empty to keep all scalar keys.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_to_keep = args.metrics if len(args.metrics) > 0 else None

    episode_map = _discover_episode_metric_files(root)
    if not episode_map:
        print("No episode metrics folders found. Exiting.")
        return

    rows = []
    for episode_id, files in episode_map.items():
        row = _episode_aggregate(episode_id, files, metrics_to_keep=metrics_to_keep)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No episodes had any files passing: siglip_similarity>0.99 and lpips_distance<0.01. Exiting.")
        return

    df = pd.DataFrame(rows)

    # Save per-episode aggregated table
    df.to_csv(outdir / "per_episode_metrics.csv", index=False)

    # Determine metric basenames from columns ending with _mean
    metric_basenames = sorted({c[:-5] for c in df.columns if c.endswith("_mean")})

    global_df = _global_stats_over_episodes(df, metric_basenames)
    global_df.to_csv(outdir / "global_stats_over_episodes.csv", index=False)
    with open(outdir / "global_stats_over_episodes.json", "w") as f:
        json.dump(global_df.to_dict(orient="records"), f, indent=2)

    with open(outdir / "README.txt", "w") as f:
        f.write(
            "Outputs:\n"
            "- per_episode_metrics.csv: one row per episode; aggregates computed only over files with\n"
            "- global_stats_over_episodes.csv/json: mean/std/n computed over the per-episode means.\n"
        )

    print(f"Done. Results saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
