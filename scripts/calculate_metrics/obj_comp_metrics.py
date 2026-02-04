#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze per-episode metrics over visibility.

Features:
1) Bin by visibility; per-bin mean/std for all other metrics.
2) Error-bar plots (metric vs visibility bin center).
3) Global mean/std for all metrics (except visibility).
4) Scatter plots of each metric vs visibility (all points).
5) Headless-safe (Agg backend). Outputs CSV/JSON + PNG plots.

Directory layout expected:
root/
  EPISODE_ID_A/
    metrics/
      metrics_0.json
      metrics_1.json
      ...
  EPISODE_ID_B/
    metrics/
      metrics_0.json
      ...

Each JSON example:
{
  "visibility": 0.6814,
  "metric_bev_iou": 0.0621,
  "metric_iou_3d": 0.0169,
  "metric_chamfer": 0.01677,
  "clip_similarity": 0.9014,
  "siglip_similarity": 0.8768,
  "lpips_distance": 0.7134,
  "vlm_obj_recognition": 0
}

Usage:
  python analyze_metrics.py /path/to/root \
      --outdir results \
      --bins 10
  # OR custom bin edges (inclusive of left edge, exclusive of right except last bin):
  python analyze_metrics.py /path/to/root \
      --outdir results \
      --bin-edges "0,0.2,0.5,0.8,1.0"
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def _discover_metric_files(root: Path) -> List[Path]:
    """
    Discover all metrics_*.json files, but only for episodes whose *last*
    metrics file has both metric_bev_iou and metric_iou_3d > 0.3.
    """
    files: List[Path] = []

    for episode_dir in root.iterdir():
        if not episode_dir.is_dir():
            continue

        metrics_dir = episode_dir / "trace" / "metrics"
        if not metrics_dir.is_dir():
            continue

        # All metrics files for this episode
        metrics_files = sorted(metrics_dir.glob("metrics_*.json"))
        if not metrics_files:
            continue

        # Look at the *last* metrics file (highest index)
        last_file = metrics_files[-1]
        data = _load_one_json(last_file)
        if not data:
            continue

        bev = data.get("metric_bev_iou", None)
        iou3d = data.get("metric_iou_3d", None)

        # Must have both metrics and both strictly > 0.3
        if bev is None or iou3d is None:
            continue
        try:
            bev_val = float(bev)
            iou3d_val = float(iou3d)
        except (TypeError, ValueError):
            continue

        # if bev_val > 0.3 and iou3d_val > 0.3:
        # Episode passes filter: include *all* its metrics files
        files.extend(metrics_files)
        # print the episode path
        print(f"Including episode: {episode_dir.name}")
        # else: skip this entire episode

    return files

def _load_one_json(p: Path) -> Optional[Dict]:
    try:
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _collect_dataframe(root: Path) -> pd.DataFrame:
    rows = []
    files = _discover_metric_files(root)
    for p in files:
        data = _load_one_json(p)
        if not data:
            continue
        # Attach episode id (parent of metrics/):
        episode_id = p.parent.parent.name
        # Only keep keys that are scalar (int/float/bool); skip nested.
        clean = {"episode_id": episode_id, "file": str(p)}
        for k, v in data.items():
            if isinstance(v, (int, float, bool)):
                clean[k] = float(v) if isinstance(v, bool) else v
        # Must have visibility to be usable for binning and plotting
        if "visibility" not in clean:
            continue
        rows.append(clean)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Ensure numeric where possible
    for col in df.columns:
        if col in ("episode_id", "file"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop rows with NaN visibility
    df = df.dropna(subset=["visibility"])
    return df


def _parse_bins(args, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    vis = df["visibility"].to_numpy()

    if args.bin_edges:
        edges = np.array([float(x) for x in args.bin_edges.split(",")], dtype=float)
        edges = np.unique(edges)
        if len(edges) < 2:
            raise ValueError("Need at least two bin edges.")
    else:
        # equal-width bins on [min_visibility, max_visibility] (default: clamp to [0,1])
        vmin = np.nanmin(vis)
        vmax = np.nanmax(vis)
        # clamp to [0,1] if visibility is logically bounded there
        vmin = max(0.0, vmin)
        vmax = min(1.0, vmax)
        edges = np.linspace(vmin, vmax, args.bins + 1)

    centers = 0.5 * (edges[:-1] + edges[1:])

    # If there are any points with visibility == 0, make the first bin's
    # center exactly 0 for nicer x-axis semantics.
    if np.any(vis == 0):
        centers[0] = 0.0

    return edges, centers


def _per_bin_stats(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    # Which metrics? Everything numeric except: 'visibility'
    numeric_cols = [c for c in df.columns if c not in ("episode_id", "file")]
    metrics = [c for c in numeric_cols if c != "visibility"]

    vis = df["visibility"].to_numpy()

    # Assign bins (1..nbins), we'll shift to 0-based
    bin_idx = np.digitize(vis, edges, right=False) - 1
    nbins = len(edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    # --- Make visibility == 0 its own bin (bin 0) ---
    mask_zero = (vis == 0)

    if nbins > 1:
        # Any non-zero points that ended up in bin 0 get pushed to bin 1
        bin_idx[(bin_idx == 0) & ~mask_zero] = 1

    # All zero-visibility points are explicitly assigned to bin 0
    bin_idx[mask_zero] = 0
    # -----------------------------------------------

    df = df.copy()
    df["vis_bin"] = bin_idx

    # Group and compute mean/std/count per bin per metric
    out_rows = []
    g = df.groupby("vis_bin", dropna=False)
    for b in range(nbins):
        sub = g.get_group(b) if b in g.groups else None
        row = {"vis_bin": b, "count": 0}
        if sub is not None and len(sub) > 0:
            row["count"] = int(len(sub))
            for m in metrics:
                vals = pd.to_numeric(sub[m], errors="coerce").dropna().to_numpy()
                if vals.size > 0:
                    row[f"{m}_mean"] = float(np.mean(vals))
                    row[f"{m}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                else:
                    row[f"{m}_mean"] = np.nan
                    row[f"{m}_std"] = np.nan
        else:
            for m in metrics:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
        out_rows.append(row)

    return pd.DataFrame(out_rows), metrics

def _global_stats(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        vals = pd.to_numeric(df[m], errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            mean = np.nan
            std = np.nan
            n = 0
        else:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            n = int(vals.size)
        rows.append({"metric": m, "mean": mean, "std": std, "n": n})
    return pd.DataFrame(rows)


def _plot_bin_curves(
    outdir: Path,
    centers: np.ndarray,
    bin_df: pd.DataFrame,
    metrics: List[str],
    title_prefix: str = "metric_vs_visibility",
):
    outdir.mkdir(parents=True, exist_ok=True)
    x = centers

    for m in metrics:
        y = bin_df[f"{m}_mean"].to_numpy()
        yerr = bin_df[f"{m}_std"].to_numpy()
        # If all-NaN, skip
        if np.all(np.isnan(y)):
            continue

        plt.figure()
        plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3)
        plt.xlabel("Visibility (bin center)")
        plt.ylabel(m.replace("_", " "))
        plt.title(f"{m} vs Visibility")
        plt.grid(True, alpha=0.3)
        fname = outdir / f"{title_prefix}_{m}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()


def _plot_scatter(
    outdir: Path,
    df: pd.DataFrame,
    metrics: List[str],
    title_prefix: str = "scatter_visibility",
):
    outdir.mkdir(parents=True, exist_ok=True)
    x = pd.to_numeric(df["visibility"], errors="coerce")

    for m in metrics:
        y = pd.to_numeric(df[m], errors="coerce")
        if y.dropna().empty:
            continue

        plt.figure()
        plt.scatter(x, y, s=8, alpha=0.35)
        plt.xlabel("Visibility")
        plt.ylabel(m.replace("_", " "))
        plt.title(f"{m} vs Visibility (all points)")
        plt.grid(True, alpha=0.3)
        fname = outdir / f"{title_prefix}_{m}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Analyze metrics vs visibility.")
    ap.add_argument("root", type=str, help="Root directory containing episode folders")
    ap.add_argument("--outdir", type=str, default="analysis_results", help="Output dir")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--bins", type=int, default=10, help="Number of equal-width bins")
    group.add_argument(
        "--bin-edges",
        type=str,
        default=None,
        help='Custom bin edges, comma-separated (e.g. "0,0.2,0.5,0.8,1.0")',
    )
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load everything
    df = _collect_dataframe(root)
    if df.empty:
        print("No valid metrics found. Exiting.")
        return

    # Save raw concatenated data (optional but handy)
    df.to_csv(outdir / "all_metrics_concatenated.csv", index=False)

    # 2) Prepare bins
    edges, centers = _parse_bins(args, df)
    np.savetxt(outdir / "visibility_bin_edges.txt", edges, fmt="%.8f")
    np.savetxt(outdir / "visibility_bin_centers.txt", centers, fmt="%.8f")

    # 3) Per-bin stats
    bin_df, metrics = _per_bin_stats(df, edges)
    bin_df.to_csv(outdir / "per_bin_stats.csv", index=False)
    with open(outdir / "per_bin_stats.json", "w") as f:
        json.dump(bin_df.to_dict(orient="list"), f, indent=2)

    # 4) Global stats
    global_df = _global_stats(df, metrics)
    global_df.to_csv(outdir / "global_stats.csv", index=False)
    with open(outdir / "global_stats.json", "w") as f:
        json.dump(global_df.to_dict(orient="records"), f, indent=2)

    # 5) Plots (metric vs visibility with error bars)
    plots_dir = outdir / "plots"
    _plot_bin_curves(plots_dir, centers, bin_df, metrics)

    # 6) Scatter plots of metric vs visibility (all points)
    _plot_scatter(plots_dir, df, metrics)

    # 7) Also save a quick README of what was produced
    with open(outdir / "README.txt", "w") as f:
        f.write(
            "Outputs:\n"
            "- all_metrics_concatenated.csv: flat table of all loaded rows.\n"
            "- visibility_bin_edges.txt / visibility_bin_centers.txt: binning info.\n"
            "- per_bin_stats.csv / per_bin_stats.json: per-visibility-bin mean/std per metric.\n"
            "- global_stats.csv / global_stats.json: global mean/std per metric.\n"
            "- plots/: PNGs for per-bin curves and scatter plots (headless-safe).\n"
        )

    print(f"Done. Results saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
