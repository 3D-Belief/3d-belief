#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze per-episode metrics (no visibility).

Features:
1) Load one metrics_*.json per episode from metrics/ subfolder.
2) Keep only scalar (int/float/bool) fields as metrics.
3) Compute global mean/std/count for each metric across episodes.
4) Save concatenated metrics CSV and global stats CSV/JSON.

Directory layout expected:
root/
  EPISODE_ID_A/
    metrics/
      metrics_0.json
  EPISODE_ID_B/
    metrics/
      metrics_0.json
  ...

Usage:
  python analyze_metrics_simple.py /path/to/root --outdir results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _discover_metric_files(root: Path) -> List[Path]:
    """
    Discover all metrics_*.json files, assuming exactly one per episode under:
        EPISODE_ID/trace/metrics/metrics_*.json
    """
    files: List[Path] = []

    for episode_dir in root.iterdir():
        if not episode_dir.is_dir():
            continue

        metrics_dir = episode_dir / "trace" / "metrics"
        if not metrics_dir.is_dir():
            continue

        episode_files = sorted(metrics_dir.glob("metrics_*.json"))
        if not episode_files:
            continue

        files.extend(episode_files)

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
    """
    Build a flat DataFrame with one row per metrics JSON.

    Only keeps episodes whose object_coverage > 0.
    """
    rows = []
    files = _discover_metric_files(root)

    for p in files:
        data = _load_one_json(p)
        if not data:
            continue

        # Filter: keep only episodes with object_coverage > 0
        oc = data.get("object_coverage", 0)
        try:
            oc_val = float(oc)
        except (TypeError, ValueError):
            continue
        # if not (oc_val > 0):
        #     continue

        # Attach episode id (parent of metrics/):
        episode_id = p.parent.parent.name
        clean = {"episode_id": episode_id, "file": str(p)}

        # Only keep scalar metrics (int/float/bool); skip lists/dicts/etc.
        for k, v in data.items():
            if isinstance(v, (int, float, bool)):
                clean[k] = float(v) if isinstance(v, bool) else v

        rows.append(clean)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Ensure numeric where possible for metric columns
    for col in df.columns:
        if col in ("episode_id", "file"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


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


def main():
    ap = argparse.ArgumentParser(description="Analyze per-episode metrics (no visibility).")
    ap.add_argument("root", type=str, help="Root directory containing episode folders")
    ap.add_argument("--outdir", type=str, default="analysis_results", help="Output dir")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _collect_dataframe(root)
    if df.empty:
        print("No valid metrics found after filtering object_coverage>0. Exiting.")
        return

    # Save raw concatenated data
    df.to_csv(outdir / "all_metrics_concatenated.csv", index=False)

    # Global stats over all scalar metrics
    metric_cols = [c for c in df.columns if c not in ("episode_id", "file")]
    global_df = _global_stats(df, metric_cols)
    global_df.to_csv(outdir / "global_stats.csv", index=False)
    with open(outdir / "global_stats.json", "w") as f:
        json.dump(global_df.to_dict(orient="records"), f, indent=2)

    with open(outdir / "README.txt", "w") as f:
        f.write(
            "Outputs:\n"
            "- all_metrics_concatenated.csv: flat table of all loaded rows (filtered by object_coverage>0).\n"
            "- global_stats.csv / global_stats.json: global mean/std/count per metric.\n"
        )

    print(f"Done. Results saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
