"""
Visualize scene graph as a 2D top-down view (XZ plane, Y is up).
- Objects: colored rectangles from bounding box XZ projection
- Walls: lines fitted via PCA through wall Gaussian positions
- GT top-down reference images copied alongside

Usage:
    python scripts/visualize_bboxes.py \
        --bbox_json outputs/inference/procthor_seg/visuals_0_108_129_0/step_18/bboxes_0_108_129.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mpl_patches
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STRUCTURAL_KEYWORDS = {"wall", "ceiling", "room", "floor"}


def is_wall(name):
    return "wall" in name.lower()


def is_structural(name):
    return any(kw in name.lower() for kw in STRUCTURAL_KEYWORDS)


def get_scene_dir(dataset_root: Path, video_idx: int) -> Path:
    scene_paths = sorted([p for p in dataset_root.glob("*/") if p.is_dir()])
    if video_idx >= len(scene_paths):
        raise ValueError(f"video_idx {video_idx} out of range ({len(scene_paths)} scenes)")
    return scene_paths[video_idx]


def deterministic_color(label, seed=17):
    """Hash-based deterministic color for a label string."""
    import hashlib, colorsys
    h = int(hashlib.md5(f"{label}:{seed}".encode()).hexdigest(), 16)
    hue = (h & 0xFFFF) / 0xFFFF
    sat = 0.5 + ((h >> 16) & 0xFF) / 0xFF * 0.3
    val = 0.65 + ((h >> 24) & 0xFF) / 0xFF * 0.3
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (r, g, b)


# ---------------------------------------------------------------------------
# Wall line fitting
# ---------------------------------------------------------------------------

def load_wall_gaussians_per_object(npz_path, scene_graph):
    """Load Gaussian positions grouped by wall object.

    Returns:
        list of (name, instance_id, points_xz) for each wall object,
        where points_xz is (K, 2) array of XZ positions.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        print(f"No npz file at {npz_path}")
        return []

    data = np.load(str(npz_path))
    means = data["means"]       # (N, 3)
    seg_rgb = data["seg_rgb"]   # (N, 3) uint8

    meta = scene_graph.get("metadata", {})
    node_rgbs = meta.get("node_rgbs", [])
    node_names = scene_graph.get("node_type_names", [])
    node_inst_ids = meta.get("node_instance_ids", [""] * len(node_names))

    wall_groups = []
    for i, name in enumerate(node_names):
        if not is_wall(name):
            continue
        if i >= len(node_rgbs):
            continue
        rgb = np.array(node_rgbs[i], dtype=np.uint8)
        mask = np.all(seg_rgb == rgb, axis=1)
        if mask.sum() < 5:
            continue
        pts = means[mask][:, [0, 2]]  # project to XZ
        wall_groups.append((name, node_inst_ids[i] if i < len(node_inst_ids) else "", pts))

    print(f"Found {len(wall_groups)} wall segments with Gaussian data")
    return wall_groups


def fit_wall_line(points_xz, interior_centroid_xz):
    """Fit a wall line by picking the AABB edge closest to the room interior.

    1. Find the thin axis in XZ (wall thickness direction)
    2. The AABB has two long edges along the other axis
    3. Pick the edge closer to interior_centroid_xz (camera-facing side)

    Args:
        points_xz: (K, 2) wall Gaussian positions in XZ
        interior_centroid_xz: (2,) centroid of non-wall objects in XZ

    Returns (start, end) as 2D vectors in the XZ plane.
    """
    x_min, z_min = np.percentile(points_xz, 10, axis=0)
    x_max, z_max = np.percentile(points_xz, 90, axis=0)
    x_extent = x_max - x_min
    z_extent = z_max - z_min

    if x_extent < z_extent:
        # Thin in X → wall runs along Z. Two candidate edges: x_min and x_max
        # Pick the one closer to interior centroid
        if abs(x_min - interior_centroid_xz[0]) < abs(x_max - interior_centroid_xz[0]):
            wall_x = x_min
        else:
            wall_x = x_max
        start = np.array([wall_x, z_min])
        end = np.array([wall_x, z_max])
    else:
        # Thin in Z → wall runs along X. Two candidate edges: z_min and z_max
        if abs(z_min - interior_centroid_xz[1]) < abs(z_max - interior_centroid_xz[1]):
            wall_z = z_min
        else:
            wall_z = z_max
        start = np.array([x_min, wall_z])
        end = np.array([x_max, wall_z])

    return start, end


# ---------------------------------------------------------------------------
# Top-down 2D rendering
# ---------------------------------------------------------------------------

def render_topdown(objects, wall_groups, output_path, max_objects=50):
    """Render a 2D top-down view (XZ plane).
    - Non-structural objects: colored rectangles
    - Walls: fitted lines
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Collect all points for auto-scaling
    all_x, all_z = [], []

    # --- Draw non-structural objects as rectangles ---
    obj_patches = []
    for obj in objects[:max_objects]:
        name = obj["name"]
        if is_structural(name):
            continue

        cx, _, cz = obj["bbox"]["center"]
        sx, _, sz = obj["bbox"]["extents"]
        color = deterministic_color(name)

        rect = mpl_patches.FancyBboxPatch(
            (cx - sx / 2, cz - sz / 2), sx, sz,
            boxstyle="round,pad=0.02",
            facecolor=(*color, 0.3), edgecolor=(*color, 0.9), linewidth=1.5,
        )
        ax.add_patch(rect)

        # Label
        ax.text(cx, cz, name, fontsize=7, ha="center", va="center",
                color=(*color,), weight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

        all_x.extend([cx - sx / 2, cx + sx / 2])
        all_z.extend([cz - sz / 2, cz + sz / 2])

        gs = obj.get("num_gaussians", 0)
        obj_patches.append(mpatches.Patch(color=color, label=f"{name} ({gs} gs)"))

    # --- Compute interior centroid from non-structural objects ---
    obj_centers = []
    for obj in objects[:max_objects]:
        if not is_structural(obj["name"]):
            cx, _, cz = obj["bbox"]["center"]
            obj_centers.append([cx, cz])
    if obj_centers:
        interior_centroid = np.mean(obj_centers, axis=0)
    else:
        interior_centroid = np.array([0.0, 0.0])

    # --- Draw walls as fitted lines ---
    wall_color = (0.4, 0.4, 0.4)
    for name, inst_id, pts_xz in wall_groups:
        # Per-axis outlier filtering on wall Gaussians
        for dim in range(2):
            if len(pts_xz) < 5:
                break
            n = len(pts_xz)
            lo = max(0.10, 2 / n)
            hi = 1.0 - lo
            lo, hi = min(lo, 0.4), max(hi, 0.6)
            q_lo = np.percentile(pts_xz[:, dim], lo * 100)
            q_hi = np.percentile(pts_xz[:, dim], hi * 100)
            pts_xz = pts_xz[(pts_xz[:, dim] >= q_lo) & (pts_xz[:, dim] <= q_hi)]

        if len(pts_xz) < 5:
            continue
        start, end = fit_wall_line(pts_xz, interior_centroid)
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=wall_color, linewidth=2.5, solid_capstyle="round")

        all_x.extend([start[0], end[0]])
        all_z.extend([start[1], end[1]])

    if wall_groups:
        obj_patches.append(mpatches.Patch(color=wall_color,
                                          label=f"Walls ({len(wall_groups)} segments)"))

    # --- Axis setup ---
    if all_x and all_z:
        x_min, x_max = min(all_x), max(all_x)
        z_min, z_max = min(all_z), max(all_z)
        pad_x = (x_max - x_min) * 0.15 + 0.5
        pad_z = (z_max - z_min) * 0.15 + 0.5
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(z_min - pad_z, z_max + pad_z)

    ax.set_aspect("equal")
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Z (world)")
    ax.set_title("Top-Down Scene Graph (XZ plane)", fontsize=14)
    ax.grid(True, alpha=0.2)

    if obj_patches:
        ax.legend(handles=obj_patches, loc="upper left", fontsize=7,
                  framealpha=0.8, ncol=max(1, len(obj_patches) // 12))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top-down view to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize scene graph top-down")
    parser.add_argument("--bbox_json", type=str, required=True)
    parser.add_argument("--video_idx", type=int, default=0)
    parser.add_argument("--dataset_root", type=str, default="datasets/spoc/unit")
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    bbox_path = Path(args.bbox_json)
    with open(bbox_path) as f:
        scene_graph = json.load(f)

    # Convert scene_graph -> internal objects list
    num_nodes = scene_graph["num_nodes"]
    positions = scene_graph["node_positions"]
    sizes = scene_graph["node_bbox_sizes"]
    names = scene_graph["node_type_names"]
    meta = scene_graph.get("metadata", {})
    rgbs = meta.get("node_rgbs", [[128, 128, 128]] * num_nodes)
    num_gs = meta.get("node_num_gaussians", [0] * num_nodes)

    objects = []
    for i in range(num_nodes):
        cx, cy, cz = positions[i]
        sx, sy, sz = sizes[i]
        objects.append({
            "object_id": i,
            "rgb": [int(rgbs[i][0]), int(rgbs[i][1]), int(rgbs[i][2])],
            "name": names[i],
            "num_gaussians": num_gs[i],
            "bbox": {
                "min": [cx - sx/2, cy - sy/2, cz - sz/2],
                "max": [cx + sx/2, cy + sy/2, cz + sz/2],
                "center": [cx, cy, cz],
                "extents": [sx, sy, sz],
            },
        })
    print(f"Loaded scene graph: {num_nodes} nodes, {scene_graph['num_edges']} edges")

    # Resolve scene directory for GT reference
    if meta.get("scene_dir"):
        scene_dir = Path(meta["scene_dir"])
    else:
        scene_dir = get_scene_dir(Path(args.dataset_root), args.video_idx)

    output_dir = Path(args.output_dir) if args.output_dir else bbox_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load wall Gaussians from npz and fit lines
    npz_path = bbox_path.with_suffix(".npz")
    wall_groups = load_wall_gaussians_per_object(npz_path, scene_graph)

    # Render top-down view
    render_topdown(objects, wall_groups, output_dir / "scene_topdown.png",
                   max_objects=args.max_objects)

    # Copy GT reference images
    for name in ["top_down_view_initial.png", "top_down_view_final.png"]:
        gt_path = scene_dir / name
        if gt_path.exists():
            from PIL import Image
            Image.open(gt_path).save(output_dir / f"gt_{name}")
            print(f"Saved GT reference: gt_{name}")

    print("Done!")


if __name__ == "__main__":
    main()
