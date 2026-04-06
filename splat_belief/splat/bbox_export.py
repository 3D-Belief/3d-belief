import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from splat_belief.splat.types import Gaussians


def _load_color_to_label(scene_dir):
    """Build a mapping from (R,G,B) tuple -> (label_name, instance_id) from
    all_semantic_meta.json. Prefers non-'Unknown' labels if available."""
    if scene_dir is None:
        return {}
    meta_path = Path(scene_dir) / "all_semantic_meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        meta = json.load(f)

    color_to_info = {}
    for step_data in meta:
        for obj in step_data["objects"].values():
            rgb = tuple(obj["color"])
            name = obj["name"]
            instance_id = obj.get("instance_id", "")
            # Derive a better name from instance_id for "Unknown" entries
            if name == "Unknown" and instance_id:
                name = instance_id.split("|")[0].replace("_", " ").replace("-", " ").title()
            # Prefer non-placeholder labels
            if rgb not in color_to_info or color_to_info[rgb][0].startswith("Unknown"):
                color_to_info[rgb] = (name, instance_id)
    return color_to_info


def _lookup_label(rgb, color_to_info, max_l2_dist=65.0):
    """Look up (label, instance_id) by RGB with fuzzy matching."""
    rgb = tuple(rgb)
    if rgb in color_to_info:
        return color_to_info[rgb]
    if not color_to_info:
        return ("Unknown", "")
    best_key = min(color_to_info.keys(),
                   key=lambda x: sum((a - b) ** 2 for a, b in zip(x, rgb)))
    best_dist = sum((a - b) ** 2 for a, b in zip(best_key, rgb)) ** 0.5
    if best_dist <= max_l2_dist:
        return color_to_info[best_key]
    return ("Unknown", "")


def _build_type_vocab(labels):
    """Build a deterministic class label -> integer ID mapping from the set
    of labels seen in this scene. Sorted alphabetically so IDs are stable
    within a single scene."""
    unique_labels = sorted(set(labels))
    return {label: i for i, label in enumerate(unique_labels)}


def _compute_edges(positions, labels, instance_ids, near_threshold=2.0):
    """Compute NEAR (distance) and SAME_ROOM (shared room) edges.

    Returns list of {src, dst, type_id, type_name} dicts (bidirectional).
    """
    n = len(positions)
    if n == 0:
        return []

    positions = np.array(positions)
    # Pairwise squared distances
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))  # (n, n)

    # Assign each object to a "room" by matching against any "Room"/"room|..." label.
    # Simplest heuristic: use the "room|X" substring from the instance_id if present,
    # otherwise fall back to "None". Objects in the same room_id are connected.
    room_of = []
    for inst in instance_ids:
        room_key = None
        if inst:
            # Look for patterns like "room|3" or "Ceiling_room|2" etc.
            parts = inst.split("|")
            for i, p in enumerate(parts):
                if "room" in p.lower():
                    # e.g., "room|3" -> "room|3", or "Ceiling_room|2|..." -> "Ceiling_room|2"
                    if i + 1 < len(parts):
                        room_key = f"{p}|{parts[i+1]}"
                    else:
                        room_key = p
                    break
        room_of.append(room_key)

    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # NEAR: within threshold distance
            if dists[i, j] < near_threshold:
                edges.append({"src": i, "dst": j, "type_id": 1, "type_name": "NEAR"})
            # SAME_ROOM: same non-None room key
            elif room_of[i] is not None and room_of[i] == room_of[j]:
                edges.append({"src": i, "dst": j, "type_id": 2, "type_name": "SAME_ROOM"})
    return edges


def _merge_by_color_proximity(objects, rgb_threshold=15.0):
    """Merge objects whose RGB colors are within threshold (L2 distance).

    Quantization drift causes one physical object to fragment into many entries
    with slightly different RGB values. This clusters them back together by
    greedily merging any pair within the threshold, then computing the union AABB.
    """
    if not objects:
        return objects

    rgbs = np.array([o["rgb"] for o in objects], dtype=np.float64)
    n = len(objects)
    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    # Merge pairs within threshold
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(rgbs[i] - rgbs[j]) < rgb_threshold:
                union(i, j)

    # Group by cluster root
    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Merge each cluster into a single object
    merged = []
    for root, indices in clusters.items():
        cluster_objs = [objects[i] for i in indices]

        # Union AABB
        all_mins = np.array([o["bbox"]["min"] for o in cluster_objs])
        all_maxs = np.array([o["bbox"]["max"] for o in cluster_objs])
        bbox_min = all_mins.min(axis=0).tolist()
        bbox_max = all_maxs.max(axis=0).tolist()
        center = [(lo + hi) / 2 for lo, hi in zip(bbox_min, bbox_max)]
        extents = [hi - lo for lo, hi in zip(bbox_min, bbox_max)]

        total_gaussians = sum(o["num_gaussians"] for o in cluster_objs)

        # Use the RGB of the largest fragment as the representative color
        largest = max(cluster_objs, key=lambda o: o["num_gaussians"])
        r, g, b = largest["rgb"]

        merged.append({
            "object_id": largest["object_id"],
            "rgb": [r, g, b],
            "rgb_hex": f"#{r:02X}{g:02X}{b:02X}",
            "num_gaussians": total_gaussians,
            "num_merged_fragments": len(indices),
            "bbox": {
                "min": bbox_min,
                "max": bbox_max,
                "center": center,
                "extents": extents,
            },
        })

    merged.sort(key=lambda o: o["num_gaussians"], reverse=True)
    return merged


def extract_bounding_boxes(
    gaussians: Gaussians,
    batch_idx: int = 0,
    opacity_threshold: float = 0.1,
    min_gaussians: int = 10,
    scene_dir: str | Path | None = None,
    rgb_merge_threshold: float = 5.0,
    outlier_percentile: float = 0.20,
    dbscan_eps: float = 0.5,
) -> dict:
    """
    Extract axis-aligned 3D bounding boxes for each segmented object.

    Groups Gaussians by their segmentation RGB color and computes an AABB
    for each group from the Gaussian means. Colors within rgb_merge_threshold
    L2 distance are merged to handle float quantization drift.

    Args:
        gaussians: Gaussians dataclass with segmentation field populated.
        batch_idx: Which batch element to process.
        opacity_threshold: Discard Gaussians with opacity below this.
        min_gaussians: Objects with fewer Gaussians go into ignored_objects.
        scene_dir: Path to the episode directory (stored in metadata for provenance).
        rgb_merge_threshold: L2 distance in RGB space (0-255) below which
            colors are considered the same object. Default 5 handles
            residual float32 quantization drift (MP4 compression is fixed
            at load time by snapping to GT colors).
        outlier_percentile: Fraction of outermost Gaussians (by distance from
            centroid) to discard per object before computing the AABB.
            0.10 = trim the farthest 10%.
        dbscan_eps: DBSCAN neighborhood radius (world units). Gaussians
            farther than this from any cluster core are noise. Default 0.5m.

    Returns:
        dict ready for JSON serialization.
    """
    if gaussians.segmentation is None:
        raise ValueError(
            "Gaussians.segmentation is None — cannot extract bounding boxes. "
            "Ensure use_segmentation=True in the encoder config."
        )

    B = gaussians.means.shape[0]
    if batch_idx >= B:
        raise ValueError(f"batch_idx {batch_idx} >= batch size {B}")

    means = gaussians.means[batch_idx]            # [N, 3]
    opacities = gaussians.opacities[batch_idx]    # [N]
    seg = gaussians.segmentation[batch_idx]       # [N, 3]

    total_gaussians = means.shape[0]

    # Opacity filter
    keep = opacities > opacity_threshold
    means = means[keep]
    seg = seg[keep]
    gaussians_after_filter = means.shape[0]

    if gaussians_after_filter == 0:
        return {
            "metadata": {
                "coordinate_frame": "world",
                "scene_dir": str(scene_dir) if scene_dir is not None else None,
                "opacity_threshold": opacity_threshold,
                "min_gaussians": min_gaussians,
                "total_gaussians": total_gaussians,
                "gaussians_after_filter": 0,
                "num_objects": 0,
                "num_ignored_objects": 0,
            },
            "objects": [],
            "ignored_objects": [],
        }

    # Quantize RGB to 8-bit and form composite key
    seg_q = (seg * 255).round().to(torch.long)
    keys = seg_q[:, 0] * 65536 + seg_q[:, 1] * 256 + seg_q[:, 2]

    unique_keys, inverse = torch.unique(keys, return_inverse=True)

    objects = []
    ignored_objects = []

    for i, key in enumerate(unique_keys):
        key_int = key.item()
        mask = inverse == i
        obj_means = means[mask]
        count = obj_means.shape[0]

        r = key_int // 65536
        g = (key_int % 65536) // 256
        b = key_int % 256

        if count < min_gaussians:
            ignored_objects.append({
                "object_id": key_int,
                "rgb": [r, g, b],
                "num_gaussians": count,
                "reason": "below_min_gaussians",
            })
            continue

        # DBSCAN clustering: keep only the largest spatial cluster.
        # Scattered noise Gaussians get discarded.
        if obj_means.shape[0] >= min_gaussians:
            pts_np = obj_means.cpu().numpy()
            db = DBSCAN(eps=dbscan_eps, min_samples=max(3, obj_means.shape[0] // 20))
            labels = db.fit_predict(pts_np)
            # Find largest cluster (label -1 = noise)
            valid_labels = labels[labels >= 0]
            if len(valid_labels) > 0:
                unique_labels, label_counts = np.unique(valid_labels, return_counts=True)
                largest_label = unique_labels[label_counts.argmax()]
                cluster_mask = torch.tensor(labels == largest_label, device=obj_means.device)
                obj_means = obj_means[cluster_mask]

            if obj_means.shape[0] < min_gaussians:
                ignored_objects.append({
                    "object_id": key_int,
                    "rgb": [r, g, b],
                    "num_gaussians": count,
                    "reason": "no_dense_cluster",
                })
                continue

        # Per-axis quantile filtering: trim outlier_percentile/2 from each
        # end of each axis independently, giving tighter AABBs.
        # Use max(percentile, min_trim/N) to ensure at least min_trim points
        # are removed per side even for small objects.
        if outlier_percentile > 0 and obj_means.shape[0] > 5:
            n = obj_means.shape[0]
            min_trim = 2  # always remove at least 2 from each tail
            lo = max(outlier_percentile / 2, min_trim / n)
            hi = 1.0 - lo
            lo = min(lo, 0.4)  # never trim more than 40% per side
            hi = max(hi, 0.6)
            for dim in range(3):
                q_lo = torch.quantile(obj_means[:, dim], lo)
                q_hi = torch.quantile(obj_means[:, dim], hi)
                keep = (obj_means[:, dim] >= q_lo) & (obj_means[:, dim] <= q_hi)
                obj_means = obj_means[keep]

        bbox_min = obj_means.min(dim=0).values
        bbox_max = obj_means.max(dim=0).values
        center = (bbox_min + bbox_max) / 2
        extents = bbox_max - bbox_min

        objects.append({
            "object_id": key_int,
            "rgb": [r, g, b],
            "rgb_hex": f"#{r:02X}{g:02X}{b:02X}",
            "num_gaussians": count,
            "bbox": {
                "min": bbox_min.cpu().tolist(),
                "max": bbox_max.cpu().tolist(),
                "center": center.cpu().tolist(),
                "extents": extents.cpu().tolist(),
            },
        })

    # Sort by number of Gaussians descending for convenience
    objects.sort(key=lambda o: o["num_gaussians"], reverse=True)

    # Merge fragments of the same object caused by float quantization drift
    num_before_merge = len(objects)
    objects = _merge_by_color_proximity(objects, rgb_threshold=rgb_merge_threshold)
    print(f"Merged {num_before_merge} color fragments into {len(objects)} objects "
          f"(rgb_merge_threshold={rgb_merge_threshold})")

    # ------------------------------------------------------------------
    # Convert internal objects list into scene graph format
    # ------------------------------------------------------------------
    color_to_info = _load_color_to_label(scene_dir)

    # Resolve labels and instance IDs for each object
    node_labels = []
    node_instance_ids = []
    for obj in objects:
        label, inst_id = _lookup_label(obj["rgb"], color_to_info)
        node_labels.append(label)
        node_instance_ids.append(inst_id)

    # Build a label -> integer ID mapping for this scene
    type_vocab = _build_type_vocab(node_labels)

    node_positions = [obj["bbox"]["center"] for obj in objects]
    node_bbox_sizes = [obj["bbox"]["extents"] for obj in objects]
    node_rotations = [[0.0, 0.0, 0.0] for _ in objects]  # AABB -> no rotation
    node_type_ids = [type_vocab[label] for label in node_labels]

    edges = _compute_edges(node_positions, node_labels, node_instance_ids)

    return {
        "num_nodes": len(objects),
        "node_type_ids": node_type_ids,
        "node_type_names": node_labels,
        "node_positions": node_positions,
        "node_bbox_sizes": node_bbox_sizes,
        "node_rotations": node_rotations,
        "num_edges": len(edges),
        "edges": edges,
        "metadata": {
            "coordinate_frame": "world",
            "scene_dir": str(scene_dir) if scene_dir is not None else None,
            "opacity_threshold": opacity_threshold,
            "min_gaussians": min_gaussians,
            "rgb_merge_threshold": rgb_merge_threshold,
            "outlier_percentile": outlier_percentile,
            "dbscan_eps": dbscan_eps,
            "total_gaussians": total_gaussians,
            "gaussians_after_filter": gaussians_after_filter,
            "num_objects_before_merge": num_before_merge,
            "num_ignored_objects": len(ignored_objects),
            "type_vocab": type_vocab,
            # Keep raw RGB info for visualization / debugging
            "node_rgbs": [obj["rgb"] for obj in objects],
            "node_num_gaussians": [obj["num_gaussians"] for obj in objects],
            "node_instance_ids": node_instance_ids,
        },
    }


def export_bounding_boxes_to_json(
    gaussians: Gaussians,
    output_path: Path,
    batch_idx: int = 0,
    opacity_threshold: float = 0.1,
    min_gaussians: int = 10,
    scene_dir: str | Path | None = None,
) -> Path:
    """
    Extract bounding boxes and write them to a JSON file.
    Also saves an npz with per-Gaussian positions and segmentation colors
    (after opacity filtering) for detailed visualization.

    Returns:
        Path to the written JSON file.
    """
    result = extract_bounding_boxes(
        gaussians,
        batch_idx=batch_idx,
        opacity_threshold=opacity_threshold,
        min_gaussians=min_gaussians,
        scene_dir=scene_dir,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved scene graph ({result['num_nodes']} nodes, {result['num_edges']} edges) to {output_path}")

    # Save per-Gaussian positions + segmentation colors for visualization
    means = gaussians.means[batch_idx]
    opacities = gaussians.opacities[batch_idx]
    seg = gaussians.segmentation[batch_idx]
    keep = opacities > opacity_threshold
    points_path = output_path.with_suffix(".npz")
    np.savez_compressed(
        points_path,
        means=means[keep].cpu().numpy(),          # (N, 3)
        seg_rgb=((seg[keep] * 255).round().cpu().numpy().astype(np.uint8)),  # (N, 3)
    )
    print(f"Saved {keep.sum().item()} Gaussian positions to {points_path}")

    return output_path
