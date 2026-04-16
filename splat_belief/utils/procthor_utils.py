"""
Utilities for ProcTHOR scene graph processing:
- Build vocabulary of object types and edge types
- Pre-compute CLIP/SigLIP text embeddings for object types
- Scene graph parsing helpers
- Wall line-segment extraction from seen_object_ids
"""
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# Edge type constants
EDGE_TYPE_CONTAINS = 0   # room contains object
EDGE_TYPE_NEAR = 1       # spatial proximity (< threshold)
EDGE_TYPE_SAME_ROOM = 2  # objects in the same room
# Padding edge type
EDGE_TYPE_PAD = -1

# Node type for padding
NODE_TYPE_PAD = 0

# Special type names for structural elements (added beyond scanned object vocabulary)
WALL_TYPE_NAME = "Wall"
FLOOR_TYPE_NAME = "Floor"
DOOR_TYPE_NAME = "Door"

# Minimum AABB half-extent for objects with missing/zero bounding boxes
MIN_AABB_HALF_EXTENT = 0.10  # 10 cm

# Default door dimensions (metres)
DEFAULT_DOOR_WIDTH = 0.90
DEFAULT_DOOR_HEIGHT = 2.10
DEFAULT_DOOR_THICKNESS = 0.50


def scan_vocabulary(dataset_root: str) -> Tuple[Dict[str, int], List[str]]:
    """
    Scan all episodes to build a vocabulary of object types.

    Returns:
        type_to_id: mapping from objectType string to integer ID (0 reserved for padding)
        id_to_type: list where id_to_type[i] is the objectType string for ID i
    """
    all_types = set()
    dataset_root = Path(dataset_root)
    # Find all episode dirs (support flat layout and split subdirs like train/, unit/)
    # NOTE: use glob.glob with recursive=True instead of Path.glob because
    # Path.glob does NOT follow symlinks (e.g. train/ → real path).
    import glob as _glob
    for sg_path in sorted(_glob.glob(str(dataset_root / "**" / "all_scene_graphs.json"), recursive=True)):
        with open(sg_path) as f:
            sg_data = json.load(f)
        for frame_data in sg_data:
            for room in frame_data["scene_graph"]:
                for child in room.get("children", []):
                    all_types.add(child["objectType"])
                    for sub in child.get("children", []):
                        all_types.add(sub["objectType"])

    sorted_types = sorted(all_types)
    # ID 0 is reserved for padding
    id_to_type = ["<pad>"] + sorted_types
    # Append structural types (Wall, Floor, Door) after all scanned object types
    for struct_type in [WALL_TYPE_NAME, FLOOR_TYPE_NAME, DOOR_TYPE_NAME]:
        if struct_type not in all_types:
            id_to_type.append(struct_type)
    type_to_id = {t: i for i, t in enumerate(id_to_type)}
    return type_to_id, id_to_type


def precompute_clip_embeddings(
    id_to_type: List[str],
    output_path: str,
    model_name: str = "openai/clip-vit-base-patch32",
    prompt_template: str = "{obj_type}",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Pre-compute text embeddings for all object types using CLIP or SigLIP.

    Args:
        id_to_type: list of object type strings (index 0 = "<pad>")
        output_path: where to save the embeddings tensor
        model_name: HuggingFace model identifier
        prompt_template: template with {obj_type} placeholder (default: bare name)
        device: device to run inference on

    Returns:
        embeddings: (n_types, D_clip) float32 tensor
    """
    from transformers import CLIPTokenizer, CLIPTextModel

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name).to(device)
    text_model.eval()

    # Build text prompts — bare names give best separability
    prompts = []
    for type_name in id_to_type:
        if type_name == "<pad>":
            prompts.append("")
        else:
            # Convert camelCase to readable: "FloorLamp" -> "floor lamp"
            readable = _camel_to_words(type_name)
            prompts.append(prompt_template.format(obj_type=readable))

    # Encode in batches using pooler_output (before text projection)
    # — gives ~2× better separability than get_text_features()
    batch_size = 128
    all_embs = []
    with torch.no_grad():
        inputs = tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch = {k: v[start:end] for k, v in inputs.items()}
            outputs = text_model(**batch)
            all_embs.append(outputs.pooler_output.cpu())

    embeddings = torch.cat(all_embs, dim=0)

    # Zero out the padding embedding
    embeddings[0] = 0.0

    embeddings = embeddings.cpu().float()
    torch.save(embeddings, output_path)
    print(f"Saved {embeddings.shape} embeddings to {output_path}")
    return embeddings


def precompute_language_embeddings(
    id_to_type: List[str],
    output_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Pre-compute text embeddings using a sentence-transformer model (e.g. MiniLM).

    Args:
        id_to_type: list of object type strings (index 0 = "<pad>")
        output_path: where to save the embeddings tensor
        model_name: sentence-transformers model identifier
        device: device to run inference on

    Returns:
        embeddings: (n_types, D_model) float32 tensor
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    # Build text prompts — bare names (same as CLIP path)
    prompts = []
    for type_name in id_to_type:
        if type_name == "<pad>":
            prompts.append("")
        else:
            readable = _camel_to_words(type_name)
            prompts.append(readable)

    with torch.no_grad():
        emb_np = model.encode(prompts, batch_size=128, show_progress_bar=False)
    embeddings = torch.from_numpy(emb_np).float()

    # Zero out the padding embedding
    embeddings[0] = 0.0

    torch.save(embeddings, output_path)
    print(f"Saved {embeddings.shape} embeddings to {output_path}")
    return embeddings


def _camel_to_words(name: str) -> str:
    """Convert camelCase/PascalCase to space-separated words.
    Strips the 'Obja' provenance prefix from Objaverse asset names.
    E.g. 'FloorLamp' -> 'floor lamp', 'ObjaCarton' -> 'carton'
    """
    import re
    if name.startswith("Obja"):
        name = name[4:]
    words = re.sub(r"([A-Z])", r" \1", name).strip().lower()
    return words


def _clean_display_name(name: str) -> str:
    """Return a human-readable display name for a scene-graph type."""
    if name.startswith("Obja"):
        name = name[4:]
    import re
    return re.sub(r"([A-Z])", r" \1", name).strip()


def save_vocabulary(
    type_to_id: Dict[str, int],
    id_to_type: List[str],
    output_dir: str,
):
    """Save vocabulary mappings as JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "type_to_id.json"), "w") as f:
        json.dump(type_to_id, f, indent=2)
    with open(os.path.join(output_dir, "id_to_type.json"), "w") as f:
        json.dump(id_to_type, f, indent=2)
    print(f"Saved vocabulary ({len(id_to_type)} types) to {output_dir}")


def load_vocabulary(vocab_dir: str) -> Tuple[Dict[str, int], List[str]]:
    """Load vocabulary from JSON files."""
    with open(os.path.join(vocab_dir, "type_to_id.json")) as f:
        type_to_id = json.load(f)
    with open(os.path.join(vocab_dir, "id_to_type.json")) as f:
        id_to_type = json.load(f)
    return type_to_id, id_to_type


def parse_scene_graph(
    frame_data: dict,
    type_to_id: Dict[str, int],
    near_threshold: float = 2.0,
    max_nodes: int = 128,
    max_edges: int = 512,
    include_walls: bool = False,
    wall_height_default: float = 2.5,
    wall_thickness: float = 0.15,
    room_bounds: Optional[Dict[int, dict]] = None,
) -> dict:
    """
    Parse a single frame's scene graph into tensors.

    Args:
        frame_data: one element from all_scene_graphs.json list
        type_to_id: vocabulary mapping
        near_threshold: distance threshold for "near" edges (meters)
        max_nodes: maximum number of nodes (padded)
        max_edges: maximum number of edges (padded)
        include_walls: whether to parse wall line-segments from seen_object_ids
        wall_height_default: fallback ceiling height if not found in data
        wall_thickness: assumed wall thickness for AABB size
        room_bounds: pre-computed room bounds from :func:`collect_room_bounds`
            (needed for door inference; pass ``None`` to skip doors)

    Returns:
        dict with sg_node_types, sg_node_positions, sg_node_rotations,
        sg_node_sizes, sg_edge_index, sg_edge_types, sg_node_mask, sg_edge_mask,
        and when include_walls=True: sg_wall_endpoints, sg_wall_heights, sg_node_is_wall
    """
    nodes = []  # list of (type_id, pos, rot, size)
    room_assignments = []  # room index for each node
    is_wall_flags = []  # bool per node
    is_door_flags = []  # bool per node
    wall_endpoint_list = []  # (2, 3) per node; zeros for objects
    wall_height_list = []  # float per node; 0 for objects

    # Build room_id → room_idx mapping from scene_graph
    room_id_to_idx = {}
    for room_idx, room in enumerate(frame_data["scene_graph"]):
        room_id_str = room.get("id", "")
        # Extract numeric room id from "room|N"
        parts = room_id_str.split("|")
        if len(parts) >= 2:
            try:
                room_id_to_idx[int(parts[1])] = room_idx
            except ValueError:
                pass

    for room_idx, room in enumerate(frame_data["scene_graph"]):
        for child in room.get("children", []):
            _add_node(child, type_to_id, nodes, room_assignments, room_idx)
            is_wall_flags.append(False)
            is_door_flags.append(False)
            wall_endpoint_list.append(np.zeros((2, 3), dtype=np.float32))
            wall_height_list.append(0.0)
            # Also add nested children (objects on top of furniture)
            for sub in child.get("children", []):
                _add_node(sub, type_to_id, nodes, room_assignments, room_idx)
                is_wall_flags.append(False)
                is_door_flags.append(False)
                wall_endpoint_list.append(np.zeros((2, 3), dtype=np.float32))
                wall_height_list.append(0.0)

    # Parse walls if requested
    if include_walls and WALL_TYPE_NAME in type_to_id:
        wall_nodes = _parse_walls(
            frame_data, type_to_id, wall_height_default, wall_thickness,
            room_id_to_idx,
        )

        # Parse doors (needs pre-computed room_bounds from the full episode)
        door_nodes = []
        if DOOR_TYPE_NAME in type_to_id and room_bounds is not None:
            door_nodes = _parse_doors(
                frame_data, type_to_id, room_bounds, room_id_to_idx,
                wall_height_default, wall_nodes=wall_nodes,
            )

        # ---- Split walls at door openings ----
        if door_nodes:
            wall_door_map = _match_doors_to_walls(wall_nodes, door_nodes)
            split_wall_nodes = []
            for wi, wn in enumerate(wall_nodes):
                if wi in wall_door_map:
                    sub_walls = _split_wall_at_doors(wn, wall_door_map[wi])
                    split_wall_nodes.extend(sub_walls)
                else:
                    split_wall_nodes.append(wn)
            wall_nodes = split_wall_nodes

        for wn in wall_nodes:
            nodes.append((wn["type_id"], wn["pos"], wn["rot"], wn["size"]))
            room_assignments.append(wn["room_idx"])
            is_wall_flags.append(True)
            is_door_flags.append(False)
            wall_endpoint_list.append(wn["endpoints"])
            wall_height_list.append(wn["height"])

    # Parse floors (always alongside walls; they use room AABB data)
    if include_walls and FLOOR_TYPE_NAME in type_to_id:
        floor_nodes = _parse_floors(frame_data, type_to_id)
        for fn in floor_nodes:
            nodes.append((fn["type_id"], fn["pos"], fn["rot"], fn["size"]))
            room_assignments.append(fn["room_idx"])
            is_wall_flags.append(False)
            is_door_flags.append(False)
            wall_endpoint_list.append(fn["endpoints"])
            wall_height_list.append(fn["height"])

    # Add door nodes to the scene graph (for GCN/semantic features)
    # but they will be excluded from z-buffer rasterization
    if include_walls and DOOR_TYPE_NAME in type_to_id and room_bounds is not None:
        # door_nodes already parsed above inside the wall block;
        # if walls were not parsed, parse doors here as fallback
        if not (WALL_TYPE_NAME in type_to_id):
            door_nodes = _parse_doors(
                frame_data, type_to_id, room_bounds, room_id_to_idx,
                wall_height_default, wall_nodes=None,
            )
        for dn in door_nodes:
            nodes.append((dn["type_id"], dn["pos"], dn["rot"], dn["size"]))
            room_assignments.append(dn["room_idx"])
            is_wall_flags.append(False)
            is_door_flags.append(True)
            wall_endpoint_list.append(dn["endpoints"])
            wall_height_list.append(dn["height"])

    n_nodes = min(len(nodes), max_nodes)

    # Build edge list
    edges = []  # list of (src, dst, edge_type)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # Same room edge
            if room_assignments[i] == room_assignments[j]:
                # Near edge based on distance
                dist = np.linalg.norm(
                    np.array(nodes[i][1]) - np.array(nodes[j][1])
                )
                if dist < near_threshold:
                    edges.append((i, j, EDGE_TYPE_NEAR))
                    edges.append((j, i, EDGE_TYPE_NEAR))
                else:
                    edges.append((i, j, EDGE_TYPE_SAME_ROOM))
                    edges.append((j, i, EDGE_TYPE_SAME_ROOM))

    n_edges = min(len(edges), max_edges)

    # Build tensors
    node_types = torch.zeros(max_nodes, dtype=torch.long)
    node_positions = torch.zeros(max_nodes, 3, dtype=torch.float32)
    node_rotations = torch.zeros(max_nodes, 3, dtype=torch.float32)
    node_sizes = torch.zeros(max_nodes, 3, dtype=torch.float32)
    node_mask = torch.zeros(max_nodes, dtype=torch.bool)
    node_is_wall = torch.zeros(max_nodes, dtype=torch.bool)
    node_is_door = torch.zeros(max_nodes, dtype=torch.bool)
    wall_endpoints = torch.zeros(max_nodes, 2, 3, dtype=torch.float32)
    wall_heights = torch.zeros(max_nodes, dtype=torch.float32)

    for i in range(n_nodes):
        type_id, pos, rot, size = nodes[i]
        node_types[i] = type_id
        node_positions[i] = torch.tensor(pos, dtype=torch.float32)
        node_rotations[i] = torch.tensor(rot, dtype=torch.float32)
        node_sizes[i] = torch.tensor(size, dtype=torch.float32)
        node_mask[i] = True
        node_is_wall[i] = is_wall_flags[i] if i < len(is_wall_flags) else False
        node_is_door[i] = is_door_flags[i] if i < len(is_door_flags) else False
        if i < len(wall_endpoint_list):
            wall_endpoints[i] = torch.from_numpy(wall_endpoint_list[i])
        if i < len(wall_height_list):
            wall_heights[i] = wall_height_list[i]

    edge_index = torch.zeros(2, max_edges, dtype=torch.long)
    edge_types = torch.zeros(max_edges, dtype=torch.long)
    edge_mask = torch.zeros(max_edges, dtype=torch.bool)

    for i in range(n_edges):
        src, dst, etype = edges[i]
        edge_index[0, i] = src
        edge_index[1, i] = dst
        edge_types[i] = etype
        edge_mask[i] = True

    result = {
        "sg_node_types": node_types,
        "sg_node_positions": node_positions,
        "sg_node_rotations": node_rotations,
        "sg_node_sizes": node_sizes,
        "sg_edge_index": edge_index,
        "sg_edge_types": edge_types,
        "sg_node_mask": node_mask,
        "sg_edge_mask": edge_mask,
    }

    if include_walls:
        result["sg_wall_endpoints"] = wall_endpoints
        result["sg_wall_heights"] = wall_heights
        result["sg_node_is_wall"] = node_is_wall
        result["sg_node_is_door"] = node_is_door

    return result


def _add_node(
    obj: dict,
    type_to_id: Dict[str, int],
    nodes: list,
    room_assignments: list,
    room_idx: int,
):
    """Extract node attributes from an object dict and append to lists."""
    obj_type = obj.get("objectType", "Unknown")
    type_id = type_to_id.get(obj_type, NODE_TYPE_PAD)

    pos = obj.get("position", {"x": 0, "y": 0, "z": 0})
    # Negate z: ProcTHOR JSON uses Unity left-handed coords (z-forward),
    # but camera poses in all_poses.npz use right-handed coords (z-negated).
    pos = [pos["x"], pos["y"], -pos["z"]]

    rot = obj.get("rotation", {"x": 0, "y": 0, "z": 0})
    rot = [rot["x"], rot["y"], rot["z"]]

    aabb = obj.get("axisAlignedBoundingBox", {})
    if "size" in aabb:
        size_dict = aabb["size"]
        size = [size_dict["x"], size_dict["y"], size_dict["z"]]
    elif "cornerPoints" in aabb:
        corners = np.array(aabb["cornerPoints"])
        size = (corners.max(axis=0) - corners.min(axis=0)).tolist()
    else:
        size = [0.0, 0.0, 0.0]

    # Enforce minimum AABB so objects with missing geometry still appear
    for dim in range(3):
        if size[dim] < MIN_AABB_HALF_EXTENT * 2:
            size[dim] = MIN_AABB_HALF_EXTENT * 2

    nodes.append((type_id, pos, rot, size))
    room_assignments.append(room_idx)


def _parse_walls(
    frame_data: dict,
    type_to_id: Dict[str, int],
    wall_height_default: float,
    wall_thickness: float,
    room_id_to_idx: Dict[int, int],
) -> List[dict]:
    """
    Extract wall line-segments and ceiling heights from seen_object_ids.

    Wall IDs follow the format: wall|<room_id>|<x1>|<z1>|<x2>|<z2>
    Ceiling IDs follow: Ceiling_room|<room_id>|0|<height>|0

    Returns list of wall node dicts, each with:
        type_id, pos, rot, size, room_idx, endpoints (2,3), height
    """
    wall_type_id = type_to_id.get(WALL_TYPE_NAME, NODE_TYPE_PAD)
    seen_ids = frame_data.get("seen_object_ids", [])

    # First pass: extract ceiling heights per room
    room_ceil_height: Dict[int, float] = {}
    for obj_id in seen_ids:
        if not isinstance(obj_id, str):
            continue
        if obj_id.startswith("Ceiling_room|"):
            parts = obj_id.split("|")
            if len(parts) >= 4:
                try:
                    rid = int(parts[1])
                    h = float(parts[3])
                    room_ceil_height[rid] = h
                except (ValueError, IndexError):
                    pass

    # Second pass: extract wall line-segments
    wall_nodes = []
    for obj_id in seen_ids:
        if not isinstance(obj_id, str):
            continue
        if not obj_id.startswith("wall|"):
            continue
        parts = obj_id.split("|")
        if len(parts) < 6:
            continue
        try:
            rid = int(parts[1])
            x1, z1 = float(parts[2]), float(parts[3])
            x2, z2 = float(parts[4]), float(parts[5])
        except (ValueError, IndexError):
            continue

        ceil_h = room_ceil_height.get(rid, wall_height_default)
        room_idx = room_id_to_idx.get(rid, 0)

        # Apply z-negation (Unity left-handed → right-handed)
        p1 = np.array([x1, 0.0, -z1], dtype=np.float32)
        p2 = np.array([x2, 0.0, -z2], dtype=np.float32)

        # Midpoint position (center of wall at half height)
        mid = (p1 + p2) / 2.0
        pos = [float(mid[0]), ceil_h / 2.0, float(mid[2])]

        # Rotation: angle of segment in XZ plane
        dx = p2[0] - p1[0]
        dz = p2[2] - p1[2]
        angle_y = math.degrees(math.atan2(dx, dz))  # rotation about Y axis
        rot = [0.0, angle_y, 0.0]

        # Size: (length, height, thickness)
        length = float(np.linalg.norm(p2 - p1))
        size = [length, ceil_h, wall_thickness]

        # Raw endpoints for spatial bias (bottom corners in world space)
        endpoints = np.stack([p1, p2], axis=0)  # (2, 3)

        wall_nodes.append({
            "type_id": wall_type_id,
            "pos": pos,
            "rot": rot,
            "size": size,
            "room_idx": room_idx,
            "endpoints": endpoints,
            "height": ceil_h,
        })

    return wall_nodes


def _parse_floors(
    frame_data: dict,
    type_to_id: Dict[str, int],
    floor_thickness: float = 0.05,
) -> List[dict]:
    """
    Extract floor quads from room AABBs in the scene graph.

    Each room in ``frame_data["scene_graph"]`` has an ``axisAlignedBoundingBox``
    covering the floor area (y size ≈ 0).  We turn each into a thin slab node
    so it can appear in the layout rasterization.

    Returns list of floor node dicts matching the wall-node interface
    (type_id, pos, rot, size, room_idx, endpoints, height).
    """
    floor_type_id = type_to_id.get(FLOOR_TYPE_NAME, NODE_TYPE_PAD)
    floor_nodes = []

    for room_idx, room in enumerate(frame_data["scene_graph"]):
        aabb = room.get("axisAlignedBoundingBox", {})
        center = aabb.get("center", None)
        size = aabb.get("size", None)
        if center is None or size is None:
            continue

        cx = center["x"]
        cy = 0.0  # floor sits at y = 0
        cz = -center["z"]  # z-negation
        sx = size["x"]
        sy = floor_thickness  # give the floor a small height so the AABB is not flat
        sz = size["z"]

        # Enforce minimum extent
        if sx < MIN_AABB_HALF_EXTENT * 2:
            sx = MIN_AABB_HALF_EXTENT * 2
        if sz < MIN_AABB_HALF_EXTENT * 2:
            sz = MIN_AABB_HALF_EXTENT * 2

        floor_nodes.append({
            "type_id": floor_type_id,
            "pos": [cx, cy, cz],
            "rot": [0.0, 0.0, 0.0],
            "size": [sx, sy, sz],
            "room_idx": room_idx,
            "endpoints": np.zeros((2, 3), dtype=np.float32),  # unused for floors
            "height": 0.0,
        })

    return floor_nodes


def collect_room_bounds(sg_data: list) -> Dict[int, dict]:
    """
    Scan **all** frames of an episode to build a map of room floor-polygon
    bounding rectangles.

    Also infers bounds for rooms that are never entered (no ``floorPolygon``)
    but are referenced by visible wall segments (``wall|<room_id>|...``).

    Args:
        sg_data: the full ``all_scene_graphs.json`` list for one episode.

    Returns:
        Dictionary ``{room_int_id: {x_min, x_max, z_min, z_max}}``
        in *original* Unity coords (before z-negation).
    """
    room_bounds: Dict[int, dict] = {}
    # ---- Primary: floor polygons from scene_graph rooms ----
    for frame_data in sg_data:
        for room in frame_data["scene_graph"]:
            rid_str = room.get("id", "")
            parts = rid_str.split("|")
            if len(parts) < 2:
                continue
            try:
                rid = int(parts[1])
            except ValueError:
                continue
            if rid in room_bounds:
                continue  # already have this room's bounds
            fp = room.get("floorPolygon", [])
            if not fp:
                continue
            xs = [p["x"] for p in fp]
            zs = [p["z"] for p in fp]
            room_bounds[rid] = {
                "x_min": min(xs), "x_max": max(xs),
                "z_min": min(zs), "z_max": max(zs),
            }

    # ---- Fallback: infer bounds from wall segment endpoints ----
    # For rooms never entered (no floorPolygon), accumulate wall coords.
    # wall IDs: wall|<room_id>|<x1>|<z1>|<x2>|<z2>
    wall_coords: Dict[int, List[float]] = {}  # rid → [x1, z1, x2, z2, ...]
    for frame_data in sg_data:
        for obj_id in frame_data.get("seen_object_ids", []):
            if not isinstance(obj_id, str) or not obj_id.startswith("wall|"):
                continue
            parts = obj_id.split("|")
            if len(parts) < 6:
                continue
            try:
                rid = int(parts[1])
                if rid in room_bounds:
                    continue  # already have polygon bounds
                x1, z1 = float(parts[2]), float(parts[3])
                x2, z2 = float(parts[4]), float(parts[5])
            except ValueError:
                continue
            wall_coords.setdefault(rid, []).extend([x1, z1, x2, z2])

    for rid, coords in wall_coords.items():
        xs = coords[0::2]  # every other element starting at 0
        zs = coords[1::2]  # every other element starting at 1
        room_bounds[rid] = {
            "x_min": min(xs), "x_max": max(xs),
            "z_min": min(zs), "z_max": max(zs),
        }

    return room_bounds


def _parse_doors(
    frame_data: dict,
    type_to_id: Dict[str, int],
    room_bounds: Dict[int, dict],
    room_id_to_idx: Dict[int, int],
    wall_height_default: float = 2.5,
    wall_nodes: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Infer door positions from ``seen_object_ids`` entries of the form
    ``door|<room1_id>|<room2_id>``.

    **Strategy (preferred):** If *wall_nodes* are provided, find the wall
    segment that separates rooms *r1* and *r2* by checking which wall from
    one room lies on the boundary of the other.  The door is placed at the
    midpoint of that wall segment — much more precise than room-AABB
    heuristics.

    **Fallback:** If no matching wall is found (or *wall_nodes* is ``None``),
    fall back to the room-AABB shared-boundary heuristic.

    Args:
        frame_data: single frame dict from ``all_scene_graphs.json``.
        type_to_id: vocabulary mapping.
        room_bounds: pre-computed room bounds from :func:`collect_room_bounds`.
        room_id_to_idx: room int id → room index in current frame's scene_graph.
        wall_height_default: ceiling height fallback (for door height cap).
        wall_nodes: already-parsed wall node dicts from :func:`_parse_walls`.

    Returns:
        List of door node dicts with keys:
        ``type_id, pos, rot, size, room_idx, endpoints, height``
    """
    door_type_id = type_to_id.get(DOOR_TYPE_NAME, NODE_TYPE_PAD)
    if door_type_id == NODE_TYPE_PAD:
        return []

    seen_ids = frame_data.get("seen_object_ids", [])
    door_nodes = []

    for obj_id in seen_ids:
        if not isinstance(obj_id, str) or not obj_id.startswith("door|"):
            continue
        parts = obj_id.split("|")
        if len(parts) < 3:
            continue
        try:
            r1 = int(parts[1])
            r2 = int(parts[2])
        except ValueError:
            continue

        b1 = room_bounds.get(r1)
        b2 = room_bounds.get(r2)
        if b1 is None or b2 is None:
            continue

        door_cx = None
        door_cz = None
        rot_y = 0.0

        # ---------- Primary: locate door via wall geometry ----------
        # Find walls belonging to room r1 or r2 that lie on the boundary
        # of the *other* room.  A wall is "on the boundary" if its
        # midpoint's perpendicular coordinate is within tolerance of the
        # other room's AABB edge AND it overlaps in the parallel axis.
        if wall_nodes:
            r1_idx = room_id_to_idx.get(r1)
            r2_idx = room_id_to_idx.get(r2)
            tol_wall = 0.3
            best_wall = None
            best_wall_len = float("inf")

            for wn in wall_nodes:
                w_room = wn["room_idx"]
                if r1_idx is not None and w_room == r1_idx:
                    other_bounds = b2
                elif r2_idx is not None and w_room == r2_idx:
                    other_bounds = b1
                else:
                    continue

                # Wall endpoints in original Unity coords (before z-negate):
                # endpoints are already z-negated, so reverse to compare
                # with room_bounds (which are in Unity coords).
                ep = wn["endpoints"]  # (2, 3) — already z-negated
                wx1, wz1 = ep[0][0], -ep[0][2]  # undo z-negate
                wx2, wz2 = ep[1][0], -ep[1][2]

                # Check if wall is approximately on a boundary of the
                # other room (perpendicular coord within tolerance of an
                # AABB face, parallel extent overlaps other room).
                dx = abs(wx2 - wx1)
                dz = abs(wz2 - wz1)

                on_boundary = False
                if dz > dx:
                    # Wall runs along Z; check if x-coord matches boundary
                    wall_x = (wx1 + wx2) / 2.0
                    for bx in [other_bounds["x_min"], other_bounds["x_max"]]:
                        if abs(wall_x - bx) < tol_wall:
                            # Check Z overlap with other room
                            wz_lo, wz_hi = min(wz1, wz2), max(wz1, wz2)
                            oz_lo, oz_hi = other_bounds["z_min"], other_bounds["z_max"]
                            if wz_hi > oz_lo and wz_lo < oz_hi:
                                on_boundary = True
                                break
                else:
                    # Wall runs along X; check if z-coord matches boundary
                    wall_z = (wz1 + wz2) / 2.0
                    for bz in [other_bounds["z_min"], other_bounds["z_max"]]:
                        if abs(wall_z - bz) < tol_wall:
                            # Check X overlap with other room
                            wx_lo, wx_hi = min(wx1, wx2), max(wx1, wx2)
                            ox_lo, ox_hi = other_bounds["x_min"], other_bounds["x_max"]
                            if wx_hi > ox_lo and wx_lo < ox_hi:
                                on_boundary = True
                                break

                if on_boundary:
                    seg_len = math.sqrt((wx2 - wx1)**2 + (wz2 - wz1)**2)
                    if seg_len < best_wall_len:
                        best_wall_len = seg_len
                        best_wall = wn

            if best_wall is not None:
                # Place door at the midpoint of the wall segment
                ep = best_wall["endpoints"]  # (2, 3), z-negated
                mid = (ep[0] + ep[1]) / 2.0
                door_cx = float(mid[0])
                door_cz = float(-mid[2])  # undo z-negate for Unity coords

                # Rotation from wall direction
                seg_dx = ep[1][0] - ep[0][0]
                seg_dz = ep[1][2] - ep[0][2]
                rot_y = math.degrees(math.atan2(seg_dx, seg_dz))

        # ---------- Fallback: room-AABB shared-boundary heuristic ----------
        if door_cx is None:
            tol = 0.3
            candidates = []  # (cx, cz, rot_y, extent)

            def _add_x_edge(bx_val, ba, bb):
                if bb["x_min"] - tol < bx_val < bb["x_max"] + tol:
                    z_lo = max(ba["z_min"], bb["z_min"])
                    z_hi = min(ba["z_max"], bb["z_max"])
                    if z_hi > z_lo:
                        candidates.append((bx_val, (z_lo + z_hi) / 2.0, 90.0, z_hi - z_lo))

            def _add_z_edge(bz_val, ba, bb):
                if bb["z_min"] - tol < bz_val < bb["z_max"] + tol:
                    x_lo = max(ba["x_min"], bb["x_min"])
                    x_hi = min(ba["x_max"], bb["x_max"])
                    if x_hi > x_lo:
                        candidates.append(((x_lo + x_hi) / 2.0, bz_val, 0.0, x_hi - x_lo))

            _add_x_edge(b1["x_min"], b1, b2)
            _add_x_edge(b1["x_max"], b1, b2)
            _add_z_edge(b1["z_min"], b1, b2)
            _add_z_edge(b1["z_max"], b1, b2)
            _add_x_edge(b2["x_min"], b2, b1)
            _add_x_edge(b2["x_max"], b2, b1)
            _add_z_edge(b2["z_min"], b2, b1)
            _add_z_edge(b2["z_max"], b2, b1)

            if not candidates:
                continue

            filtered = []
            for cx, cz, ry, ext in candidates:
                if ry == 90.0:
                    both_min = abs(cx - b1["x_min"]) < tol and abs(cx - b2["x_min"]) < tol
                    both_max = abs(cx - b1["x_max"]) < tol and abs(cx - b2["x_max"]) < tol
                    if both_min or both_max:
                        continue
                else:
                    both_min = abs(cz - b1["z_min"]) < tol and abs(cz - b2["z_min"]) < tol
                    both_max = abs(cz - b1["z_max"]) < tol and abs(cz - b2["z_max"]) < tol
                    if both_min or both_max:
                        continue
                filtered.append((cx, cz, ry, ext))

            if not filtered:
                filtered = candidates

            best = min(filtered, key=lambda c: c[3])
            door_cx, door_cz, rot_y = best[0], best[1], best[2]

        door_h = min(DEFAULT_DOOR_HEIGHT, wall_height_default)
        # Z-negate for right-handed coords
        pos = [door_cx, door_h / 2.0, -door_cz]
        rot = [0.0, rot_y, 0.0]
        # Orient AABB: width goes along the opening direction
        # For wall-based positioning, use the wall direction to orient the door
        abs_rot = abs(rot_y)
        if 45.0 < abs_rot < 135.0:
            # Door opening roughly along Z → width in Z, thickness in X
            size = [DEFAULT_DOOR_THICKNESS, door_h, DEFAULT_DOOR_WIDTH]
        else:
            # Door opening roughly along X → width in X, thickness in Z
            size = [DEFAULT_DOOR_WIDTH, door_h, DEFAULT_DOOR_THICKNESS]

        # Assign to whichever room is in the current frame; fallback to 0
        room_idx = room_id_to_idx.get(r1, room_id_to_idx.get(r2, 0))

        door_nodes.append({
            "type_id": door_type_id,
            "pos": pos,
            "rot": rot,
            "size": size,
            "room_idx": room_idx,
            "endpoints": np.zeros((2, 3), dtype=np.float32),
            "height": 0.0,
        })

    return door_nodes


def _match_doors_to_walls(
    wall_nodes: List[dict],
    door_nodes: List[dict],
    tolerance: float = 0.5,
) -> Dict[int, List[dict]]:
    """
    Match each door to the wall segment it sits on.

    For each door, project its center onto every wall line-segment (in the XZ
    plane).  If the perpendicular distance is within *tolerance* and the
    projection falls within the segment, the door is assigned to that wall.

    Args:
        wall_nodes: list from ``_parse_walls()``.
        door_nodes: list from ``_parse_doors()``.
        tolerance: maximum perpendicular distance (meters) for a match.

    Returns:
        dict ``{wall_index: [door_info, ...]}`` where each *door_info* has:
        ``t`` (parametric position 0→1 along wall), ``half_width``, ``height``
    """
    wall_door_map: Dict[int, List[dict]] = {}

    for door in door_nodes:
        door_pos = np.array(door["pos"], dtype=np.float32)  # already z-negated
        door_xz = np.array([door_pos[0], door_pos[2]], dtype=np.float32)

        # Door AABB width along the opening direction
        # Orientation: rot_y == 0 → opening along X, rot_y == 90 → along Z
        rot_y = door["rot"][1]
        if abs(rot_y - 90.0) < 1.0:
            door_half_w = door["size"][2] / 2.0  # Z extent
        else:
            door_half_w = door["size"][0] / 2.0  # X extent
        door_height = door["size"][1]

        best_wall_idx = -1
        best_dist = float("inf")
        best_t = 0.5

        for wi, wall in enumerate(wall_nodes):
            # Wall endpoints: (2, 3) in world coords (already z-negated)
            wp1 = wall["endpoints"][0]  # (3,)
            wp2 = wall["endpoints"][1]
            a = np.array([wp1[0], wp1[2]], dtype=np.float32)
            b = np.array([wp2[0], wp2[2]], dtype=np.float32)

            ab = b - a
            seg_len_sq = float(np.dot(ab, ab))
            if seg_len_sq < 1e-8:
                continue

            # Parametric projection of door center onto wall segment
            ap = door_xz - a
            t_param = float(np.dot(ap, ab)) / seg_len_sq
            t_clamped = max(0.0, min(1.0, t_param))

            # Closest point on wall segment
            proj = a + t_clamped * ab
            dist = float(np.linalg.norm(door_xz - proj))

            if dist < best_dist:
                best_dist = dist
                best_wall_idx = wi
                best_t = t_clamped

        if best_wall_idx >= 0 and best_dist < tolerance:
            wall_door_map.setdefault(best_wall_idx, []).append({
                "t": best_t,
                "half_width": door_half_w,
                "height": door_height,
            })

    return wall_door_map


def _split_wall_at_doors(wall_node: dict, door_infos: List[dict]) -> List[dict]:
    """
    Split a wall segment into sub-walls that leave openings for doors.

    Produces:
      - **Side pieces**: wall sections between consecutive door edges and
        wall endpoints (full ceiling height).
      - **Lintel pieces**: for each door, a horizontal strip above the door
        opening (from door top to ceiling).

    Args:
        wall_node: one wall node dict from ``_parse_walls()``.
        door_infos: list of ``{t, half_width, height}`` from
            ``_match_doors_to_walls()`` for this wall.

    Returns:
        list of sub-wall node dicts (same schema as ``_parse_walls()`` output).
    """
    p1 = wall_node["endpoints"][0].copy()  # (3,) float32
    p2 = wall_node["endpoints"][1].copy()

    seg_vec = p2 - p1
    seg_len = float(np.linalg.norm(seg_vec))
    if seg_len < 1e-6:
        return [wall_node]

    ceil_h = wall_node["height"]
    wall_type_id = wall_node["type_id"]
    wall_room_idx = wall_node["room_idx"]
    wall_thickness = wall_node["size"][2]  # thickness dimension unchanged

    seg_dir = seg_vec / seg_len  # unit direction along the wall

    # Convert door parametric positions to absolute distance along segment
    # and sort by position
    doors = []
    for di in door_infos:
        center_dist = di["t"] * seg_len
        half_w = di["half_width"]
        left = max(0.0, center_dist - half_w)
        right = min(seg_len, center_dist + half_w)
        door_top = di["height"]  # height of door top (e.g. 2.1m)
        if left < right:
            doors.append((left, right, door_top))

    doors.sort(key=lambda d: d[0])

    # Merge overlapping door intervals
    merged = []
    for left, right, dtop in doors:
        if merged and left <= merged[-1][1]:
            # Overlap → merge; take the max right and max door height
            prev_l, prev_r, prev_top = merged[-1]
            merged[-1] = (prev_l, max(prev_r, right), max(prev_top, dtop))
        else:
            merged.append((left, right, dtop))

    def _make_sub_wall(sp1: np.ndarray, sp2: np.ndarray, bottom_y: float,
                       top_y: float) -> Optional[dict]:
        """Create a sub-wall dict from two endpoints and height range."""
        sub_len = float(np.linalg.norm(sp2 - sp1))
        if sub_len < MIN_AABB_HALF_EXTENT:
            return None
        sub_h = top_y - bottom_y
        if sub_h < MIN_AABB_HALF_EXTENT:
            return None
        mid = (sp1 + sp2) / 2.0
        center_y = (bottom_y + top_y) / 2.0
        pos = [float(mid[0]), center_y, float(mid[2])]

        dx = sp2[0] - sp1[0]
        dz = sp2[2] - sp1[2]
        angle_y = math.degrees(math.atan2(dx, dz))
        rot = [0.0, angle_y, 0.0]

        size = [sub_len, sub_h, wall_thickness]

        # Endpoints for wall-quad projection (bottom corners)
        ep1 = sp1.copy()
        ep1[1] = bottom_y
        ep2 = sp2.copy()
        ep2[1] = bottom_y
        endpoints = np.stack([ep1, ep2], axis=0)

        return {
            "type_id": wall_type_id,
            "pos": pos,
            "rot": rot,
            "size": size,
            "room_idx": wall_room_idx,
            "endpoints": endpoints,
            "height": sub_h,
        }

    sub_walls = []
    cursor = 0.0  # current position along segment

    for d_left, d_right, d_top in merged:
        # --- Side piece: from cursor to door left edge (full height) ---
        if d_left > cursor + MIN_AABB_HALF_EXTENT:
            sp1 = p1 + cursor * seg_dir
            sp2 = p1 + d_left * seg_dir
            sw = _make_sub_wall(sp1, sp2, 0.0, ceil_h)
            if sw is not None:
                sub_walls.append(sw)

        # --- Lintel piece: above door opening (door_top → ceiling) ---
        if d_top < ceil_h - MIN_AABB_HALF_EXTENT:
            lp1 = p1 + d_left * seg_dir
            lp2 = p1 + d_right * seg_dir
            sw = _make_sub_wall(lp1, lp2, d_top, ceil_h)
            if sw is not None:
                sub_walls.append(sw)

        cursor = d_right

    # --- Final side piece: from last door right edge to wall end ---
    if cursor < seg_len - MIN_AABB_HALF_EXTENT:
        sp1 = p1 + cursor * seg_dir
        sp2 = p2.copy()
        sw = _make_sub_wall(sp1, sp2, 0.0, ceil_h)
        if sw is not None:
            sub_walls.append(sw)

    # If splitting produced nothing (e.g. door covers entire wall), return empty
    # so the original wall is effectively removed.
    return sub_walls


# ---------------------------------------------------------------------------
# Dense Layout Rasterization: Scene Graph → Per-Pixel Class + Depth Maps
# ---------------------------------------------------------------------------

def rasterize_scene_graph(
    sg_node_types: torch.Tensor,       # (B, M) long — class IDs
    sg_node_positions: torch.Tensor,   # (B, M, 3) world-space centers
    sg_node_sizes: torch.Tensor,       # (B, M, 3) AABB full extents
    sg_node_mask: torch.Tensor,        # (B, M) bool
    camera_c2w: torch.Tensor,          # (B, T, 4, 4) absolute cam-to-world
    intrinsics: torch.Tensor,          # (B, 3, 3) normalised intrinsics
    output_h: int = 32,
    output_w: int = 32,
    sg_wall_endpoints: Optional[torch.Tensor] = None,  # (B, M, 2, 3)
    sg_wall_heights: Optional[torch.Tensor] = None,    # (B, M)
    sg_node_is_wall: Optional[torch.Tensor] = None,    # (B, M) bool
    sg_node_is_door: Optional[torch.Tensor] = None,    # (B, M) bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize the scene graph into per-pixel semantic class + depth maps.

    For each camera view, projects every node's bounding geometry (AABB for
    objects, wall quad for walls) into 2D, creates a coverage mask, then
    uses a Z-buffer (argmin depth) to pick the closest node per pixel.

    Runs on GPU inside ``torch.no_grad()``.

    Returns:
        layout_cls:   (B*T, output_h, output_w) long — class ID per pixel (0 = background)
        layout_depth: (B*T, output_h, output_w) float — camera-space depth (0 = background)
    """
    B, M = sg_node_types.shape
    T = camera_c2w.shape[1]
    device = sg_node_types.device
    dtype = sg_node_positions.dtype

    # --- world → camera transform ---
    w2c = torch.inverse(camera_c2w)                          # (B, T, 4, 4)
    R = w2c[:, :, :3, :3]                                    # (B, T, 3, 3)
    t_vec = w2c[:, :, :3, 3]                                 # (B, T, 3)

    # --- intrinsics ---
    fx = intrinsics[:, 0, 0]                                 # (B,)
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    # ===== Compute per-node 2D bbox + mean depth for each view =====

    # -- Objects: project 8 AABB corners --
    half = sg_node_sizes / 2                                 # (B, M, 3)
    signs = sg_node_positions.new_tensor(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
         [ 1, -1, -1], [ 1, -1, 1], [ 1, 1, -1], [ 1, 1, 1]]
    )                                                        # (8, 3)
    offsets = half.unsqueeze(2) * signs                      # (B, M, 8, 3)
    corners = sg_node_positions.unsqueeze(2) + offsets       # (B, M, 8, 3)

    # Transform to camera space: (B, M, 8, 3) → (B, T, M, 8, 3)
    corners_exp = corners.unsqueeze(1).expand(-1, T, -1, -1, -1)
    p_cam = torch.einsum("btij,btmcj->btmci", R, corners_exp) \
            + t_vec[:, :, None, None, :]                     # (B, T, M, 8, 3)

    z_corners = p_cam[..., 2]                                # (B, T, M, 8)
    visible = z_corners > 0                                  # (B, T, M, 8)
    z_safe = z_corners.clamp(min=1e-4)

    # Project to normalised [0, 1] image coords
    u_c = fx[:, None, None, None] * (p_cam[..., 0] / z_safe) \
          + cx[:, None, None, None]                          # (B, T, M, 8)
    v_c = fy[:, None, None, None] * (p_cam[..., 1] / z_safe) \
          + cy[:, None, None, None]

    INF = torch.tensor(float("inf"), device=device, dtype=dtype)  # scalar; no allocation pressure
    u_for_min = torch.where(visible, u_c, INF)
    u_for_max = torch.where(visible, u_c, -INF)
    v_for_min = torch.where(visible, v_c, INF)
    v_for_max = torch.where(visible, v_c, -INF)

    u_min = u_for_min.min(dim=-1).values                    # (B, T, M)
    u_max = u_for_max.max(dim=-1).values
    v_min = v_for_min.min(dim=-1).values
    v_max = v_for_max.max(dim=-1).values

    # --- Near-plane clipping for AABB edges ---
    # If an edge connects a visible corner (z>0) to a behind-camera corner
    # (z<=0), interpolate to z=NEAR_Z and project the intersection point.
    # This prevents partial-visibility nodes from having a shrunk bbox.
    NEAR_Z = 0.05
    aabb_edge_idx = torch.tensor(
        [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]],
        device=device, dtype=torch.long,
    )  # (12, 2) — edges of the AABB (corners differing in 1 coordinate)
    ea = p_cam[:, :, :, aabb_edge_idx[:, 0], :]              # (B, T, M, 12, 3)
    eb = p_cam[:, :, :, aabb_edge_idx[:, 1], :]
    za, zb = ea[..., 2], eb[..., 2]                          # (B, T, M, 12)
    crosses = (za > NEAR_Z) != (zb > NEAR_Z)
    if crosses.any():
        dz_e = zb - za
        safe_dz = torch.where(dz_e.abs() > 1e-8, dz_e, torch.ones_like(dz_e))
        t_clip = ((NEAR_Z - za) / safe_dz).clamp(0.0, 1.0)
        clip_pt = ea + t_clip.unsqueeze(-1) * (eb - ea)      # (B, T, M, 12, 3)
        clip_z = clip_pt[..., 2].clamp(min=NEAR_Z)
        clip_u = fx[:, None, None, None] * (clip_pt[..., 0] / clip_z) \
                 + cx[:, None, None, None]
        clip_v = fy[:, None, None, None] * (clip_pt[..., 1] / clip_z) \
                 + cy[:, None, None, None]
        # Clamp clip projections to [0, 1] to prevent bbox explosion
        clip_u = clip_u.clamp(0.0, 1.0)
        clip_v = clip_v.clamp(0.0, 1.0)
        u_min = torch.min(u_min, torch.where(crosses, clip_u, INF).min(dim=-1).values)
        u_max = torch.max(u_max, torch.where(crosses, clip_u, -INF).max(dim=-1).values)
        v_min = torch.min(v_min, torch.where(crosses, clip_v, INF).min(dim=-1).values)
        v_max = torch.max(v_max, torch.where(crosses, clip_v, -INF).max(dim=-1).values)

    # Mean depth of visible corners
    z_sum = (z_safe * visible.float()).sum(dim=-1)           # (B, T, M)
    z_count = visible.float().sum(dim=-1).clamp(min=1.0)
    mean_depth = z_sum / z_count                             # (B, T, M)

    # Mean UV of visible corners (for depth gradient computation)
    vis_f = visible.float()                                  # (B, T, M, 8)
    u_sum = (u_c * vis_f).sum(dim=-1)                        # (B, T, M)
    v_sum = (v_c * vis_f).sum(dim=-1)
    u_mean = u_sum / z_count                                 # (B, T, M)
    v_mean = v_sum / z_count

    # Depth gradient (dz/du, dz/dv) via weighted least-squares on visible corners.
    # Model: z ≈ z_mean + dz_du * (u − u_mean) + dz_dv * (v − v_mean)
    du = u_c - u_mean.unsqueeze(-1)                          # (B, T, M, 8)
    dv = v_c - v_mean.unsqueeze(-1)
    dz = z_safe - mean_depth.unsqueeze(-1)

    Wuu = (vis_f * du * du).sum(dim=-1)                      # (B, T, M)
    Wvv = (vis_f * dv * dv).sum(dim=-1)
    Wuv = (vis_f * du * dv).sum(dim=-1)
    Wuz = (vis_f * du * dz).sum(dim=-1)
    Wvz = (vis_f * dv * dz).sum(dim=-1)

    det = (Wuu * Wvv - Wuv * Wuv).clamp(min=1e-10)
    grad_u = (Wvv * Wuz - Wuv * Wvz) / det                  # dz/du  (B, T, M)
    grad_v = (Wuu * Wvz - Wuv * Wuz) / det                  # dz/dv

    any_visible = visible.any(dim=-1)                        # (B, T, M)

    # -- Walls: override bbox for wall nodes with wall quad projection --
    if sg_wall_endpoints is not None and sg_wall_heights is not None \
            and sg_node_is_wall is not None:
        # Build 4 quad corners: bottom-p1, bottom-p2, top-p1, top-p2
        w_p1 = sg_wall_endpoints[:, :, 0, :]                # (B, M, 3)
        w_p2 = sg_wall_endpoints[:, :, 1, :]                # (B, M, 3)
        y_off = torch.zeros_like(w_p1)
        y_off[..., 1] = sg_wall_heights                     # ceiling height in Y
        w_corners = torch.stack([w_p1, w_p2, w_p1 + y_off, w_p2 + y_off], dim=2)  # (B, M, 4, 3)

        # Project wall corners
        wc_exp = w_corners.unsqueeze(1).expand(-1, T, -1, -1, -1)
        wp_cam = torch.einsum("btij,btmcj->btmci", R, wc_exp) \
                 + t_vec[:, :, None, None, :]                # (B, T, M, 4, 3)
        wz = wp_cam[..., 2]                                  # (B, T, M, 4)
        w_vis = wz > 0
        wz_safe = wz.clamp(min=1e-4)

        wu = fx[:, None, None, None] * (wp_cam[..., 0] / wz_safe) \
             + cx[:, None, None, None]
        wv = fy[:, None, None, None] * (wp_cam[..., 1] / wz_safe) \
             + cy[:, None, None, None]

        wu_min = torch.where(w_vis, wu, INF).min(dim=-1).values
        wu_max = torch.where(w_vis, wu, -INF).max(dim=-1).values
        wv_min = torch.where(w_vis, wv, INF).min(dim=-1).values
        wv_max = torch.where(w_vis, wv, -INF).max(dim=-1).values

        # Near-plane clipping for wall quad edges
        wall_edge_idx = torch.tensor(
            [[0,1],[0,2],[1,3],[2,3]], device=device, dtype=torch.long,
        )  # 4 edges of the quad
        wea = wp_cam[:, :, :, wall_edge_idx[:, 0], :]        # (B, T, M, 4, 3)
        web = wp_cam[:, :, :, wall_edge_idx[:, 1], :]
        wza, wzb = wea[..., 2], web[..., 2]
        w_crosses = (wza > NEAR_Z) != (wzb > NEAR_Z)
        if w_crosses.any():
            wdz_e = wzb - wza
            w_safe_dz = torch.where(wdz_e.abs() > 1e-8, wdz_e, torch.ones_like(wdz_e))
            wt_clip = ((NEAR_Z - wza) / w_safe_dz).clamp(0.0, 1.0)
            w_clip_pt = wea + wt_clip.unsqueeze(-1) * (web - wea)
            w_clip_z = w_clip_pt[..., 2].clamp(min=NEAR_Z)
            w_clip_u = fx[:, None, None, None] * (w_clip_pt[..., 0] / w_clip_z) \
                       + cx[:, None, None, None]
            w_clip_v = fy[:, None, None, None] * (w_clip_pt[..., 1] / w_clip_z) \
                       + cy[:, None, None, None]
            # Clamp clip projections to [0, 1] to prevent bbox explosion
            w_clip_u = w_clip_u.clamp(0.0, 1.0)
            w_clip_v = w_clip_v.clamp(0.0, 1.0)
            wu_min = torch.min(wu_min, torch.where(w_crosses, w_clip_u, INF).min(dim=-1).values)
            wu_max = torch.max(wu_max, torch.where(w_crosses, w_clip_u, -INF).max(dim=-1).values)
            wv_min = torch.min(wv_min, torch.where(w_crosses, w_clip_v, INF).min(dim=-1).values)
            wv_max = torch.max(wv_max, torch.where(w_crosses, w_clip_v, -INF).max(dim=-1).values)

        wz_sum = (wz_safe * w_vis.float()).sum(dim=-1)
        wz_cnt = w_vis.float().sum(dim=-1).clamp(min=1.0)
        w_mean_depth = wz_sum / wz_cnt

        # Wall depth gradients
        wvis_f = w_vis.float()                               # (B, T, M, 4)
        wu_sum = (wu * wvis_f).sum(dim=-1)
        wv_sum = (wv * wvis_f).sum(dim=-1)
        w_u_mean = wu_sum / wz_cnt
        w_v_mean = wv_sum / wz_cnt

        wdu = wu - w_u_mean.unsqueeze(-1)
        wdv = wv - w_v_mean.unsqueeze(-1)
        wdz = wz_safe - w_mean_depth.unsqueeze(-1)
        wWuu = (wvis_f * wdu * wdu).sum(dim=-1)
        wWvv = (wvis_f * wdv * wdv).sum(dim=-1)
        wWuv = (wvis_f * wdu * wdv).sum(dim=-1)
        wWuz = (wvis_f * wdu * wdz).sum(dim=-1)
        wWvz = (wvis_f * wdv * wdz).sum(dim=-1)
        wdet = (wWuu * wWvv - wWuv * wWuv).clamp(min=1e-10)
        w_grad_u = (wWvv * wWuz - wWuv * wWvz) / wdet
        w_grad_v = (wWuu * wWvz - wWuv * wWuz) / wdet

        w_any_vis = w_vis.any(dim=-1)                        # (B, T, M)

        # Overwrite wall entries (only for wall nodes)
        is_wall = sg_node_is_wall.unsqueeze(1).expand(-1, T, -1)  # (B, T, M)
        u_min = torch.where(is_wall, wu_min, u_min)
        u_max = torch.where(is_wall, wu_max, u_max)
        v_min = torch.where(is_wall, wv_min, v_min)
        v_max = torch.where(is_wall, wv_max, v_max)
        mean_depth = torch.where(is_wall, w_mean_depth, mean_depth)
        u_mean = torch.where(is_wall, w_u_mean, u_mean)
        v_mean = torch.where(is_wall, w_v_mean, v_mean)
        grad_u = torch.where(is_wall, w_grad_u, grad_u)
        grad_v = torch.where(is_wall, w_grad_v, grad_v)
        any_visible = torch.where(is_wall, w_any_vis, any_visible)

    # -- Clamp bboxes to [0, 1] and invalidate fully-behind-camera nodes --
    fully_behind = ~any_visible | ~sg_node_mask.unsqueeze(1).expand(-1, T, -1)
    u_min = u_min.clamp(0, 1)
    u_max = u_max.clamp(0, 1)
    v_min = v_min.clamp(0, 1)
    v_max = v_max.clamp(0, 1)

    # Scale normalised coords to pixel coords
    u_min_px = (u_min * output_w).long().clamp(0, output_w - 1)
    u_max_px = (u_max * output_w).long().clamp(0, output_w - 1)
    v_min_px = (v_min * output_h).long().clamp(0, output_h - 1)
    v_max_px = (v_max * output_h).long().clamp(0, output_h - 1)

    # Flatten (B, T) → BT
    BT = B * T
    u_min_px = u_min_px.reshape(BT, M)
    u_max_px = u_max_px.reshape(BT, M)
    v_min_px = v_min_px.reshape(BT, M)
    v_max_px = v_max_px.reshape(BT, M)
    mean_depth_flat = mean_depth.reshape(BT, M)
    u_mean_flat = u_mean.reshape(BT, M)
    v_mean_flat = v_mean.reshape(BT, M)
    grad_u_flat = grad_u.reshape(BT, M)
    grad_v_flat = grad_v.reshape(BT, M)
    fully_behind_flat = fully_behind.reshape(BT, M)
    node_types_exp = sg_node_types.unsqueeze(1).expand(-1, T, -1).reshape(BT, M)

    # ===== Z-buffer rasterization =====
    # Build pixel grid
    py = torch.arange(output_h, device=device)               # (H,)
    px = torch.arange(output_w, device=device)                # (W,)
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")   # (H, W)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)                 # (1, 1, H, W)
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)                 # (1, 1, H, W)

    # Per-node coverage: pixel inside 2D bbox → (BT, M, H, W)
    u_min_2d = u_min_px.unsqueeze(-1).unsqueeze(-1)           # (BT, M, 1, 1)
    u_max_2d = u_max_px.unsqueeze(-1).unsqueeze(-1)
    v_min_2d = v_min_px.unsqueeze(-1).unsqueeze(-1)
    v_max_2d = v_max_px.unsqueeze(-1).unsqueeze(-1)

    inside = (grid_x >= u_min_2d) & (grid_x <= u_max_2d) \
           & (grid_y >= v_min_2d) & (grid_y <= v_max_2d)     # (BT, M, H, W)

    # Invalidate behind-camera / padded nodes
    inside = inside & ~fully_behind_flat.unsqueeze(-1).unsqueeze(-1)

    # Exclude door nodes from z-buffer so objects behind the opening are visible
    if sg_node_is_door is not None:
        is_door_flat = sg_node_is_door.unsqueeze(1).expand(-1, T, -1).reshape(BT, M)
        inside = inside & ~is_door_flat.unsqueeze(-1).unsqueeze(-1)

    # Per-pixel depth for each node via linear depth gradient in image space.
    # z(u,v) ≈ z_mean + dz/du·(u − u_mean) + dz/dv·(v − v_mean)
    # Normalised pixel centres:
    grid_u_norm = (grid_x.float() + 0.5) / output_w          # (1, 1, H, W)
    grid_v_norm = (grid_y.float() + 0.5) / output_h

    pixel_depth = (
        mean_depth_flat.unsqueeze(-1).unsqueeze(-1)
        + grad_u_flat.unsqueeze(-1).unsqueeze(-1)
          * (grid_u_norm - u_mean_flat.unsqueeze(-1).unsqueeze(-1))
        + grad_v_flat.unsqueeze(-1).unsqueeze(-1)
          * (grid_v_norm - v_mean_flat.unsqueeze(-1).unsqueeze(-1))
    )                                                         # (BT, M, H, W)
    pixel_depth = pixel_depth.clamp(min=1e-4)                 # avoid negative from extrapolation

    depth_per_node = torch.where(
        inside, pixel_depth, INF.expand_as(inside),
    )                                                         # (BT, M, H, W)

    # Z-buffer: argmin over nodes
    winner_idx = depth_per_node.argmin(dim=1)                 # (BT, H, W)
    winner_depth = depth_per_node.gather(
        1, winner_idx.unsqueeze(1)
    ).squeeze(1)                                              # (BT, H, W)

    # Gather class from winner
    layout_cls = node_types_exp.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, output_h, output_w)
    layout_cls = layout_cls.gather(1, winner_idx.unsqueeze(1)).squeeze(1)  # (BT, H, W)

    # Background: where no node covers the pixel (depth still INF)
    is_bg = winner_depth >= 1e6
    layout_cls = layout_cls.masked_fill(is_bg, 0)
    layout_depth = winner_depth.masked_fill(is_bg, 0.0)

    return layout_cls.long(), layout_depth


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ProcTHOR scene graph vocabulary and embeddings")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to POC dataset root")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for vocab + embeddings")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP/SigLIP model name from HuggingFace")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Scan vocabulary
    print("Scanning vocabulary...")
    type_to_id, id_to_type = scan_vocabulary(args.dataset_root)
    print(f"Found {len(id_to_type) - 1} object types (+ 1 padding)")

    # Step 2: Save vocabulary
    save_vocabulary(type_to_id, id_to_type, args.output_dir)

    # Step 3: Pre-compute CLIP embeddings
    print(f"Pre-computing embeddings with {args.clip_model}...")
    embeddings_path = os.path.join(args.output_dir, "sg_type_embeddings.pt")
    precompute_clip_embeddings(
        id_to_type,
        embeddings_path,
        model_name=args.clip_model,
        device=args.device,
    )
    print("Done!")
