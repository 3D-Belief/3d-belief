"""
Utilities for ProcTHOR scene graph processing:
- Build vocabulary of object types and edge types
- Pre-compute CLIP/SigLIP text embeddings for object types
- Scene graph parsing helpers
"""
import json
import os
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


def scan_vocabulary(dataset_root: str) -> Tuple[Dict[str, int], List[str]]:
    """
    Scan all episodes to build a vocabulary of object types.

    Returns:
        type_to_id: mapping from objectType string to integer ID (0 reserved for padding)
        id_to_type: list where id_to_type[i] is the objectType string for ID i
    """
    all_types = set()
    dataset_root = Path(dataset_root)
    for ep_dir in sorted(dataset_root.iterdir()):
        sg_path = ep_dir / "all_scene_graphs.json"
        if not sg_path.exists():
            continue
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
    type_to_id = {t: i for i, t in enumerate(id_to_type)}
    return type_to_id, id_to_type


def precompute_clip_embeddings(
    id_to_type: List[str],
    output_path: str,
    model_name: str = "openai/clip-vit-base-patch32",
    prompt_template: str = "a {obj_type} in a room",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Pre-compute text embeddings for all object types using CLIP or SigLIP.

    Args:
        id_to_type: list of object type strings (index 0 = "<pad>")
        output_path: where to save the embeddings tensor
        model_name: HuggingFace model identifier
        prompt_template: template with {obj_type} placeholder
        device: device to run inference on

    Returns:
        embeddings: (n_types, D_clip) float32 tensor
    """
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Build text prompts
    prompts = []
    for type_name in id_to_type:
        if type_name == "<pad>":
            prompts.append("")
        else:
            # Convert camelCase to readable: "FloorLamp" -> "Floor Lamp"
            readable = _camel_to_words(type_name)
            prompts.append(prompt_template.format(obj_type=readable))

    # Encode all prompts
    with torch.no_grad():
        inputs = tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        if hasattr(model, "get_text_features"):
            # CLIP-style model
            embeddings = model.get_text_features(**inputs)
        else:
            # Fallback: use last hidden state pooled
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

    # Zero out the padding embedding
    embeddings[0] = 0.0

    embeddings = embeddings.cpu().float()
    torch.save(embeddings, output_path)
    print(f"Saved {embeddings.shape} embeddings to {output_path}")
    return embeddings


def _camel_to_words(name: str) -> str:
    """Convert camelCase/PascalCase to space-separated words.
    E.g. 'FloorLamp' -> 'floor lamp', 'ObjaCarton' -> 'obja carton'
    """
    import re
    words = re.sub(r"([A-Z])", r" \1", name).strip().lower()
    return words


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
) -> dict:
    """
    Parse a single frame's scene graph into tensors.

    Args:
        frame_data: one element from all_scene_graphs.json list
        type_to_id: vocabulary mapping
        near_threshold: distance threshold for "near" edges (meters)
        max_nodes: maximum number of nodes (padded)
        max_edges: maximum number of edges (padded)

    Returns:
        dict with sg_node_types, sg_node_positions, sg_node_rotations,
        sg_node_sizes, sg_edge_index, sg_edge_types, sg_node_mask, sg_edge_mask
    """
    nodes = []  # list of (type_id, pos, rot, size)
    room_assignments = []  # room index for each node

    for room_idx, room in enumerate(frame_data["scene_graph"]):
        for child in room.get("children", []):
            _add_node(child, type_to_id, nodes, room_assignments, room_idx)
            # Also add nested children (objects on top of furniture)
            for sub in child.get("children", []):
                _add_node(sub, type_to_id, nodes, room_assignments, room_idx)

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

    for i in range(n_nodes):
        type_id, pos, rot, size = nodes[i]
        node_types[i] = type_id
        node_positions[i] = torch.tensor(pos, dtype=torch.float32)
        node_rotations[i] = torch.tensor(rot, dtype=torch.float32)
        node_sizes[i] = torch.tensor(size, dtype=torch.float32)
        node_mask[i] = True

    edge_index = torch.zeros(2, max_edges, dtype=torch.long)
    edge_types = torch.zeros(max_edges, dtype=torch.long)
    edge_mask = torch.zeros(max_edges, dtype=torch.bool)

    for i in range(n_edges):
        src, dst, etype = edges[i]
        edge_index[0, i] = src
        edge_index[1, i] = dst
        edge_types[i] = etype
        edge_mask[i] = True

    return {
        "sg_node_types": node_types,
        "sg_node_positions": node_positions,
        "sg_node_rotations": node_rotations,
        "sg_node_sizes": node_sizes,
        "sg_edge_index": edge_index,
        "sg_edge_types": edge_types,
        "sg_node_mask": node_mask,
        "sg_edge_mask": edge_mask,
    }


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
    pos = [pos["x"], pos["y"], pos["z"]]

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

    nodes.append((type_id, pos, rot, size))
    room_assignments.append(room_idx)


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
