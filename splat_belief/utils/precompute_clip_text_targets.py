"""Precompute SPOC class names and CLIP-text embeddings."""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = str(REPO_ROOT / "data" / "all_rerendered_root" / "train")
DEFAULT_OUT = str(REPO_ROOT / "outputs" / "cache" / "clip_text_table.pt")


def camel_to_words(name):
    if name.startswith("Obja"):
        name = name[4:]
    return re.sub(r"([A-Z])", r" \1", name).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out_path", default=DEFAULT_OUT)
    ap.add_argument("--max_scenes", type=int, default=2000,
                    help="how many scenes to scan to estimate the vocabulary")
    ap.add_argument("--min_scenes_per_class", type=int, default=20,
                    help="drop classes seen in fewer than this many scenes")
    ap.add_argument("--prompt_template", default="{name}",
                    help="bare object name by default; CLIP-text is most"
                         " discriminative without 'a photo of a' boilerplate")
    ap.add_argument("--model_name", default="ViT-B-16")
    ap.add_argument("--pretrained", default="laion2b_s34b_b88k")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    Path(os.path.dirname(args.out_path)).mkdir(parents=True, exist_ok=True)

    scenes = sorted(os.listdir(args.dataset_root))
    print(f"train split has {len(scenes)} scenes; scanning {min(args.max_scenes, len(scenes))}")
    name_counter = Counter()
    scene_with_name = Counter()
    obja_prefix_count = 0
    for sc in scenes[: args.max_scenes]:
        meta_path = os.path.join(args.dataset_root, sc, "all_semantic_meta.json")
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            continue
        seen = set()
        for fr in meta:
            for inst_id, info in fr.get("objects", {}).items():
                name = info.get("name", "Unknown")
                if name.startswith("Obja"):
                    obja_prefix_count += 1
                name_counter[name] += 1
                seen.add(name)
        for n in seen:
            scene_with_name[n] += 1
    print(f"saw {len(name_counter)} unique names total"
          f" (Obja-prefixed instances: {obja_prefix_count})")

    keep = []
    for n in sorted(scene_with_name):
        if n == "Unknown":
            continue
        if scene_with_name[n] >= args.min_scenes_per_class:
            keep.append(n)
    keep.sort()
    print(f"keeping {len(keep)} classes (>= {args.min_scenes_per_class} scenes coverage)")

    import open_clip
    print(f"loading CLIP text encoder: {args.model_name} / {args.pretrained}")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, precision="fp16",
    )
    clip_model = clip_model.to(args.device).eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)

    id_to_name = ["<ignore>"] + keep
    name_to_id = {n: i for i, n in enumerate(id_to_name)}

    prompts = [""] * len(id_to_name)
    for n in keep:
        prompts[name_to_id[n]] = args.prompt_template.format(name=camel_to_words(n))

    tokens = tokenizer(prompts).to(args.device)
    with torch.no_grad():
        out = clip_model.encode_text(tokens)
        out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    embeddings = out.cpu().float()
    embeddings[0].zero_()  # explicit zero for the ignore row

    n_scenes = torch.tensor(
        [0] + [scene_with_name.get(n, 0) for n in keep], dtype=torch.int64
    )

    payload = {
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "embeddings": embeddings,
        "n_scenes_per_class": n_scenes,
        "prompt_template": args.prompt_template,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
    }
    torch.save(payload, args.out_path)
    print(f"saved {args.out_path}")
    print(f"  classes: {len(id_to_name)} (incl. ignore)")
    print(f"  embeddings: {tuple(embeddings.shape)}")

    cls_emb = embeddings[1:]  # drop ignore row
    sims = cls_emb @ cls_emb.T
    n = sims.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    mean_off = sims[mask].mean().item()
    max_off = sims[mask].max().item()
    print(f"\nClass-text separability (cos sim across {n} classes):")
    print(f"  mean off-diagonal = {mean_off:.3f}")
    print(f"  max  off-diagonal = {max_off:.3f}")

    print("\nTop-15 most-covered classes:")
    pairs = sorted(((scene_with_name[n], n) for n in keep), reverse=True)[:15]
    for s, n in pairs:
        print(f"  {n:<25}  scenes={s}")


if __name__ == "__main__":
    main()
