"""
Pre-compute CameraHead poses for the entire ProcTHOR dataset.

Runs VGGT aggregator + finetuned CameraHead on all frames of each scene,
and saves predicted poses as `predicted_poses.npz` alongside the existing
`all_poses.npz` in each scene directory.

Output format matches `all_poses.npz`:
  key "poses": [N, 4, 4] c2w matrices in 3d-belief convention
  key "intrinsics": [N, 3, 3] normalized intrinsics

Usage:
    python splat_belief/experiment/precompute_poses.py \
        --data_root ../datasets/poc_dataset \
        --camera_head_ckpt checkpoints/camera_head_procthor/camera_head_best.pth \
        --stage train \
        --batch_frames 16
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from splat_belief.splat.alignment.vggt.models.vggt import VGGT
from splat_belief.splat.pose_estimator.coordinate_utils import (
    camera_head_to_3dbelief,
    normalize_poses_to_first_frame,
)


def load_video_frames(video_path: str, image_size: int = 518) -> torch.Tensor:
    """Load all frames from video, resize to image_size x image_size.

    Returns:
        frames: [N, 3, H, W] tensor in [0, 1].
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame.astype(np.float32)).permute(2, 0, 1) / 255.0
        frame = F.interpolate(frame.unsqueeze(0), size=(image_size, image_size),
                              mode="bilinear", align_corners=False)[0]
        frames.append(frame)
    cap.release()
    return torch.stack(frames) if frames else torch.empty(0, 3, image_size, image_size)


@torch.no_grad()
def predict_scene_poses(
    vggt: torch.nn.Module,
    frames: torch.Tensor,
    batch_frames: int = 16,
    num_iterations: int = 4,
    device: torch.device = torch.device("cuda"),
) -> tuple:
    """Predict poses for all frames in a scene using sliding-window batching.

    Uses the first frame as the reference for normalization.
    All frames are processed together if possible, otherwise in overlapping
    windows that share the first frame for consistent normalization.

    Args:
        vggt: VGGT model with camera_head.
        frames: [N, 3, 518, 518] all frames for this scene.
        batch_frames: Max frames per forward pass.
        num_iterations: CameraHead refinement iterations.
        device: CUDA device.

    Returns:
        c2w_all: [N, 4, 4] numpy array of c2w in 3d-belief convention.
        intrinsics_all: [N, 3, 3] numpy array of normalized intrinsics.
    """
    N = frames.shape[0]

    if N <= batch_frames:
        # Process all at once
        images = frames.unsqueeze(0).to(device)  # [1, N, 3, H, W]
        agg_tokens, _ = vggt.aggregator(images)
        with torch.amp.autocast("cuda", enabled=False):
            pose_enc_list = vggt.camera_head(agg_tokens, num_iterations=num_iterations)
        pose_enc = pose_enc_list[-1]  # [1, N, 9]
        c2w, intrinsics = camera_head_to_3dbelief(pose_enc, image_hw=(518, 518))
        c2w = normalize_poses_to_first_frame(c2w)
        return c2w[0].cpu().numpy(), intrinsics[0].cpu().numpy()

    # For longer sequences, process in overlapping windows.
    # Always include frame 0 as anchor for consistent global coordinate frame.
    c2w_all = np.zeros((N, 4, 4), dtype=np.float32)
    intrinsics_all = np.zeros((N, 3, 3), dtype=np.float32)
    processed = np.zeros(N, dtype=bool)

    # First pass: frame 0 alone to get reference
    images_ref = frames[0:1].unsqueeze(0).to(device)
    agg_tokens_ref, _ = vggt.aggregator(images_ref)
    with torch.amp.autocast("cuda", enabled=False):
        pose_enc_ref = vggt.camera_head(agg_tokens_ref, num_iterations=num_iterations)[-1]
    c2w_ref, intr_ref = camera_head_to_3dbelief(pose_enc_ref, image_hw=(518, 518))
    # Reference frame is identity after normalization
    inv_ref = torch.linalg.inv(c2w_ref[:, 0:1])  # [1, 1, 4, 4]

    # Process in windows: always include frame 0 + window frames
    window_size = batch_frames - 1  # reserve 1 slot for anchor frame 0
    for start in range(0, N, window_size):
        end = min(start + window_size, N)
        window_ids = list(range(start, end))

        # Include frame 0 as the first frame if not already in window
        if 0 not in window_ids:
            batch_ids = [0] + window_ids
        else:
            batch_ids = window_ids

        batch_frames_tensor = frames[batch_ids].unsqueeze(0).to(device)
        agg_tokens, _ = vggt.aggregator(batch_frames_tensor)
        with torch.amp.autocast("cuda", enabled=False):
            pose_enc = vggt.camera_head(agg_tokens, num_iterations=num_iterations)[-1]
        c2w, intrinsics = camera_head_to_3dbelief(pose_enc, image_hw=(518, 518))
        c2w = normalize_poses_to_first_frame(c2w)  # normalized to batch_ids[0]

        # If frame 0 was prepended, the poses are already in the right coordinate frame
        # (normalized to frame 0). Just extract the window frames.
        c2w_np = c2w[0].cpu().numpy()
        intr_np = intrinsics[0].cpu().numpy()

        for i, fid in enumerate(batch_ids):
            if not processed[fid]:
                c2w_all[fid] = c2w_np[i]
                intrinsics_all[fid] = intr_np[i]
                processed[fid] = True

    return c2w_all, intrinsics_all


def main():
    parser = argparse.ArgumentParser(description="Pre-compute CameraHead poses for ProcTHOR")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--camera_head_ckpt", type=str, default=None,
                        help="Path to finetuned CameraHead checkpoint (.pth). If omitted, uses pretrained VGGT CameraHead.")
    parser.add_argument("--stage", type=str, default="train",
                        choices=["train", "unit"])
    parser.add_argument("--batch_frames", type=int, default=16,
                        help="Max frames per VGGT forward pass")
    parser.add_argument("--num_iterations", type=int, default=4)
    parser.add_argument("--output_filename", type=str, default="predicted_poses.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading VGGT-1B...")
    vggt = VGGT.from_pretrained("facebook/VGGT-1B")
    vggt.eval()
    for p in vggt.parameters():
        p.requires_grad = False

    # Optionally load finetuned CameraHead weights
    if args.camera_head_ckpt is not None:
        print(f"Loading finetuned CameraHead from {args.camera_head_ckpt}...")
        ckpt = torch.load(args.camera_head_ckpt, map_location="cpu", weights_only=True)
        vggt.camera_head.load_state_dict(ckpt)
    else:
        print("Using pretrained VGGT CameraHead (no finetuned checkpoint)")
    vggt = vggt.to(device)

    # Discover scenes
    image_root = Path(args.data_root) / args.stage
    scene_paths = sorted([p for p in image_root.glob("*/") if p.is_dir()])
    print(f"Found {len(scene_paths)} scenes in {image_root}")

    for scene_path in tqdm(scene_paths, desc="Scenes"):
        rgb_file = scene_path / "rgb_trajectory.mp4"
        if not rgb_file.exists():
            continue

        output_file = scene_path / args.output_filename
        if output_file.exists():
            continue  # Skip already processed

        # Load frames
        frames = load_video_frames(str(rgb_file), image_size=518)
        if frames.shape[0] == 0:
            print(f"Skipping {scene_path.name}: no frames")
            continue

        # Predict poses
        c2w, intrinsics = predict_scene_poses(
            vggt, frames,
            batch_frames=args.batch_frames,
            num_iterations=args.num_iterations,
            device=device,
        )

        # Save
        np.savez_compressed(
            output_file,
            poses=c2w,
            intrinsics=intrinsics,
        )

    print("Done!")


if __name__ == "__main__":
    main()
