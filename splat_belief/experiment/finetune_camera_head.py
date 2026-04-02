"""
Finetune VGGT CameraHead on ProcTHOR dataset.

Loads the full VGGT-1B model (frozen aggregator + trainable CameraHead),
feeds ProcTHOR images through the aggregator, predicts poses via CameraHead,
and supervises with GT poses using geodesic rotation loss + L1 translation loss.

Usage:
    python splat_belief/experiment/finetune_camera_head.py \
        --data_root ../datasets/poc_dataset \
        --output_dir checkpoints/camera_head_procthor \
        --epochs 20 --lr 1e-4 --batch_size 4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from splat_belief.splat.alignment.vggt.models.vggt import VGGT
from splat_belief.splat.alignment.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
    extri_intri_to_pose_encoding,
)
from splat_belief.splat.pose_estimator.coordinate_utils import (
    camera_head_to_3dbelief,
    threeDbelief_to_camera_head,
    normalize_poses_to_first_frame,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProcTHORPoseDataset(Dataset):
    """Loads ProcTHOR image sequences + GT poses for CameraHead finetuning.

    Each sample is a group of S frames (context + target) from a single scene.
    Returns images preprocessed for VGGT (518x518) and GT poses in both
    3d-belief convention and CameraHead 9D encoding.
    """

    def __init__(
        self,
        root: str,
        stage: str = "train",
        num_views: int = 8,
        image_size: int = 518,
        max_scenes: int = None,
        max_frame_gap: int = 30,
    ):
        super().__init__()
        self.num_views = num_views
        self.image_size = image_size
        self.max_frame_gap = max_frame_gap
        self.rng = np.random.default_rng(42)

        image_root = Path(root) / stage
        scene_paths = sorted([p for p in image_root.glob("*/") if p.is_dir()])
        if max_scenes is not None:
            scene_paths = scene_paths[:max_scenes]

        self.scenes = []
        for sp in scene_paths:
            rgb_file = sp / "rgb_trajectory.mp4"
            pose_file = sp / "all_poses.npz"
            if rgb_file.exists() and pose_file.exists():
                n_frames = np.load(pose_file)["poses"].shape[0]
                if n_frames >= num_views:
                    self.scenes.append((sp, n_frames))

        print(f"[PoseDataset] {len(self.scenes)} scenes with >= {num_views} frames")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path, n_frames = self.scenes[idx]

        # Sample num_views frames within a contiguous window
        max_start = max(0, n_frames - self.max_frame_gap)
        start = self.rng.integers(0, max_start + 1)
        end = min(start + self.max_frame_gap, n_frames)
        frame_ids = sorted(self.rng.choice(range(start, end), size=self.num_views, replace=False))

        # Load images
        cap = cv2.VideoCapture(str(scene_path / "rgb_trajectory.mp4"))
        images = []
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return self[self.rng.integers(0, len(self.scenes))]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.tensor(frame.astype(np.float32)).permute(2, 0, 1) / 255.0
            frame = F.interpolate(frame.unsqueeze(0), size=(self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False)[0]
            images.append(frame)
        cap.release()
        images = torch.stack(images)  # [S, 3, 518, 518]

        # Load GT poses (c2w in 3d-belief convention)
        poses_data = np.load(scene_path / "all_poses.npz")
        all_poses = poses_data["poses"]  # [N, 4, 4] c2w

        gt_c2w_list = []
        conversion = np.diag([1, -1, -1, 1]).astype(np.float32)
        for fid in frame_ids:
            c2w = all_poses[fid].astype(np.float32)
            # Convert to 3d-belief convention: c2w in scene file -> w2c -> apply flip -> Camera.c2w_mat
            w2c = np.linalg.inv(c2w)
            extrinsics = conversion @ w2c
            c2w_3dbelief = np.linalg.inv(extrinsics)
            gt_c2w_list.append(c2w_3dbelief)

        gt_c2w = torch.tensor(np.stack(gt_c2w_list)).float()  # [S, 4, 4]

        # Normalize relative to first frame
        inv_first = torch.linalg.inv(gt_c2w[0:1])  # [1, 4, 4]
        gt_c2w_norm = inv_first @ gt_c2w  # [S, 4, 4]

        # Fixed intrinsics from ProcTHOR Camera class
        intrinsics_norm = torch.tensor([
            [0.390, 0, 0.5],
            [0, 0.385, 0.5],
            [0, 0, 1],
        ], dtype=torch.float32).unsqueeze(0).expand(self.num_views, -1, -1)  # [S, 3, 3]

        return {
            "images": images,         # [S, 3, 518, 518]
            "gt_c2w": gt_c2w_norm,     # [S, 4, 4]
            "intrinsics": intrinsics_norm,  # [S, 3, 3]
        }


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def geodesic_rotation_loss(R_pred, R_gt):
    """Geodesic distance between rotation matrices.

    Args:
        R_pred: [B, S, 3, 3] predicted rotations.
        R_gt: [B, S, 3, 3] ground truth rotations.

    Returns:
        Scalar mean geodesic angle (radians).
    """
    R_diff = R_pred.transpose(-1, -2) @ R_gt  # [B, S, 3, 3]
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    return angle.mean()


def translation_loss(t_pred, t_gt):
    """L1 loss on translation vectors.

    Args:
        t_pred: [B, S, 3] predicted translations.
        t_gt: [B, S, 3] ground truth translations.

    Returns:
        Scalar mean L1 loss.
    """
    return F.l1_loss(t_pred, t_gt)


def pose_loss(c2w_pred, c2w_gt, rot_weight=1.0, trans_weight=1.0):
    """Combined pose loss: geodesic rotation + L1 translation.

    Args:
        c2w_pred: [B, S, 4, 4] predicted c2w.
        c2w_gt: [B, S, 4, 4] ground truth c2w.

    Returns:
        Dictionary with total loss and components.
    """
    R_pred = c2w_pred[..., :3, :3]
    R_gt = c2w_gt[..., :3, :3]
    t_pred = c2w_pred[..., :3, 3]
    t_gt = c2w_gt[..., :3, 3]

    rot_loss = geodesic_rotation_loss(R_pred, R_gt)
    trans_loss = translation_loss(t_pred, t_gt)
    total = rot_weight * rot_loss + trans_weight * trans_loss

    return {
        "total": total,
        "rotation": rot_loss,
        "translation": trans_loss,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VGGT model
    print("Loading VGGT-1B...")
    vggt = VGGT.from_pretrained("facebook/VGGT-1B")
    vggt = vggt.to(device)

    # Freeze everything except camera_head
    for p in vggt.parameters():
        p.requires_grad = False
    for p in vggt.camera_head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in vggt.camera_head.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vggt.parameters())
    print(f"Trainable CameraHead params: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M total")

    # Dataset
    dataset = ProcTHORPoseDataset(
        root=args.data_root,
        stage="train",
        num_views=args.num_views,
        max_scenes=args.max_scenes,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        vggt.camera_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader)
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        vggt.camera_head.train()
        epoch_losses = {"total": 0, "rotation": 0, "translation": 0}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            images = batch["images"].to(device)       # [B, S, 3, 518, 518]
            gt_c2w = batch["gt_c2w"].to(device)        # [B, S, 4, 4]

            # Run aggregator (frozen)
            with torch.no_grad():
                aggregated_tokens_list, patch_start_idx = vggt.aggregator(images)

            # Run CameraHead (trainable)
            with torch.amp.autocast("cuda", enabled=False):
                pose_enc_list = vggt.camera_head(
                    aggregated_tokens_list,
                    num_iterations=args.num_refinement_iterations,
                )

            # Compute loss on all iterations (with decreasing weight for earlier ones)
            total_loss = torch.tensor(0.0, device=device)
            for iter_idx, pose_enc in enumerate(pose_enc_list):
                # Convert predicted pose encoding to c2w
                c2w_pred, intrinsics_pred = camera_head_to_3dbelief(
                    pose_enc, image_hw=(518, 518)
                )
                # Normalize to first frame
                c2w_pred_norm = normalize_poses_to_first_frame(c2w_pred)

                iter_weight = 0.5 ** (len(pose_enc_list) - 1 - iter_idx)
                losses = pose_loss(
                    c2w_pred_norm, gt_c2w,
                    rot_weight=args.rot_weight,
                    trans_weight=args.trans_weight,
                )
                total_loss = total_loss + iter_weight * losses["total"]

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vggt.camera_head.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Log the last iteration losses
            epoch_losses["total"] += losses["total"].item()
            epoch_losses["rotation"] += losses["rotation"].item()
            epoch_losses["translation"] += losses["translation"].item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "rot": f"{np.degrees(losses['rotation'].item()):.2f}°",
                "trans": f"{losses['translation'].item():.4f}",
            })

        # Epoch summary
        avg = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        print(f"Epoch {epoch + 1}: loss={avg['total']:.4f}, "
              f"rot={np.degrees(avg['rotation']):.2f}°, "
              f"trans={avg['translation']:.4f}")

        # Save checkpoint
        if avg["total"] < best_loss:
            best_loss = avg["total"]
            save_path = output_dir / "camera_head_best.pth"
            torch.save(vggt.camera_head.state_dict(), save_path)
            print(f"Saved best CameraHead to {save_path}")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = output_dir / f"camera_head_epoch{epoch + 1}.pth"
            torch.save(vggt.camera_head.state_dict(), save_path)

    # Save final
    save_path = output_dir / "camera_head_final.pth"
    torch.save(vggt.camera_head.state_dict(), save_path)
    print(f"Saved final CameraHead to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Finetune VGGT CameraHead on ProcTHOR")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to ProcTHOR dataset root")
    parser.add_argument("--output_dir", type=str, default="checkpoints/camera_head_procthor",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_views", type=int, default=8,
                        help="Number of frames per training sample")
    parser.add_argument("--num_refinement_iterations", type=int, default=4)
    parser.add_argument("--rot_weight", type=float, default=1.0)
    parser.add_argument("--trans_weight", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
