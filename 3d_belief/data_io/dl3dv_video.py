from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union, Literal, Dict, Any
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2

from utils import normalize_to_neg_one_to_one, rotation_angle
from layers import T5Encoder

Stage = Literal["train", "test", "unit"]


class CameraDL3DV(object):
    """Wrap DL3DV per-frame camera (intrinsic + pose 4x4)."""
    def __init__(self, intrinsic: np.ndarray, pose_4x4: np.ndarray, pose_is_c2w: bool):
        assert intrinsic.shape == (3, 3)
        assert pose_4x4.shape == (4, 4)
        self.intrinsics = intrinsic.astype(np.float32)
        if pose_is_c2w:
            self.c2w_mat = pose_4x4.astype(np.float32)
            self.w2c_mat = np.linalg.inv(self.c2w_mat).astype(np.float32)
        else:
            self.w2c_mat = pose_4x4.astype(np.float32)
            self.c2w_mat = np.linalg.inv(self.w2c_mat).astype(np.float32)


def _normalize_K(K: np.ndarray, H: int, W: int) -> np.ndarray:
    """Normalize intrinsics to [0,1] image coords: u' = u/W, v' = v/H."""
    Kn = K.copy()
    Kn[0, 0] /= W
    Kn[1, 1] /= H
    Kn[0, 2] /= W
    Kn[1, 2] /= H
    return Kn


def _center_crop_np(x: np.ndarray, size: int) -> np.ndarray:
    """Center-crop HW or HWC numpy array to (size,size)."""
    H, W = x.shape[:2]
    start_x = (W - size) // 2
    start_y = (H - size) // 2
    return x[start_y:start_y + size, start_x:start_x + size]


def _read_video_frame_rgb(video_file: Union[str, Path], frame_id_0based: int) -> np.ndarray:
    """
    Read one RGB frame from mp4 using cv2.VideoCapture.
    Returns HxWx3 uint8 (RGB).
    """
    def _try_read_seek(fid: int) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Try direct seek (twice, reopening each time)
    fid = int(frame_id_0based)
    frame = _try_read_seek(fid)
    if frame is not None:
        return frame
    frame = _try_read_seek(fid)
    if frame is not None:
        return frame
    # Slow but robust fallback: sequential grab up to fid
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Failed to open video: {video_file}")
    # Clamp if frame count is known (may be unreliable but prevents obvious OOB)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n_frames > 0:
        fid = max(0, min(fid, n_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(fid):
        ok = cap.grab()
        if not ok:
            cap.release()
            raise ValueError(f"Failed to grab up to frame {frame_id_0based} from video: {video_file}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise ValueError(f"Failed to read frame {frame_id_0based} from video: {video_file}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

class DL3DVDataset(Dataset):
    """
    Compressed DL3DV dataset:
      <root>/<split>/
        cam_merged/<group>/<episode>.npz
        videos/<group>/<episode>.mp4
    """

    z_near: float = 0.1
    z_far: float = 60.0

    def __init__(
        self,
        root: Union[str, Path],
        num_context: int,
        num_target: int,
        context_min_distance: int,
        context_max_distance: int,
        stage: Stage = "train",
        intermediate: bool = False,
        num_intermediate: Optional[int] = 3,
        language_encoder: Optional[T5Encoder] = None,
        overfit_to_index: Optional[int] = None,
        max_scenes: Optional[int] = None,
        image_size: int = 64,
        adjacent_angle: float = np.pi / 4,
        use_depth_supervision: bool = False,
        pose_is_c2w: bool = True,
        global_seed: int = 42,
        npz_cache_size: int = 8,
    ) -> None:
        super().__init__()
        assert num_context == 1, "This implementation currently supports num_context==1."
        assert num_target >= 1

        self.root = Path(root)
        self.stage = stage
        self.num_context = num_context
        self.num_target = num_target
        self.context_min_distance = context_min_distance
        self.context_max_distance = context_max_distance
        self.adjacent_angle = adjacent_angle
        self.image_size = int(image_size)
        self.intermediate = intermediate
        self.num_intermediate = num_intermediate
        self.pose_is_c2w = pose_is_c2w
        self.overfit_to_index = overfit_to_index
        self.normalize = normalize_to_neg_one_to_one

        self.rng = np.random.default_rng(global_seed)
        self.global_seed = global_seed

        if language_encoder is not None:
            self.lang_identity = language_encoder("").detach()
        else:
            self.lang_identity = None

        split_dir = self.root / {"train": "train", "test": "test", "unit": "unit"}[stage]
        self.videos_root = split_dir / "videos"
        self.cams_root = split_dir / "cam_merged"

        # Build episode list by matching videos/<group>/<ep>.mp4 with cam_merged/<group>/<ep>.npz
        episodes: List[Dict[str, Any]] = []
        skipped_mismatch = 0
        skipped_bad_video = 0
        skipped_bad_npz = 0
        if self.videos_root.exists():
            for mp4 in sorted(self.videos_root.glob("*/*.mp4")):
                group = mp4.parent.name
                ep = mp4.stem
                npz = self.cams_root / group / f"{ep}.npz"
                if npz.exists():
                    # video frame count
                    cap = cv2.VideoCapture(str(mp4))
                    if not cap.isOpened():
                        cap.release()
                        skipped_bad_video += 1
                        continue
                    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    cap.release()
                    if n_video <= 0:
                        skipped_bad_video += 1
                        continue

                    # npz pose + frame_files count
                    try:
                        data = np.load(npz, allow_pickle=True, mmap_mode="r")
                        n_pose = int(data["pose"].shape[0])
                        n_files = int(len(data["frame_files"]))
                    except Exception:
                        skipped_bad_npz += 1
                        continue

                    if not (n_video == n_pose == n_files):
                        skipped_mismatch += 1
                        continue

                    episodes.append({
                        "group": group,
                        "episode": ep,
                        "video": mp4,
                        "cam": npz,
                        "num_frames": min(n_video, n_pose, n_files),
                    })

        if max_scenes is not None:
            episodes = episodes[:max_scenes]

        self.episodes = episodes
        print(f"[DL3DVVideo] Episodes {len(self.episodes)} (split={stage})")
        print(f"[DL3DVVideo] Skipped mismatch={skipped_mismatch}, bad_video={skipped_bad_video}, bad_npz={skipped_bad_npz}")

        # Tiny in-process cache for npz contents (per worker process)
        self._npz_cache: Dict[str, Dict[str, Any]] = {}
        self._npz_cache_order: List[str] = []
        self._npz_cache_size = int(npz_cache_size)

    def __len__(self) -> int:
        return len(self.episodes)

    # ---------- NPZ loading ----------
    def _load_episode_npz(self, npz_path: Path) -> Dict[str, Any]:
        key = str(npz_path)
        if key in self._npz_cache:
            return self._npz_cache[key]

        data = np.load(npz_path, allow_pickle=True)
        frame_files = data["frame_files"]
        intr = data["intrinsic"]
        pose = data["pose"]

        # Normalize/standardize types
        # frame_files can be np.ndarray(dtype=object) or list-like
        frame_files_list = [str(x) for x in frame_files.tolist()] if hasattr(frame_files, "tolist") else [str(x) for x in frame_files]
        intr = np.asarray(intr, dtype=np.float32)   # (N,3,3)
        pose = np.asarray(pose, dtype=np.float32)   # (N,4,4)

        frame_ids = np.arange(len(frame_files_list), dtype=np.int64)

        packed = {"frame_files": frame_files_list, "frame_ids": frame_ids, "intrinsic": intr, "pose": pose}

        # cache insert with FIFO eviction
        self._npz_cache[key] = packed
        self._npz_cache_order.append(key)
        if len(self._npz_cache_order) > self._npz_cache_size:
            old = self._npz_cache_order.pop(0)
            self._npz_cache.pop(old, None)

        return packed

    # ---------- Frame I/O ----------
    def _read_frame(self, video_path: Path, cam_data: Dict[str, Any], idx_in_npz: int) -> Tuple[torch.Tensor, CameraDL3DV]:
        """
        idx_in_npz indexes into cam_data['intrinsic'][idx] and cam_data['pose'][idx].
        RGB frame is read from video using cam_data['frame_ids'][idx] as the video frame index.
        """
        frame_id = int(cam_data["frame_ids"][idx_in_npz])

        # Read RGB from mp4
        frame_u8 = _read_video_frame_rgb(video_path, frame_id)  # HxWx3 uint8 RGB
        height, width = frame_u8.shape[:2]

        # Center crop to square
        crop_size = min(height, width)
        frame_u8 = _center_crop_np(frame_u8, crop_size)

        rgb = (frame_u8.astype(np.float32) / 255.0)  # HxWx3 [0,1]
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()  # CxHxW

        # Resize
        rgb_t = F.interpolate(
            rgb_t.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            antialias=True,
        )[0]

        # Camera
        K = cam_data["intrinsic"][idx_in_npz].astype(np.float32)  # (3,3)
        K = _normalize_K(K, H=height, W=width)

        pose = cam_data["pose"][idx_in_npz].astype(np.float32)    # (4,4)
        cam = CameraDL3DV(K, pose, pose_is_c2w=self.pose_is_c2w)

        return rgb_t, cam

    # ---------- Sampling ----------
    def __getitem__(self, index: int):
        seed = (hash((int(index), self.global_seed)) % (2**32)) + (time.time_ns() % (2**32))
        self.rng = np.random.default_rng(seed)

        def fallback():
            seed_fb = (hash((int(index), self.global_seed+42)) % (2**32)) + (time.time_ns() % (2**32))
            self.rng = np.random.default_rng(seed_fb)
            return self[int(self.rng.integers(0, len(self.episodes)))]

        if len(self.episodes) == 0:
            raise RuntimeError(f"No episodes found under {self.root} split={self.stage}")

        try:
            ep_idx = int(self.rng.integers(0, len(self.episodes)))
            if self.overfit_to_index is not None:
                ep_idx = int(self.overfit_to_index) % len(self.episodes)

            ep = self.episodes[ep_idx]
            video_path: Path = ep["video"]
            cam_path: Path = ep["cam"]

            cam_data = self._load_episode_npz(cam_path)
            num_frames = int(ep.get("num_frames", cam_data["pose"].shape[0]))

            if num_frames < self.num_target + 1:
                return fallback()

            # Choose start frame constrained by context_max_distance
            max_start = max(num_frames - self.context_max_distance, 2)
            start_idx = int(self.rng.choice(max_start, 1)[0])
            if start_idx + 1 >= num_frames - 1:
                return fallback()

            # Get z-axis at start
            _, cam_start = self._read_frame(video_path, cam_data, start_idx)
            z_start = cam_start.c2w_mat[:3, 2].copy()

            # Find an end frame where angle exceeds threshold or distance exceeds max
            end_idx = min(start_idx + 1, num_frames - 1)
            for idx2 in range(start_idx + 1, num_frames):
                _, cam_i = self._read_frame(video_path, cam_data, idx2)
                z_i = cam_i.c2w_mat[:3, 2].copy()
                ang = rotation_angle(z_start, z_i)
                end_idx = idx2
                if ang > self.adjacent_angle or (idx2 - start_idx) > self.context_max_distance:
                    break

            if bool(self.rng.choice([True, False])):
                start_idx, end_idx = end_idx, start_idx

            ctxt_idx = [start_idx]
            trgt_idx = [end_idx]

            # Intermediates between min(ctxt,trgt) and max(ctxt,trgt)
            intm_idx = None
            if self.intermediate and self.num_intermediate and self.num_intermediate > 0:
                lo = min(ctxt_idx[0], trgt_idx[0])
                hi = max(ctxt_idx[0], trgt_idx[0])
                if hi > lo:
                    choices = np.arange(lo, hi + 1)
                    intm_idx = self.rng.choice(choices, self.num_intermediate, replace=True).tolist()

            # ----- Load target(s) -----
            trgt_rgbs, trgt_c2w, trgt_intr = [], [], []
            for i in trgt_idx:
                rgb_t, cam = self._read_frame(video_path, cam_data, int(i))
                trgt_rgbs.append(rgb_t)
                trgt_c2w.append(torch.from_numpy(cam.c2w_mat))
                trgt_intr.append(torch.from_numpy(cam.intrinsics))
            trgt_rgb = torch.stack(trgt_rgbs, dim=0)                # (T, 3, H, W)
            trgt_c2w = torch.stack(trgt_c2w, dim=0).float()         # (T, 4, 4)

            # ----- Load context(s) -----
            ctxt_rgbs, ctxt_c2w, ctxt_intr = [], [], []
            for i in ctxt_idx:
                rgb_t, cam = self._read_frame(video_path, cam_data, int(i))
                ctxt_rgbs.append(rgb_t)
                ctxt_c2w.append(torch.from_numpy(cam.c2w_mat))
                ctxt_intr.append(torch.from_numpy(cam.intrinsics))

            ctxt_rgb = torch.stack(ctxt_rgbs, dim=0)                # (C, 3, H, W)
            ctxt_c2w = torch.stack(ctxt_c2w, dim=0).float()         # (C, 4, 4)
            ctxt_intrinsics = torch.stack(ctxt_intr, dim=0).float() # (C, 3, 3)

            # ----- Load intermediates (optional) -----
            if self.intermediate and intm_idx is not None and len(intm_idx) > 0:
                intm_rgbs, intm_c2w, intm_intr = [], [], []
                for i in intm_idx:
                    rgb_t, cam = self._read_frame(video_path, cam_data, int(i))
                    intm_rgbs.append(rgb_t)
                    intm_c2w.append(torch.from_numpy(cam.c2w_mat))
                    intm_intr.append(torch.from_numpy(cam.intrinsics))

                intm_rgb = torch.stack(intm_rgbs, dim=0)            # (I, 3, H, W)
                intm_c2w = torch.stack(intm_c2w, dim=0).float()     # (I, 4, 4)
                intm_intrinsics = torch.stack(intm_intr, dim=0).float()
            else:
                intm_rgb = intm_c2w = intm_intrinsics = None

            # ----- Pose normalization relative to first context -----
            inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])  # (4,4)
            ctxt_rel = torch.einsum("ab,cbf->caf", inv_ctxt_c2w, ctxt_c2w)  # (C,4,4)
            trgt_rel = torch.einsum("ab,cbf->caf", inv_ctxt_c2w, trgt_c2w)  # (T,4,4)
            if intm_c2w is not None:
                intm_rel = torch.einsum("ab,cbf->caf", inv_ctxt_c2w, intm_c2w)  # (I,4,4)

            # ----- Pack return dict outputs -----
            ret_dict = {
                "ctxt_c2w": ctxt_rel,
                "trgt_c2w": trgt_rel,
                "ctxt_rgb": self.normalize(ctxt_rgb),
                "trgt_rgb": self.normalize(trgt_rgb),
                "ctxt_abs_camera_poses": ctxt_c2w,
                "trgt_abs_camera_poses": trgt_c2w,
                "intrinsics": ctxt_intrinsics[0],  # first context intrinsics
                "near": self.z_near,
                "far": self.z_far,
                "idx": torch.tensor([index]),
                "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
                "num_context": torch.tensor([1]),
                "lang": self.lang_identity,
            }

            if intm_rgb is not None:
                ret_dict.update({
                    "intm_c2w": intm_rel,
                    "intm_rgb": self.normalize(intm_rgb),
                    "intm_abs_camera_poses": intm_c2w,
                })

            # Keep same return signature as your original: (ret_dict, trgt_rgb)
            return ret_dict, trgt_rgb
        except Exception as e:
            return fallback()