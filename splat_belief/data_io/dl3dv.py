from pathlib import Path
from typing import List, Optional, Tuple, Union, Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from torch.utils.data import Dataset
from PIL import Image
import cv2
import time
from typing import Literal


from splat_belief.utils.vision_utils import normalize_to_neg_one_to_one, rotation_angle, inverse_transformation, select_random_sequence
from splat_belief.splat.layers import T5Encoder

Stage = Literal["train", "test", "unit"]


class CameraDL3DV(object):
    """Wrap DL3DV per-frame camera (.npz)."""
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


def _safe_imread_rgb(path: Union[str, Path]) -> np.ndarray:
    """Return HxWx3 float32 in [0,1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0)

def _normalize_K(K, H, W):
    """Normalize intrinsics to [0,1] image coords: u' = u/W, v' = v/H."""
    Kn = K.copy()
    Kn[0, 0] /= W
    Kn[1, 1] /= H
    Kn[0, 2] /= W
    Kn[1, 2] /= H
    return Kn

def _center_crop(x, size):
    """
    Center-crop an image to (size, size).
    Accepts:
      - HW (H, W)
      - HWC (H, W, C)
    Returns:
      - HWC (H, W, C) or HW (H, W)
    """
    H, W = x.shape[:2]
    start_x = (W - size) // 2
    start_y = (H - size) // 2
    x_cropped = x[start_y:start_y+size, start_x:start_x+size]
    return x_cropped

def _resize_chw(x, size, mode="bilinear"):
    """
    Resize an image/tensor to (size, size).
    Accepts:
      - HW (H, W)
      - CHW (C, H, W)
    Returns:
      - CHW (C, H, W)
    """
    # Ensure CHW
    if x.ndim == 2:          # HW -> CHW
        x = x.unsqueeze(0)
    elif x.ndim == 3:        # CHW
        pass
    else:
        raise ValueError(f"_resize_chw expects HW or CHW, got shape {tuple(x.shape)}")

    # Now make NCHW
    x4 = x.unsqueeze(0)       # 1, C, H, W

    # antialias only valid for bilinear/bicubic
    use_aa = mode in ("bilinear", "bicubic")

    # bool -> float for interpolation; restore later if nearest
    orig_dtype = x4.dtype
    if orig_dtype == torch.bool:
        x4 = x4.float()

    y4 = F.interpolate(x4, size=(size, size), mode=mode,
                       antialias=use_aa if use_aa else False)

    y = y4[0]                 # back to CHW

    # If this was a mask and we used nearest, return boolean
    if orig_dtype == torch.bool:
        if mode == "nearest":
            y = (y > 0.5)
        else:
            # For non-nearest, cast back to bool via threshold (or keep float if preferred)
            y = (y > 0.5)

    return y

class DL3DVDataset(Dataset):
    """
    Dataset for DL3DV.

    Directory layout per scene:
      <root>/<scene>/<subscene>/dense/
        rgb/*.png
        depth/*.npy
        cam/<basename>.npz   # keys: 'intrinsic' (3x3), 'pose' (4x4)
        sky_mask/*.png       # optional
        outlier_mask/*.png   # optional
    """
    examples: List[Path]
    stage: Stage

    z_near: float = 0.1
    z_far: float = 60.0
    image_size: int = 64

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
        image_size: Optional[int] = 64,
        adjacent_angle: Optional[float] = np.pi / 4,
        use_depth_supervision: bool = True,
        pose_is_c2w: bool = True,
        apply_depth_masks: bool = True,
        clamp_depth_quantile: float = 98.0,
        global_seed: int = 42,
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
        self.use_depth_supervision = use_depth_supervision
        self.image_size = int(image_size)
        self.intermediate = intermediate
        self.num_intermediate = num_intermediate
        self.pose_is_c2w = pose_is_c2w
        self.apply_depth_masks = apply_depth_masks
        self.clamp_depth_quantile = clamp_depth_quantile
        self.overfit_to_index = overfit_to_index
        self.normalize = normalize_to_neg_one_to_one
        self.rng = np.random.default_rng(global_seed)
        self.global_seed = global_seed

        if language_encoder is not None:
            self.lang_identity = language_encoder("").detach()
        else:
            self.lang_identity = None

        sub_dir = {"train": "train", "test": "test", "unit": "unit"}[stage]
        image_root = self.root / sub_dir
        scene_path_list = sorted([p for p in Path(image_root).glob("*/*/") if p.is_dir()])
        scene_roots = scene_path_list
        if max_scenes is not None:
            scene_roots = scene_path_list[:max_scenes]

        # Gather scenes: <root>/<scene>/<subscene>/dense/rgb/*.png
        # Only keep subscenes with enough frames.
        self.scene_dirs: List[Path] = []

        for scene in scene_roots:
            dense = scene / "dense"
            rgb_dir = dense / "rgb"
            if not rgb_dir.exists():
                continue
            rgb_paths = sorted([p for p in rgb_dir.glob("*.png")])
            if len(rgb_paths) == 0:
                continue
            self.scene_dirs.append(dense)

        # Build per-scene index lists
        self.all_basenames: List[np.ndarray] = []
        if self.use_depth_supervision:
            self.has_depth_for_scene: List[bool] = []

        for dense in self.scene_dirs:
            rgb_paths = sorted([p for p in (dense / "rgb").glob("*.png")])
            # Sort by numeric basename if possible, otherwise lexicographic
            def _key(p: Path):
                stem = p.stem
                try:
                    return int(stem)
                except ValueError:
                    return stem
            rgb_paths = sorted(rgb_paths, key=_key)
            self.all_basenames.append(np.array([p.stem for p in rgb_paths]))

            if self.use_depth_supervision:
                # Just record presence; file-by-file checked at load time
                self.has_depth_for_scene.append((dense / "depth").exists())

        self.indices = torch.arange(0, len(self.scene_dirs))
        print(f"[DL3DV] Scenes {len(self.scene_dirs)}")

    def __len__(self) -> int:
        return len(self.scene_dirs)

    # ---------- I/O helpers ----------
    def _read_frame(self, dense_dir: Path, basename: str) -> Tuple[torch.Tensor, CameraDL3DV]:
        rgb = _safe_imread_rgb(dense_dir / "rgb" / f"{basename}.png")  # HxWx3 in [0,1]
        height, width = rgb.shape[:2]
        rgb = _center_crop(rgb, rgb.shape[0] if rgb.shape[0]<rgb.shape[1] else rgb.shape[1])

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()  # CxHxW

        cam_npz = np.load(dense_dir / "cam" / f"{basename}.npz")
        K = cam_npz["intrinsic"].astype(np.float32)
        K = _normalize_K(K, H=height, W=width)
        rgb_t = _resize_chw(rgb_t, self.image_size, mode="bilinear")
        pose = cam_npz["pose"].astype(np.float32)  # 4x4
        cam = CameraDL3DV(K, pose, pose_is_c2w=self.pose_is_c2w)

        return rgb_t, cam

    def _read_depth(self, dense_dir: Path, basename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        depth = np.load(dense_dir / "depth" / f"{basename}.npy").astype(np.float32)  # HxW
        depth = _center_crop(depth, depth.shape[0] if depth.shape[0]<depth.shape[1] else depth.shape[1])

        if self.apply_depth_masks:
            sky_path = dense_dir / "sky_mask" / f"{basename}.png"
            outlier_path = dense_dir / "outlier_mask" / f"{basename}.png"
            if sky_path.exists():
                sky_mask = cv2.imread(str(sky_path), cv2.IMREAD_UNCHANGED)
                sky_mask = _center_crop(sky_mask, sky_mask.shape[0] if sky_mask.shape[0]<sky_mask.shape[1] else sky_mask.shape[1])
                if sky_mask is not None:
                    depth[sky_mask >= 127] = -1.0  # invalidate sky
            if outlier_path.exists():
                outlier_mask = cv2.imread(str(outlier_path), cv2.IMREAD_UNCHANGED)
                outlier_mask = _center_crop(outlier_mask, outlier_mask.shape[0] if outlier_mask.shape[0]<outlier_mask.shape[1] else outlier_mask.shape[1])
                if outlier_mask is not None:
                    depth[outlier_mask >= 127] = 0.0  # outliers as zero

            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            pos = depth > 0
            if pos.any() and self.clamp_depth_quantile is not None:
                thr = np.percentile(depth[pos], self.clamp_depth_quantile)
                depth[depth > thr] = 0.0  # clamp far tails to 0 (invalid)

        # valid where depth > 0 after masking/clamping
        mask = (depth > 0).astype(np.float32)  # HxW, float for resize

        # resize to (image_size, image_size)
        depth_t = torch.from_numpy(depth).float().unsqueeze(0)
        depth_t = _resize_chw(depth_t, self.image_size, "bilinear").squeeze(0)   # (H,W)

        mask = (depth > 0).astype(np.float32)  # 0/1 float
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        mask_t = _resize_chw(mask_t, self.image_size, "nearest").squeeze(0)      # (H,W) float

        return depth_t, mask_t

    # ---------- Sampling ----------
    def __getitem__(self, index: int):
        seed = (hash((int(index), self.global_seed)) % (2**32)) + (time.time_ns() % (2**32))
        self.rng = np.random.default_rng(seed)

        scene_idx = int(self.rng.integers(0, len(self.scene_dirs)))
        if self.overfit_to_index is not None:
            scene_idx = int(self.overfit_to_index)

        def fallback():
            seed_fb = (hash((int(index), self.global_seed)) % (2**32)) + (time.time_ns() % (2**32))
            self.rng = np.random.default_rng(seed_fb)
            return self[int(self.rng.integers(0, len(self.scene_dirs)))]

        dense = self.scene_dirs[scene_idx]
        basenames = self.all_basenames[scene_idx]
        num_frames = len(basenames)
        if num_frames < self.num_target + 1:
            return fallback()

        # Choose start frame constrained by context_max_distance
        max_start = max(num_frames - self.context_max_distance, 2)
        start_idx = int(self.rng.choice(max_start, 1)[0])
        if start_idx + 1 >= num_frames - 1:
            return fallback()

        # Get z-axis at start
        _, cam_start = self._read_frame(dense, basenames[start_idx])
        z_start = cam_start.c2w_mat[:3, 2].copy()

        # Find an end frame where angle exceeds threshold or distance exceeds max
        end_idx = min(start_idx + 1, num_frames - 1)
        for idx in range(start_idx + 1, num_frames):
            _, cam_i = self._read_frame(dense, basenames[idx])
            z_i = cam_i.c2w_mat[:3, 2].copy()
            ang = rotation_angle(z_start, z_i)
            end_idx = idx
            if ang > self.adjacent_angle or (idx - start_idx) > self.context_max_distance:
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
            # ensure at least one candidate; if equal, skip intermediates
            if hi > lo:
                choices = np.arange(lo, hi + 1)
                intm_idx = self.rng.choice(choices, self.num_intermediate, replace=True).tolist()

        # ----- Load target(s) -----
        trgt_rgbs, trgt_c2w, trgt_intr = [], [], []
        trgt_depths = [] if self.use_depth_supervision else None
        trgt_depth_masks = [] if self.use_depth_supervision else None
        for i in trgt_idx:
            rgb_t, cam = self._read_frame(dense, basenames[i])
            trgt_rgbs.append(rgb_t)
            trgt_c2w.append(torch.from_numpy(cam.c2w_mat))
            trgt_intr.append(torch.from_numpy(cam.intrinsics))
            if self.use_depth_supervision:
                d, m = self._read_depth(dense, basenames[i])
                trgt_depths.append(d)
                trgt_depth_masks.append(m)
        trgt_rgb = torch.stack(trgt_rgbs, dim=0)                      # (T, 3, H, W)
        trgt_c2w = torch.stack(trgt_c2w, dim=0).float()               # (T, 4, 4)
        if self.use_depth_supervision:
            trgt_depth = torch.stack(trgt_depths, dim=0)          # (T, H, W)
            trgt_depth_mask = torch.stack(trgt_depth_masks, dim=0)  # (T, H, W) bool

        # ----- Load context(s) -----
        ctxt_rgbs, ctxt_c2w, ctxt_intr = [], [], []
        ctxt_depths = [] if self.use_depth_supervision else None
        ctxt_depth_masks = [] if self.use_depth_supervision else None
        for i in ctxt_idx:
            rgb_t, cam = self._read_frame(dense, basenames[i])
            ctxt_rgbs.append(rgb_t)
            ctxt_c2w.append(torch.from_numpy(cam.c2w_mat))
            ctxt_intr.append(torch.from_numpy(cam.intrinsics))
            if self.use_depth_supervision:
                d, m = self._read_depth(dense, basenames[i])
                ctxt_depths.append(d)
                ctxt_depth_masks.append(m)
        
        ctxt_rgb = torch.stack(ctxt_rgbs, dim=0)                      # (C, 3, H, W)
        ctxt_c2w = torch.stack(ctxt_c2w, dim=0).float()               # (C, 4, 4)
        ctxt_intrinsics = torch.stack(ctxt_intr, dim=0).float()       # (C, 3, 3)
        if self.use_depth_supervision:
            ctxt_depth = torch.stack(ctxt_depths, dim=0)            # (C, H, W)
            ctxt_depth_mask = torch.stack(ctxt_depth_masks, dim=0)  # (C, H, W) bool

        # ----- Load intermediates (optional) -----
        if self.intermediate and intm_idx is not None and len(intm_idx) > 0:
            intm_rgbs, intm_c2w, intm_intr = [], [], []
            intm_depths = [] if self.use_depth_supervision else None
            intm_depth_masks = [] if self.use_depth_supervision else None
            for i in intm_idx:
                rgb_t, cam = self._read_frame(dense, basenames[int(i)])
                intm_rgbs.append(rgb_t)
                intm_c2w.append(torch.from_numpy(cam.c2w_mat))
                intm_intr.append(torch.from_numpy(cam.intrinsics))
                if self.use_depth_supervision:
                    d, m = self._read_depth(dense, basenames[int(i)])
                    intm_depths.append(d)
                    intm_depth_masks.append(m)
            
            intm_rgb = torch.stack(intm_rgbs, dim=0)                  # (I, 3, H, W)
            intm_c2w = torch.stack(intm_c2w, dim=0).float()           # (I, 4, 4)
            intm_intrinsics = torch.stack(intm_intr, dim=0).float()   # (I, 3, 3)
            if self.use_depth_supervision:
                intm_depth = torch.stack(intm_depths, dim=0)                # (I, H, W)
                intm_depth_mask = torch.stack(intm_depth_masks, dim=0)      # (I, H, W) bool
        else:
            intm_rgb = intm_c2w = intm_intrinsics = intm_depth = None

        # ----- Pose normalization relative to first context -----
        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])                     # (4,4)
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
        if self.use_depth_supervision:
            ret_dict.update({
                "ctxt_depth": ctxt_depth,
                "trgt_depth": trgt_depth,
                "ctxt_depth_mask": ctxt_depth_mask,  
                "trgt_depth_mask": trgt_depth_mask,  
            })
        if intm_rgb is not None:
            ret_dict.update({
                "intm_c2w": intm_rel,                         # (I,4,4)
                "intm_rgb": self.normalize(intm_rgb),         # (I,3,H,W)
                "intm_abs_camera_poses": intm_c2w,            # (I,4,4)
            })
            if intm_rgb is not None and self.use_depth_supervision:
                ret_dict.update({
                    "intm_depth": intm_depth,
                    "intm_depth_mask": intm_depth_mask,  
                })
        return ret_dict, trgt_rgb
