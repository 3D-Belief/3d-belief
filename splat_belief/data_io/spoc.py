from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from torch.utils.data import Dataset
from numpy.random import default_rng
from splat_belief.utils.vision_utils import *
from splat_belief.splat.layers import T5Encoder
from numpy import random
import scipy
import cv2
import time
from PIL import Image
from numpy.random import default_rng

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


Stage = Literal["train", "test", "unit", "one", "one_test"]

class Camera(object):
    def __init__(self, extrinsics):
        fx, fy, cx, cy = 0.390, 0.385, 0.5, 0.5

        self.intrinsics = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32,
        )

        self.w2c_mat = extrinsics
        self.c2w_mat = np.linalg.inv(extrinsics)

class SPOCDataset(Dataset):
    examples: List[Path]
    stage: Stage
    to_tensor: tf.ToTensor
    overfit_to_index: Optional[int]
    num_target: int
    context_min_distance: int
    context_max_distance: int

    z_near: float = 0.2
    z_far: float = 20.0
    z_filter: float = 19.0
    image_size: int = 64
    background_color: torch.tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

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
        adjacent_distance: Optional[float] = 1.0,
        use_depth_supervision: bool = True,
        load_class_maps: bool = False,
        class_name_to_id: Optional[dict] = None,
        class_color_tolerance: float = 25.0,
    ) -> None:
        super().__init__()
        self.overfit_to_index = overfit_to_index
        self.num_context = num_context
        self.num_target = num_target
        self.context_min_distance = context_min_distance
        self.context_max_distance = context_max_distance
        self.adjacent_angle = adjacent_angle
        self.adjacent_distance = adjacent_distance
        self.use_depth_supervision = use_depth_supervision
        self.image_size = image_size
        self.intermediate = intermediate
        self.num_intermediate = num_intermediate

        self.load_class_maps = load_class_maps
        self.class_name_to_id = class_name_to_id or {}
        self.class_color_tolerance = float(class_color_tolerance)
        self._scene_meta_cache = {}  # scene_path -> list[frame meta dict]
        sub_dir = {"train": "train", "test": "test", "unit": "unit"}[stage]
        image_root = Path(root) / sub_dir
        scene_path_list = sorted([p for p in Path(image_root).glob("*/") if p.is_dir()])

        if max_scenes is not None:
            scene_path_list = scene_path_list[:max_scenes]
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        self.rng = default_rng()
        self.global_seed = 42
        self.normalize = normalize_to_neg_one_to_one

        if language_encoder is not None:
            self.lang_identity = language_encoder("").detach()

        self.len = 0
        self.scene_path_list = []
        self.num_frames_per_scene = []
        
        for scene_path in scene_path_list:
            # Check if the new data format exists
            rgb_file = scene_path / "rgb_trajectory.mp4"
            depth_file = scene_path / "all_depths.npz"
            pose_file = scene_path / "all_poses.npz"
            
            if rgb_file.exists() and depth_file.exists() and pose_file.exists():
                # Load poses to get number of frames
                poses_data = np.load(pose_file)
                num_frames = poses_data['poses'].shape[0]
                self.len += num_frames
                self.num_frames_per_scene.append(num_frames)
                self.scene_path_list.append(scene_path)

        print("[SPOC] Scenes", len(self.scene_path_list))
        print("length dataset", self.len)

    def read_frame(self, scene_path, frame_id):
        """
        Read RGB frame, depth, and camera pose from npz and video files.
        
        Args:
            scene_path: Path to scene directory
            frame_id: Frame index
            
        Returns:
            rgb: RGB image tensor [3, H, W]
            depth: Depth map [H, W]
            depth_mask: Depth mask [H, W]
            cam_param: Camera object with intrinsics and pose
        """
        # Read RGB from video
        rgb_file = scene_path / "rgb_trajectory.mp4"
        cap = cv2.VideoCapture(str(rgb_file))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        rgb = torch.tensor(frame.astype(np.float32)).permute(2, 0, 1) / 255.0
        rgb = F.interpolate(
            rgb.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            antialias=True,
        )[0]

        # Read depth from npz
        depth_file = scene_path / "all_depths.npz"
        depths_data = np.load(depth_file)
        depth_val = depths_data['depths'][frame_id]  # shape [240, 240]
        depth = torch.tensor(depth_val, dtype=torch.float32).unsqueeze(0)
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            antialias=True,
        )[0]

        # Add depth mask
        depth_mask = (depth < self.z_filter)

        # Read pose from npz
        pose_file = scene_path / "all_poses.npz"
        poses_data = np.load(pose_file)
        extrinsics = poses_data['poses'][frame_id]  # shape [4, 4]
        extrinsics = np.linalg.inv(extrinsics)
        conversion = np.diag([1, -1, -1, 1])
        extrinsics = conversion @ extrinsics

        cam_param = Camera(extrinsics)

        return rgb, depth, depth_mask, cam_param

    def _scene_meta(self, scene_path):
        key = str(scene_path)
        if key in self._scene_meta_cache:
            return self._scene_meta_cache[key]
        meta_path = scene_path / "all_semantic_meta.json"
        if not meta_path.exists():
            self._scene_meta_cache[key] = None
            return None
        import json as _json
        with open(meta_path) as f:
            meta = _json.load(f)
        self._scene_meta_cache[key] = meta
        return meta

    def _read_class_map(self, scene_path, frame_id):
        """Build a per-pixel class-id map for one frame.

        Reads the lossy semantic_trajectory.mp4 frame, finds the closest GT
        instance color (within tolerance) per pixel, maps that instance's
        `name` field through `class_name_to_id`, and returns an int64 tensor
        of shape (image_size, image_size). Pixels with no matching color or
        with name not in the vocabulary become 0 (= ignore).
        """
        meta = self._scene_meta(scene_path)
        if meta is None or frame_id >= len(meta):
            return torch.zeros((self.image_size, self.image_size), dtype=torch.long)

        sem_path = scene_path / "semantic_trajectory.mp4"
        if not sem_path.exists():
            return torch.zeros((self.image_size, self.image_size), dtype=torch.long)

        cap = cv2.VideoCapture(str(sem_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return torch.zeros((self.image_size, self.image_size), dtype=torch.long)
        sem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8
        H, W, _ = sem_rgb.shape

        objects = meta[int(frame_id)].get("objects", {})
        if not objects:
            return torch.zeros((self.image_size, self.image_size), dtype=torch.long)

        # Build per-instance (color, class_id) only for known classes.
        colors, class_ids = [], []
        for inst_id, info in objects.items():
            name = info.get("name", "Unknown")
            cid = self.class_name_to_id.get(name, 0)
            if cid == 0:
                continue
            colors.append(info["color"])
            class_ids.append(cid)
        if not colors:
            cmap = np.zeros((H, W), dtype=np.int64)
        else:
            colors_np = np.asarray(colors, dtype=np.int16)             # (K, 3)
            class_ids_np = np.asarray(class_ids, dtype=np.int64)        # (K,)
            # For each pixel, find nearest known color; mark ignore if it's
            # farther than `class_color_tolerance`.
            sem_int = sem_rgb.astype(np.int16)
            # Distance tensor (H, W, K)
            diff = sem_int[:, :, None, :] - colors_np[None, None, :, :]
            d2 = (diff.astype(np.int32) ** 2).sum(axis=-1)
            nearest = d2.argmin(axis=-1)
            min_d2 = np.take_along_axis(d2, nearest[..., None], axis=-1).squeeze(-1)
            tol2 = self.class_color_tolerance ** 2
            cmap = np.where(min_d2 <= tol2, class_ids_np[nearest], 0).astype(np.int64)

        # Resize to image_size with nearest-neighbor preservation.
        if (H, W) != (self.image_size, self.image_size):
            cmap = cv2.resize(
                cmap, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
        return torch.from_numpy(cmap)

    def __len__(self) -> int:
        return len(self.scene_path_list)

    def __getitem__(self, index: int):
        seed = hash((index, self.global_seed)) % (2**32) + time.time_ns() % (2**32)
        self.rng = np.random.default_rng(seed)
        scene_idx = self.rng.integers(0, len(self.scene_path_list))
        if self.overfit_to_index is not None:
            scene_idx = self.overfit_to_index

        def fallback():
            """Used if the desired index can't be loaded."""
            seed = hash((index, self.global_seed)) % (2**32) + time.time_ns() % (2**32)
            self.rng = np.random.default_rng(seed)
            return self[self.rng.integers(0, len(self.scene_path_list))]

        scene_path = self.scene_path_list[scene_idx]
        num_frames = self.num_frames_per_scene[scene_idx]
        
        if num_frames < 1 + 1:
            return fallback()
        
        start_idx = self.rng.choice(max(num_frames - self.context_max_distance, 2), 1)[0]

        if start_idx + 1 >= num_frames - 1:
            return fallback()

        # z axis and translation of the start pose
        _, _, _, cam_param = self.read_frame(scene_path, start_idx)
        pose_ctxt = cam_param.c2w_mat
        z_start = pose_ctxt[:, 2][:3]
        t_start = pose_ctxt[:, 3][:3]

        for idx in range(start_idx + 1, num_frames):
            _, _, _, cam_param = self.read_frame(scene_path, idx)
            pose_idx = cam_param.c2w_mat
            z_idx = pose_idx[:, 2][:3]
            t_idx = pose_idx[:, 3][:3]
            angle = rotation_angle(z_start, z_idx)
            dist = np.linalg.norm(t_idx - t_start)
            end_idx = idx
            if (
                angle > self.adjacent_angle
                or dist > self.adjacent_distance
                or idx - start_idx > self.context_max_distance
            ):
                end_idx = idx
                break

        flip = self.rng.choice([True, False])
        if flip:
            temp = start_idx
            start_idx = end_idx
            end_idx = temp

        ctxt_idx = [start_idx]
        trgt_idx = [end_idx]

        # intermediate frames
        if self.intermediate:
            start = min(trgt_idx[0], ctxt_idx[0])
            end = max(trgt_idx[0], ctxt_idx[0])
            available_choices = np.arange(start, end)
            intm_idx = self.rng.choice(available_choices, self.num_intermediate, replace=True)

        trgt_rgbs = []
        trgt_depths = []
        trgt_depth_masks = []
        trgt_c2w = []
        trgt_intrinsics = []
        for id in trgt_idx:
            rgb, depth, depth_mask, cam_param = self.read_frame(scene_path, id)
            trgt_rgbs.append(rgb)
            trgt_depths.append(depth)
            trgt_depth_masks.append(depth_mask)
            trgt_intrinsics.append(cam_param.intrinsics)
            trgt_c2w.append(cam_param.c2w_mat)
        trgt_c2w = torch.tensor(np.array(trgt_c2w)).float()
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)
        trgt_depth = torch.stack(trgt_depths, axis=0)
        trgt_depth_mask = torch.stack(trgt_depth_masks, axis=0)

        # load the ctxt
        ctxt_rgbs = []
        ctxt_depths = []
        ctxt_depth_masks = []
        ctxt_c2w = []
        ctxt_intrinsics = []
        for id in ctxt_idx:
            rgb, depth, depth_mask, cam_param = self.read_frame(scene_path, id)
            ctxt_rgbs.append(rgb)
            ctxt_depths.append(depth)
            ctxt_depth_masks.append(depth_mask)
            ctxt_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            ctxt_c2w.append(cam_param.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, axis=0)
        ctxt_depth = torch.stack(ctxt_depths, axis=0)
        ctxt_depth_mask = torch.stack(ctxt_depth_masks, axis=0)
        ctxt_intrinsics = torch.stack(ctxt_intrinsics, axis=0)

        # load the intermediate
        if self.intermediate and (intm_idx is not None):
            intm_rgbs = []
            intm_depths = []
            intm_depth_masks = []
            intm_c2w = []
            intm_intrinsics = []
            for id in intm_idx:
                rgb, depth, depth_mask, cam_param = self.read_frame(scene_path, id)
                intm_rgbs.append(rgb)
                intm_depths.append(depth)
                intm_depth_masks.append(depth_mask)
                intm_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
                intm_c2w.append(cam_param.c2w_mat)
            intm_c2w = torch.tensor(np.array(intm_c2w)).float()
            intm_rgb = torch.stack(intm_rgbs, axis=0)
            intm_depth = torch.stack(intm_depths, axis=0)
            intm_depth_mask = torch.stack(intm_depth_masks, axis=0)
            intm_intrinsics = torch.stack(intm_intrinsics, axis=0)

        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])
        ctxt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(1, 1, 1)
        trgt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(1, 1, 1)
        if self.intermediate and (intm_idx is not None):
            intm_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(self.num_intermediate, 1, 1)
        ret_dict = {
                "ctxt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", ctxt_inv_ctxt_c2w_repeat, ctxt_c2w
                ),
                "trgt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", trgt_inv_ctxt_c2w_repeat, trgt_c2w
                ),
                "ctxt_rgb": self.normalize(ctxt_rgb),
                "trgt_rgb": self.normalize(trgt_rgb),
                "ctxt_abs_camera_poses": ctxt_c2w,
                "trgt_abs_camera_poses": trgt_c2w,
                "intrinsics": ctxt_intrinsics[0],
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
        if self.load_class_maps:
            ret_dict.update({
                "ctxt_class_map": torch.stack(
                    [self._read_class_map(scene_path, fid) for fid in ctxt_idx], dim=0
                ),
                "trgt_class_map": torch.stack(
                    [self._read_class_map(scene_path, fid) for fid in trgt_idx], dim=0
                ),
            })
        if self.intermediate and (intm_idx is not None):
            ret_dict.update({
                "intm_c2w": torch.einsum(
                    "ijk, ikl -> ijl", intm_inv_ctxt_c2w_repeat, intm_c2w
                ),
                "intm_rgb": self.normalize(intm_rgb),
                "intm_abs_camera_poses": intm_c2w,
            })
            if self.load_class_maps:
                ret_dict["intm_class_map"] = torch.stack(
                    [self._read_class_map(scene_path, int(fid)) for fid in intm_idx], dim=0
                )
            if self.use_depth_supervision:
                ret_dict.update({
                    "intm_depth": intm_depth,
                    "intm_depth_mask": intm_depth_mask,
                })
        ret = (
            ret_dict,
            trgt_rgb,
        )
        return ret

    # Data items for static inference
    def data_for_video(self, video_idx, ctxt_idx, trgt_idx, num_frames_render=20):
        scene_path = self.scene_path_list[video_idx]
        num_frames = self.num_frames_per_scene[video_idx]

        trgt_rgbs = []
        trgt_depths = []
        trgt_c2w = []
        trgt_intrinsics = []
        for id in trgt_idx:
            id = min(id, num_frames - 1)
            id = max(id, 0)
            rgb, depth, depth_mask, cam_param = self.read_frame(scene_path, id)
            trgt_rgbs.append(rgb)
            trgt_depths.append(depth)
            trgt_intrinsics.append(cam_param.intrinsics)
            trgt_c2w.append(cam_param.c2w_mat)
        trgt_c2w = torch.tensor(np.array(trgt_c2w)).float()
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)
        trgt_depth = torch.stack(trgt_depths, axis=0)

        # load the ctxt
        ctxt_rgbs = []
        ctxt_depths = []
        ctxt_c2w = []
        ctxt_intrinsics = []
        for id in ctxt_idx:
            id = min(id, num_frames - 1)
            id = max(id, 0)
            rgb, depth, depth_mask, cam_param = self.read_frame(scene_path, id)
            ctxt_rgbs.append(rgb)
            ctxt_depths.append(depth)
            ctxt_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            ctxt_c2w.append(cam_param.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, axis=0)
        ctxt_depth = torch.stack(ctxt_depths, axis=0)
        ctxt_intrinsics = torch.stack(ctxt_intrinsics, axis=0)

        render_poses = []
        num_frames_render = min(ctxt_idx[0], num_frames - 1) - min(
            trgt_idx[0], num_frames - 1
        )
        noflip = False
        if num_frames_render < 0:
            noflip = True
            num_frames_render *= -1

        for i in range(1, num_frames_render + 1):
            if noflip:
                id = ctxt_idx[0] + i
            else:
                id = trgt_idx[0] + i
            _, _, _, cam_param = self.read_frame(scene_path, id)
            render_poses.append(cam_param.c2w_mat)
        render_poses = torch.tensor(np.array(render_poses)).float()

        print(
            f"ctxt_idx: {ctxt_idx}, trgt_idx: {trgt_idx}, num_frames_render: {num_frames_render}, num_frames: {num_frames}"
        )

        num_frames_render = render_poses.shape[0]
        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])
        ctxt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            1, 1, 1
        )
        trgt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            1, 1, 1
        )

        return (
            {
                "ctxt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", ctxt_inv_ctxt_c2w_repeat, ctxt_c2w
                ),
                "trgt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", trgt_inv_ctxt_c2w_repeat, trgt_c2w
                ),
                "ctxt_rgb": self.normalize(ctxt_rgb),
                "trgt_rgb": self.normalize(trgt_rgb),
                "ctxt_depth": ctxt_depth,
                "trgt_depth": trgt_depth,
                "ctxt_abs_camera_poses": ctxt_c2w,
                "trgt_abs_camera_poses": trgt_c2w,
                "intrinsics": ctxt_intrinsics[0],
                "near": self.z_near,
                "far": self.z_far,
                "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
                "lang": self.lang_identity,
            },
            ctxt_rgb,
            trgt_rgb,
            "",
        )

    def data_for_temporal(self, video_idx: int, frames_render: Union[int, List]=20):
        scene_path = self.scene_path_list[video_idx]
        num_frames = self.num_frames_per_scene[video_idx]

        # get a list of frame indices
        if isinstance(frames_render, (int, np.integer)):
            ids = select_random_sequence(self.rng, num_frames, frames_render)
            start_idx = ids[0]
            end_idx = ids[-1]
        else:
            start_idx = frames_render[0]
            end_idx = frames_render[1]
            assert start_idx < num_frames and end_idx < num_frames
            ids = np.arange(start_idx, end_idx + 1)

        rgbs = []
        depths = []
        pose_c2w = []
        intrinsics = []
        for id in ids:
            rgb, depth, depth_mask, cam_param = self.read_frame(scene_path, id)
            rgbs.append(rgb)
            depths.append(depth)
            intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            pose_c2w.append(cam_param.c2w_mat)

        print(
            f"start_idx: {ids[0]}, end_idx: {ids[-1]}, num_frames_render: {len(ids)}"
        )

        abs_camera_poses = [torch.tensor(np.array([c2w])).float() for c2w in pose_c2w]
        inv_camera_poses = [inverse_transformation(c2w[0]) for c2w in abs_camera_poses]
        inv_camera_poses_repeat = [inv_camera_poses[0].unsqueeze(0).repeat(1, 1, 1) for _ in inv_camera_poses]
        render_poses = []
        
        for i, (inv_c2w_repeat, c2w) in enumerate(zip(inv_camera_poses_repeat, abs_camera_poses)):
            render_poses.append(torch.einsum("ijk, ikl -> ijl", inv_c2w_repeat, c2w))

        ret = {
                "render_poses": render_poses,
                "rgbs": [self.normalize(torch.stack([rgb], axis=0)) for rgb in rgbs],
                "depth": [torch.stack([depth], axis=0) for depth in depths],
                "abs_camera_poses": abs_camera_poses,
                "intrinsics": [torch.stack([intrinsic], axis=0)[0] for intrinsic in intrinsics],
                "near": self.z_near,
                "far": self.z_far,
                "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
                "lang": self.lang_identity,
            }
        
        assert len(render_poses) == len(ret["rgbs"]) == len(ret["abs_camera_poses"]) == len(ret["intrinsics"])

        return (
            ret,
            [torch.stack([rgb], axis=0) for rgb in rgbs],
            start_idx,
            end_idx,
        )
