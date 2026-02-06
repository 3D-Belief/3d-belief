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

    z_near: float = 0.01
    z_far: float = 50.0
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
        adjacent_angle: Optional[float] = np.pi/4,
    ) -> None:
        super().__init__()
        self.overfit_to_index = overfit_to_index
        self.num_context = num_context
        self.num_target = num_target
        self.context_min_distance = context_min_distance
        self.context_max_distance = context_max_distance
        self.adjacent_angle = adjacent_angle
        self.image_size = image_size
        self.intermediate = intermediate
        self.num_intermediate = num_intermediate
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

        dummy_img_path = str(next(scene_path_list[0].glob("rgb/*.png")))
        dummy_img = cv2.imread(dummy_img_path)
        h, w = dummy_img.shape[:2]

        if language_encoder is not None:
            self.lang_identity = language_encoder("").detach()

        self.len = 0
        all_rgb_files = []
        all_depth_files = []
        all_pose_files = []
        all_obj_binary_masks = []
        all_timestamps = []
        self.scene_path_list = []
        for i, scene_path in enumerate(scene_path_list):
            rgb_files = sorted(scene_path.glob("rgb/*.png"))
            depth_files = sorted(scene_path.glob("depth/*.npy"))
            pose_files = sorted(scene_path.glob("pose/*.npy"))
            obj_binary_mask_files = sorted(scene_path.glob("semantic/*object_binary_mask.png"))
            self.len += len(rgb_files)
            timestamps = [int(rgb_file.name.split(".")[0].split("_")[-1]) for rgb_file in rgb_files]
            sorted_ids = np.argsort(timestamps)
            all_rgb_files.append(np.array(rgb_files)[sorted_ids])
            all_depth_files.append(np.array(depth_files)[sorted_ids])
            all_pose_files.append(np.array(pose_files)[sorted_ids])
            all_obj_binary_masks.append(np.array(obj_binary_mask_files)[sorted_ids])
            self.scene_path_list.append(scene_path)
            all_timestamps.append(np.array(timestamps)[sorted_ids])

        self.indices = torch.arange(0, len(self.scene_path_list))
        self.all_rgb_files = all_rgb_files
        self.all_depth_files = all_depth_files
        self.all_pose_files = all_pose_files
        self.all_obj_binary_masks = all_obj_binary_masks
        self.all_timestamps = all_timestamps
        print("NUM IMAGES", self.len)
        print("NUM SCENES", len(self.scene_path_list))
        self.all_ctxt_rgb_files = self.all_rgb_files
        self.all_ctxt_depth_files = self.all_depth_files
        self.all_ctxt_pose_files = self.all_pose_files
        self.all_ctxt_obj_binary_masks = self.all_obj_binary_masks
        self.all_trgt_rgb_files = self.all_rgb_files
        self.all_trgt_depth_files = self.all_depth_files
        self.all_trgt_pose_files = self.all_pose_files
        self.all_trgt_obj_binary_masks = self.all_obj_binary_masks

    def read_frame(self, rgb_files, depth_files, pose_files, obj_binary_mask_files, id):
        rgb_file = rgb_files[id]
        depth_file = depth_files[id]
        pose_file = pose_files[id]
        obj_binary_mask_file = obj_binary_mask_files[id]
        rgb = (
            torch.tensor(
                np.asarray(Image.open(rgb_file)).astype(np.float32)
            ).permute(2, 0, 1)
            / 255.0
        )
        # print(rgb.shape, "SHAPE")
        rgb = F.interpolate(
            rgb.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            antialias=True,
        )[0]

        extrinsics = np.load(pose_file)
        extrinsics = np.linalg.inv(extrinsics)
        conversion = np.diag([1, -1, -1, 1])
        extrinsics = conversion @ extrinsics

        cam_param = Camera(extrinsics)

        depth = torch.tensor(np.load(depth_file), dtype=torch.float32).unsqueeze(0)
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            antialias=True,
        )[0]

        obj_binary_mask = (
            torch.tensor(
                np.asarray(Image.open(obj_binary_mask_file)).astype(np.float32)
            ).unsqueeze(0)
            / 255.0
        )
        obj_binary_mask = F.interpolate(
            obj_binary_mask.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        )[0]
        obj_binary_mask = obj_binary_mask > 0.5

        # add depth mask
        depth_mask = (depth < self.z_filter)

        return rgb, depth, obj_binary_mask, depth_mask, cam_param

    def __len__(self) -> int:
        return len(self.all_rgb_files)

    def __getitem__(self, index: int):
        seed = hash((index, self.global_seed)) % (2**32) + time.time_ns() % (2**32)
        self.rng = np.random.default_rng(seed)
        scene_idx = self.rng.integers(0, len(self.all_rgb_files))
        if self.overfit_to_index is not None:
            scene_idx = self.overfit_to_index

        def fallback():
            """Used if the desired index can't be loaded."""
            seed = hash((index, self.global_seed)) % (2**32) + time.time_ns() % (2**32)
            self.rng = np.random.default_rng(seed)
            return self[self.rng.integers(0, len(self.all_rgb_files))]

        rgb_files = self.all_rgb_files[scene_idx]
        depth_files = self.all_depth_files[scene_idx]
        pose_files = self.all_pose_files[scene_idx]
        obj_binary_mask_files = self.all_obj_binary_masks[scene_idx]
        timestamps = self.all_timestamps[scene_idx]
        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)
        if num_frames < 1 + 1:
            return fallback()
        
        start_idx = self.rng.choice(max(len(rgb_files)-self.context_max_distance, 2), 1)[0]

        if start_idx+1>=len(rgb_files)-1:
            return fallback()

        # z axis of in pose pf the start idx
        _, _, _, _, cam_param = self.read_frame(rgb_files, depth_files, pose_files, obj_binary_mask_files, start_idx)
        pose_ctxt = cam_param.c2w_mat
        z_start = pose_ctxt[:, 2][:3]

        for idx in range(start_idx+1, num_frames):
            _, _, _, _, cam_param = self.read_frame(rgb_files, depth_files, pose_files, obj_binary_mask_files, idx)
            pose_idx = cam_param.c2w_mat
            z_idx = pose_idx[:, 2][:3]
            angle = rotation_angle(z_start, z_idx)
            end_idx = idx
            if angle > self.adjacent_angle or idx-start_idx > self.context_max_distance:
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
        trgt_obj_binary_masks = []
        trgt_depth_masks = []
        trgt_c2w = []
        trgt_intrinsics = []
        for id in trgt_idx:
            rgb, depth, obj_binary_mask, depth_mask, cam_param = self.read_frame(rgb_files, depth_files, pose_files, obj_binary_mask_files, id)
            trgt_rgbs.append(rgb)
            trgt_depths.append(depth)
            trgt_obj_binary_masks.append(obj_binary_mask)
            trgt_depth_masks.append(depth_mask)
            trgt_intrinsics.append(cam_param.intrinsics)
            trgt_c2w.append(cam_param.c2w_mat)
        trgt_c2w = torch.tensor(np.array(trgt_c2w)).float()
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)
        trgt_depth = torch.stack(trgt_depths, axis=0)
        trgt_obj_binary_mask = torch.stack(trgt_obj_binary_masks, axis=0)
        trgt_depth_mask = torch.stack(trgt_depth_masks, axis=0)

        # load the ctxt
        ctxt_rgbs = []
        ctxt_depths = []
        ctxt_obj_binary_masks = []
        ctxt_depth_masks = []
        ctxt_c2w = []
        ctxt_intrinsics = []
        for id in ctxt_idx:
            rgb, depth, obj_binary_mask, depth_mask, cam_param = self.read_frame(rgb_files, depth_files, pose_files, obj_binary_mask_files, id)
            ctxt_rgbs.append(rgb)
            ctxt_depths.append(depth)
            ctxt_obj_binary_masks.append(obj_binary_mask)
            ctxt_depth_masks.append(depth_mask)
            ctxt_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            ctxt_c2w.append(cam_param.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, axis=0)
        ctxt_depth = torch.stack(ctxt_depths, axis=0)
        ctxt_depth_mask = torch.stack(ctxt_depth_masks, axis=0)
        ctxt_obj_binary_mask = torch.stack(ctxt_obj_binary_masks, axis=0)
        ctxt_intrinsics = torch.stack(ctxt_intrinsics, axis=0)

        # load the intermediate
        if self.intermediate and (intm_idx is not None):
            intm_rgbs = []
            intm_depths = []
            intm_obj_binary_masks = []
            intm_depth_masks = []
            intm_c2w = []
            intm_intrinsics = []
            for id in intm_idx:
                rgb, depth, obj_binary_mask, depth_mask, cam_param = self.read_frame(rgb_files, depth_files, pose_files, obj_binary_mask_files, id)
                intm_rgbs.append(rgb)
                intm_depths.append(depth)
                intm_obj_binary_masks.append(obj_binary_mask)
                intm_depth_masks.append(depth_mask)
                intm_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
                intm_c2w.append(cam_param.c2w_mat)
            intm_c2w = torch.tensor(np.array(intm_c2w)).float()
            intm_rgb = torch.stack(intm_rgbs, axis=0)
            intm_depth = torch.stack(intm_depths, axis=0)
            intm_depth_mask = torch.stack(intm_depth_masks, axis=0)
            intm_obj_binary_mask = torch.stack(intm_obj_binary_masks, axis=0)
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
                "ctxt_depth": ctxt_depth,
                "trgt_depth": trgt_depth,
                "ctxt_depth_mask": ctxt_depth_mask,
                "trgt_depth_mask": trgt_depth_mask,
                "ctxt_obj_binary_mask": ctxt_obj_binary_mask,
                "trgt_obj_binary_mask": trgt_obj_binary_mask,
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
        if self.intermediate and (intm_idx is not None):
            ret_dict.update({
                "intm_c2w": torch.einsum(
                    "ijk, ikl -> ijl", intm_inv_ctxt_c2w_repeat, intm_c2w
                ),
                "intm_rgb": self.normalize(intm_rgb),
                "intm_depth": intm_depth,
                "intm_depth_mask": intm_depth_mask,
                "intm_obj_binary_mask": intm_obj_binary_mask,
                "intm_abs_camera_poses": intm_c2w,
            })
        ret = (
            ret_dict,
            trgt_rgb,  # rearrange(rendered["image"], "c h w -> (h w) c"),
        )
        return ret

    # Data items for static inference
    def data_for_video(self, video_idx, ctxt_idx, trgt_idx, num_frames_render=20):
        scene_idx = video_idx
        rgb_files = self.all_rgb_files[scene_idx]
        depth_files = self.all_depth_files[scene_idx]
        pose_files = self.all_pose_files[scene_idx]
        timestamps = self.all_timestamps[scene_idx]
        assert (timestamps == sorted(timestamps)).all()

        trgt_rgbs = []
        trgt_depths = []
        trgt_c2w = []
        trgt_intrinsics = []
        for id in trgt_idx:
            id = min(id, len(rgb_files) - 1)
            id = max(id, 0)
            rgb, depth, cam_param = self.read_frame(rgb_files, depth_files, pose_files, id)
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
            id = min(id, len(rgb_files) - 1)
            id = max(id, 0)
            rgb, depth, cam_param = self.read_frame(rgb_files, depth_files, pose_files, id)
            ctxt_rgbs.append(rgb)
            ctxt_depths.append(depth)
            ctxt_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            ctxt_c2w.append(cam_param.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, axis=0)
        ctxt_depth = torch.stack(ctxt_depths, axis=0)
        ctxt_intrinsics = torch.stack(ctxt_intrinsics, axis=0)

        render_poses = []
        num_frames_render = min(ctxt_idx[0], len(rgb_files) - 1) - min(
            trgt_idx[0], len(rgb_files) - 1
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
            rgb_file = rgb_files[id]
            _, _, cam_param = self.read_frame(rgb_files, depth_files, pose_files, id)
            render_poses.append(cam_param.c2w_mat)
        render_poses = torch.tensor(np.array(render_poses)).float()

        print(
            f"ctxt_idx: {ctxt_idx}, trgt_idx: {trgt_idx}, num_frames_render: {num_frames_render}, len(rgb_files): {len(rgb_files)}"
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
            ctxt_rgb,  # rearrange(rendered["image"], "c h w -> (h w) c"),
            trgt_rgb,  # rearrange(rendered["image"], "c h w -> (h w) c"),
            "",
        )

    def data_for_temporal(self, video_idx: int, frames_render: Union[int, List]=20):
        scene_idx = video_idx
        rgb_files = self.all_rgb_files[scene_idx]
        depth_files = self.all_depth_files[scene_idx]
        pose_files = self.all_pose_files[scene_idx]

        # get a list of frame indices
        if isinstance(frames_render, (int, np.integer)):
            ids = select_random_sequence(self.rng, len(rgb_files), frames_render)
            start_idx = ids[0]
            end_idx = ids[-1]
        else:
            start_idx = frames_render[0]
            end_idx = frames_render[1]
            assert start_idx<len(rgb_files) and end_idx<len(rgb_files)
            ids = np.arange(start_idx, end_idx+1)

        rgbs = []
        depths = []
        pose_c2w = []
        intrinsics = []
        for id in ids:
            rgb, depth, cam_param = self.read_frame(rgb_files, depth_files, pose_files, id)
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
        
        assert len(render_poses)==len(ret["rgbs"])==len(ret["abs_camera_poses"])==len(ret["intrinsics"])

        return (
            ret,
            [torch.stack([rgb], axis=0) for rgb in rgbs],  # rearrange(rendered["image"], "c h w -> (h w) c"),
            start_idx,
            end_idx,
        )