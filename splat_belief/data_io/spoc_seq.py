from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import time
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

class SPOCDatasetSeq(Dataset):
    examples: List[Path]
    stage: Stage
    to_tensor: tf.ToTensor
    overfit_to_index: Optional[int]
    num_target: int
    context_min_distance: int
    context_max_distance: int

    z_near: float = 0.01
    z_far: float = 50.0
    image_size: int = 64
    background_color: torch.tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    avg_key_frame_interval = 8

    def __init__(
        self,
        root: Union[str, Path],
        num_context: int,
        num_target: int,
        context_min_distance: int,
        context_max_distance: int,
        stage: Stage = "train",
        num_intermediate: Optional[int] = 3,
        language_encoder: Optional[T5Encoder] = None,
        overfit_to_index: Optional[int] = None,
        max_scenes: Optional[int] = None,
        image_size: Optional[int] = 64,
        adjacent_angle: float = 0.523,
        adjacent_distance: float = 1.0,
    ) -> None:
        super().__init__()
        self.overfit_to_index = overfit_to_index
        self.num_context = num_context
        self.num_target = num_target
        self.context_min_distance = context_min_distance
        self.context_max_distance = context_max_distance
        self.image_size = image_size
        self.num_intermediate = num_intermediate
        self.adjacent_angle = adjacent_angle
        self.adjacent_distance = adjacent_distance
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
        # check if there is a depth/ directory
        depth_dir = scene_path_list[0] / "depth"
        if not depth_dir.exists():
            self.use_depth = False
        else:
            self.use_depth = True

        if language_encoder is not None:
            self.lang_identity = language_encoder("").detach()

        self.len = 0
        all_rgb_files = []
        all_depth_files = []
        all_pose_files = []
        all_timestamps = []
        self.scene_path_list = []
        for i, scene_path in enumerate(scene_path_list):
            rgb_files = sorted(scene_path.glob("rgb/*.png"))
            if self.use_depth:
                depth_files = sorted(scene_path.glob("depth/*.npy"))
            pose_files = sorted(scene_path.glob("pose/*.npy"))
            self.len += len(rgb_files)
            timestamps = [int(rgb_file.name.split(".")[0].split("_")[-1]) for rgb_file in rgb_files]
            sorted_ids = np.argsort(timestamps)
            all_rgb_files.append(np.array(rgb_files)[sorted_ids])
            if self.use_depth:
                all_depth_files.append(np.array(depth_files)[sorted_ids])
            else:
                all_depth_files.append([])
            all_pose_files.append(np.array(pose_files)[sorted_ids])
            self.scene_path_list.append(scene_path)
            all_timestamps.append(np.array(timestamps)[sorted_ids])
        self.indices = torch.arange(0, len(self.scene_path_list))
        self.all_rgb_files = all_rgb_files
        self.all_depth_files = all_depth_files # empty if no depth
        self.all_pose_files = all_pose_files
        self.all_timestamps = all_timestamps
        print("NUM IMAGES", self.len)
        print("NUM SCENES", len(self.scene_path_list))
        self.all_ctxt_rgb_files = self.all_rgb_files
        self.all_ctxt_depth_files = self.all_depth_files
        self.all_ctxt_pose_files = self.all_pose_files
        self.all_trgt_rgb_files = self.all_rgb_files
        self.all_trgt_depth_files = self.all_depth_files
        self.all_trgt_pose_files = self.all_pose_files

    def read_frame(self, rgb_files, depth_files, pose_files, id):
        rgb_file = rgb_files[id]
        if self.use_depth:
            depth_file = depth_files[id]
        pose_file = pose_files[id]
        rgb = (
            torch.tensor(
                np.asarray(Image.open(rgb_file)).astype(np.float32)
            ).permute(2, 0, 1)
            / 255.0
        )
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

        if self.use_depth:
            depth = torch.tensor(np.load(depth_file), dtype=torch.float32).unsqueeze(0)
            depth = F.interpolate(
                depth.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                antialias=True,
            )[0]
        else:
            depth = None
        return rgb, depth, cam_param

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

        context_max_distance = self.context_max_distance * self.num_context

        rgb_files = self.all_rgb_files[scene_idx]
        depth_files = self.all_depth_files[scene_idx]
        pose_files = self.all_pose_files[scene_idx]

        # get a list of frame indices
        if len(rgb_files)<2:
            return fallback()

        # select starting idx
        start_idx = self.rng.integers(1, max(2, len(rgb_files)-context_max_distance*self.avg_key_frame_interval))
        ids = list(range(start_idx, len(rgb_files)))
        first_id = ids.pop(0)
        rgb, depth, cam_param = self.read_frame(rgb_files, depth_files, pose_files, first_id)

        rgbs = [rgb]
        if self.use_depth:
            depths = [depth]
        pose_c2w = [cam_param.c2w_mat]
        intrinsics = [torch.tensor(np.array(cam_param.intrinsics)).float()]
        inv_camera_pose_first = inverse_transformation(torch.tensor(np.array([cam_param.c2w_mat])).float()[0])
        intms_dict = {
            "rgbs": [self.normalize(torch.stack([rgb], axis=0))]*self.num_intermediate,
            "intrinsics": [torch.stack([torch.tensor(np.array(cam_param.intrinsics)).float()], axis=0)[0]]*self.num_intermediate,
            "pose": [torch.einsum("ijk, ikl -> ijl", inv_camera_pose_first.unsqueeze(0).repeat(1, 1, 1), torch.tensor(np.array([cam_param.c2w_mat])).float())]*self.num_intermediate,
        }
        if self.use_depth:
            intms_dict["depth"] = [torch.stack([depth], axis=0)]*self.num_intermediate
        intms = [intms_dict]

        intm_keys = ["rgbs", "depth", "intrinsics", "pose"] if self.use_depth else ["rgbs", "intrinsics", "pose"]

        z_start = cam_param.c2w_mat[:, 2][:3]  # initial forward vector
        t_start = cam_param.c2w_mat[:, 3][:3]  # initial translation
        z_previous = z_start
        t_previous = t_start
        current_intm = {k: [] for k in intm_keys}

        for id in ids:
            rgb, depth, cam_param = self.read_frame(rgb_files, depth_files, pose_files, id)
            current_pose = cam_param.c2w_mat
            z_idx = current_pose[:, 2][:3]  # current forward vector
            t_idx = current_pose[:, 3][:3]  # current translation
            angle = rotation_angle(z_previous, z_idx)
            distance = np.linalg.norm(t_idx - t_previous)
            current_intm["rgbs"].append(rgb)
            if self.use_depth:
                current_intm["depth"].append(depth)
            current_intm["intrinsics"].append(torch.tensor(np.array(cam_param.intrinsics)).float())
            current_intm["pose"].append(cam_param.c2w_mat)

            if angle > self.adjacent_angle or distance > self.adjacent_distance:
                z_previous = z_idx
                t_previous = t_idx
                rgbs.append(rgb)
                if self.use_depth:
                    depths.append(depth)
                intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
                pose_c2w.append(cam_param.c2w_mat)
                intm_indices = self.rng.choice(len(current_intm["rgbs"]),
                          size=self.num_intermediate,
                          replace=True)
                current_intm = {
                    key: [current_intm[key][i] for i in intm_indices]
                    for key in current_intm
                }
                current_intm["rgbs"] = [self.normalize(torch.stack([rgb], axis=0)) for rgb in current_intm["rgbs"]]
                if self.use_depth:
                    current_intm["depth"] = [torch.stack([depth], axis=0) for depth in current_intm["depth"]]
                current_intm["intrinsics"] = [torch.stack([intrinsic], axis=0)[0] for intrinsic in current_intm["intrinsics"]]

                abs_camera_poses = [torch.tensor(np.array([c2w])).float() for c2w in current_intm["pose"]]
                inv_camera_poses_repeat = [inv_camera_pose_first.unsqueeze(0).repeat(1, 1, 1) for _ in current_intm["pose"]]
                
                for intm_idx, (inv_c2w_repeat, c2w) in enumerate(zip(inv_camera_poses_repeat, abs_camera_poses)):
                    current_intm["pose"][intm_idx] = torch.einsum("ijk, ikl -> ijl", inv_c2w_repeat, c2w)
                
                assert len(current_intm["rgbs"])==len(current_intm["pose"])

                intms.append(current_intm)
                current_intm = {k: [] for k in intm_keys}
            
            if len(rgbs) >= context_max_distance:
                break
        
        # cycle padding
        cur_len = len(rgbs)
        if cur_len < context_max_distance:
            pad_count = context_max_distance - cur_len
            n = cur_len
            # build one back-and-forth cycle of indices
            if n > 1:
                # go from last-1 down to 0, then up from 1 to last
                pad_indices = list(range(n-2, -1, -1)) + list(range(1, n))
            else:
                pad_indices = [0]
            # append samples in zig-zag order until full
            for i in range(pad_count):
                idx = pad_indices[i % len(pad_indices)]
                rgbs.append(rgbs[idx])
                if self.use_depth:
                    depths.append(depths[idx])
                intrinsics.append(intrinsics[idx])
                pose_c2w.append(pose_c2w[idx])
                intms.append(intms[idx])

        assert len(rgbs)==context_max_distance and len(rgbs)==len(pose_c2w)

        abs_camera_poses = [torch.tensor(np.array([c2w])).float() for c2w in pose_c2w]
        inv_camera_poses = [inverse_transformation(c2w[0]) for c2w in abs_camera_poses]
        inv_camera_poses_repeat = [inv_camera_poses[0].unsqueeze(0).repeat(1, 1, 1) for _ in inv_camera_poses]
        render_poses = []
        
        for i, (inv_c2w_repeat, c2w) in enumerate(zip(inv_camera_poses_repeat, abs_camera_poses)):
            render_poses.append(torch.einsum("ijk, ikl -> ijl", inv_c2w_repeat, c2w))

        ret = {
                "render_poses": render_poses,
                "rgbs": [self.normalize(torch.stack([rgb], axis=0)) for rgb in rgbs],
                "intms": intms,
                "abs_camera_poses": abs_camera_poses,
                "intrinsics": [torch.stack([intrinsic], axis=0)[0] for intrinsic in intrinsics],
                "near": self.z_near,
                "far": self.z_far,
                "idx": torch.tensor([index]),
                "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
                "lang": self.lang_identity,
            }
        if self.use_depth:
            ret["depth"] = [torch.stack([depth], axis=0) for depth in depths]

        assert len(render_poses)==len(ret["rgbs"])==len(ret["abs_camera_poses"])==len(ret["intrinsics"])

        return (
            ret,
            [torch.stack([rgb], axis=0) for rgb in rgbs],  # rearrange(rendered["image"], "c h w -> (h w) c"),
            len(ids),
        )

    # Data items for temporal inference
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
        if self.use_depth:
            depths = []
        pose_c2w = []
        intrinsics = []
        for id in ids:
            rgb, depth, cam_param = self.read_frame(rgb_files, depth_files, pose_files, id)
            rgbs.append(rgb)
            if self.use_depth:
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
                "abs_camera_poses": abs_camera_poses,
                "intrinsics": [torch.stack([intrinsic], axis=0)[0] for intrinsic in intrinsics],
                "near": self.z_near,
                "far": self.z_far,
                "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
                "lang": self.lang_identity,
            }
        if self.use_depth:
            ret["depth"] = [torch.stack([depth], axis=0) for depth in depths] 

        assert len(render_poses)==len(ret["rgbs"])==len(ret["abs_camera_poses"])==len(ret["intrinsics"])

        return (
            ret,
            [torch.stack([rgb], axis=0) for rgb in rgbs],  # rearrange(rendered["image"], "c h w -> (h w) c"),
            start_idx,
            end_idx,
        )