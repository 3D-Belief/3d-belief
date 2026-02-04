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
from utils import *
from layers import T5Encoder
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


Stage = Literal["train", "test", "unit"]

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[:4]
        # fx = fx * (640.0 / 360.0)
        assert np.allclose(cx, 0.5)
        assert np.allclose(cy, 0.5)
        # print("intrinsics", fx, fy)

        self.intrinsics = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32,
        )

        w2c_mat = np.array(entry[6:], dtype=np.float32).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4, dtype=np.float32)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

class RealEstate10kDatasetSeq(Dataset):
    examples: List[Path]
    pose_file: Path
    stage: Stage
    to_tensor: tf.ToTensor
    overfit_to_index: Optional[int]
    num_target: int
    context_min_distance: int
    context_max_distance: int

    z_near: float = 0.1
    z_far: float = 50.0
    threshold: float = 60.0
    image_size: int = 64
    background_color: torch.tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    avg_key_frame_interval = 10

    def __init__(
        self,
        root: Union[str, Path],
        num_context: int,
        num_target: int,
        context_min_distance: int,
        context_max_distance: int,
        stage: Stage = "train",
        num_intermediate: Optional[int] = 10,
        language_encoder: Optional[T5Encoder] = None,
        overfit_to_index: Optional[int] = None,
        max_scenes: Optional[int] = None,
        pose_root: Optional[Union[str, Path]] = None,
        image_size: Optional[int] = 64,
        adjacent_angle: float = 0.523,
        adjacent_distance: float = 1.0,
        use_depth_supervision: bool = False,
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
        self.use_depth_supervision = use_depth_supervision
        sub_dir = {"train": "train", "test": "test", "unit": "unit"}[stage]
        image_root = Path(root) / sub_dir
        scene_path_list = sorted(list(Path(image_root).glob("*/")))

        if max_scenes is not None:
            scene_path_list = scene_path_list[:max_scenes]

        self.stage = stage
        self.to_tensor = tf.ToTensor()
        self.rng = default_rng()
        self.global_seed = 42
        self.normalize = normalize_to_neg_one_to_one

        if pose_root is None:
            pose_root = root
        
        if self.stage == "unit":
            pose_file = Path(pose_root) / "test.mat"
        else:
            pose_file = Path(pose_root) / f"{stage}.mat"
        self.all_cam_params = scipy.io.loadmat(pose_file)

        dummy_img_path = str(next(scene_path_list[0].glob("*.jpg")))
        dummy_img = cv2.imread(dummy_img_path)
        h, w = dummy_img.shape[:2]

        if language_encoder is not None:
            self.lang_identity = language_encoder("").detach()

        self.len = 0
        all_rgb_files = []
        if self.use_depth_supervision:
            all_depth_files = []
        all_timestamps = []
        self.scene_path_list = []
        for i, scene_path in enumerate(scene_path_list):
            rgb_files = sorted(scene_path.glob("*.jpg"))
            if self.use_depth_supervision:
                depth_files = sorted(scene_path.glob("*.npz"))
                if len(depth_files) == 0:
                    # Skip this scene if no depth file found
                    continue
                depth_file = depth_files[0]
                all_depth_files.append(depth_file)
            self.len += len(rgb_files)
            timestamps = [int(rgb_file.name.split(".")[0]) for rgb_file in rgb_files]
            sorted_ids = np.argsort(timestamps)
            all_rgb_files.append(np.array(rgb_files)[sorted_ids])
            self.scene_path_list.append(scene_path)
            all_timestamps.append(np.array(timestamps)[sorted_ids])
        self.all_rgb_files = np.concatenate(all_rgb_files)
        self.indices = torch.arange(0, len(self.scene_path_list))
        self.all_rgb_files = all_rgb_files
        if self.use_depth_supervision:
            self.all_depth_files = all_depth_files
        self.all_timestamps = all_timestamps
        print("length dataset", self.len)

    def read_image(self, rgb_files, id):
        rgb_file = rgb_files[id]
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

        cam_param = self.all_cam_params[str(rgb_file.parent.name)][id][1:]
        cam_param = Camera(cam_param.flatten().tolist())
        
        return rgb, cam_param
    
    def read_depth(self, depth_file, id):
        depth_data = np.load(depth_file)
        depth = depth_data["depths"][id, :]
        depth = torch.tensor(depth).float()
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            antialias=True,
        )[0]
        return depth

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
            # print(f"Fallback for index {index} in scene {scene_idx}.")
            seed = hash((index + time.time_ns(), self.global_seed)) % (2**32) + time.time_ns() % (2**32)
            self.rng = np.random.default_rng(seed)
            return self[self.rng.integers(0, len(self.all_rgb_files))]

        context_max_distance = self.context_max_distance * self.num_context

        rgb_files = self.all_rgb_files[scene_idx]
        if self.use_depth_supervision:
            depth_file = self.all_depth_files[scene_idx]

        # get a list of frame indices
        if len(rgb_files)<2:
            return fallback()

        # select starting idx
        start_idx = self.rng.integers(1, max(2, len(rgb_files)-context_max_distance*self.avg_key_frame_interval))
        ids = list(range(start_idx, len(rgb_files)))
        first_id = ids.pop(0)
        rgb, cam_param = self.read_image(rgb_files, first_id)
        if self.use_depth_supervision:
            depth = self.read_depth(depth_file, first_id)
            # if depth contains values > self.z_far, fallback
            if (depth > self.z_far).any():
                return fallback()

        rgbs = [rgb]
        if self.use_depth_supervision:
            depths = [depth]
        pose_c2w = [cam_param.c2w_mat]
        intrinsics = [torch.tensor(np.array(cam_param.intrinsics)).float()]
        inv_camera_pose_first = inverse_transformation(torch.tensor(np.array([cam_param.c2w_mat])).float()[0])
        intms_dict = {
            "rgbs": [self.normalize(torch.stack(rgbs, axis=0))]*self.num_intermediate,
            "intrinsics": [torch.stack([torch.tensor(np.array(cam_param.intrinsics)).float()], axis=0)[0]]*self.num_intermediate,
            "pose": [torch.einsum("ijk, ikl -> ijl", inv_camera_pose_first.unsqueeze(0).repeat(1, 1, 1), torch.tensor(np.array([cam_param.c2w_mat])).float())]*self.num_intermediate,
        }
        if self.use_depth_supervision:
            intms_dict["depth"] = [torch.stack(depths, axis=0)]*self.num_intermediate
        intms = [intms_dict]

        intm_keys = ["rgbs", "intrinsics", "pose"]
        if self.use_depth_supervision:
            intm_keys.append("depth")

        z_start = cam_param.c2w_mat[:, 2][:3]  # initial forward vector
        t_start = cam_param.c2w_mat[:, 3][:3]  # initial translation
        z_previous = z_start
        t_previous = t_start
        current_intm = {k: [] for k in intm_keys}

        for id in ids:
            rgb, cam_param = self.read_image(rgb_files, id)
            current_pose = cam_param.c2w_mat
            z_idx = current_pose[:, 2][:3]  # current forward vector
            t_idx = current_pose[:, 3][:3]  # current translation
            angle = rotation_angle(z_previous, z_idx)
            distance = np.linalg.norm(t_idx - t_previous)
            current_intm["rgbs"].append(rgb)
            current_intm["intrinsics"].append(torch.tensor(np.array(cam_param.intrinsics)).float())
            current_intm["pose"].append(cam_param.c2w_mat)
            if self.use_depth_supervision:
                depth = self.read_depth(depth_file, id)
                current_intm["depth"].append(depth)

            if angle > self.adjacent_angle or distance > self.adjacent_distance:
                z_previous = z_idx
                t_previous = t_idx
                rgbs.append(rgb)
                if self.use_depth_supervision:
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
                current_intm["intrinsics"] = [torch.stack([intrinsic], axis=0)[0] for intrinsic in current_intm["intrinsics"]]
                if self.use_depth_supervision:
                    current_intm["depth"] = [torch.stack([depth], axis=0) for depth in current_intm["depth"]]

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
                if self.use_depth_supervision:
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
        if self.use_depth_supervision:
            ret["depth"] = [torch.stack([depth], axis=0) for depth in depths]
        assert len(render_poses)==len(ret["rgbs"])==len(ret["abs_camera_poses"])==len(ret["intrinsics"])

        return (
            ret,
            [torch.stack([rgb], axis=0) for rgb in rgbs],  # rearrange(rendered["image"], "c h w -> (h w) c"),
            len(ids),
        )

    def data_for_video(self, video_idx, ctxt_idx, trgt_idx, num_frames_render=20):
        scene_idx = video_idx
        rgb_files = self.all_rgb_files[scene_idx]
        timestamps = self.all_timestamps[scene_idx]
        assert (timestamps == sorted(timestamps)).all()

        trgt_rgbs = []
        trgt_c2w = []
        trgt_intrinsics = []
        for id in trgt_idx:
            id = min(id, len(rgb_files) - 1)
            id = max(id, 0)
            rgb, cam_param = self.read_image(rgb_files, id)
            trgt_rgbs.append(rgb)
            trgt_intrinsics.append(cam_param.intrinsics)
            trgt_c2w.append(cam_param.c2w_mat)
        trgt_c2w = torch.tensor(np.array(trgt_c2w)).float()
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)

        # load the ctxt
        ctxt_rgbs = []
        ctxt_c2w = []
        ctxt_intrinsics = []
        for id in ctxt_idx:
            id = min(id, len(rgb_files) - 1)
            id = max(id, 0)
            rgb, cam_param = self.read_image(rgb_files, id)
            ctxt_rgbs.append(rgb)
            ctxt_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            ctxt_c2w.append(cam_param.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, axis=0)
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
            # id = ctxt_idx[0] + i * (trgt_idx[0] - ctxt_idx[0]) // (num_frames_render)
            if noflip:
                id = ctxt_idx[0] + i
            else:
                id = trgt_idx[0] + i
            rgb_file = rgb_files[id]
            cam_param = self.all_cam_params[str(rgb_file.parent.name)][id][1:]
            cam_param = Camera(cam_param.flatten().tolist())
            render_poses.append(cam_param.c2w_mat)
        render_poses = torch.tensor(np.array(render_poses)).float()

        print(
            f"ctxt_idx: {ctxt_idx}, trgt_idx: {trgt_idx}, num_frames_render: {num_frames_render}, len(rgb_files): {len(rgb_files)}"
        )

        num_frames_render = render_poses.shape[0]
        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])
        ctxt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            self.num_context, 1, 1
        )
        trgt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            self.num_target, 1, 1
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
        )

    def data_for_temporal(self, video_idx: int, frames_render: Union[int, List]=20):
        scene_idx = video_idx
        rgb_files = self.all_rgb_files[scene_idx]

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
        pose_c2w = []
        intrinsics = []
        for id in ids:
            rgb, cam_param = self.read_image(rgb_files, id)
            rgbs.append(rgb)
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
        
        assert len(render_poses)==len(ret["rgbs"])==len(ret["abs_camera_poses"])==len(ret["intrinsics"])

        return (
            ret,
            [torch.stack([rgb], axis=0) for rgb in rgbs],  # rearrange(rendered["image"], "c h w -> (h w) c"),
            start_idx,
            end_idx,
        )