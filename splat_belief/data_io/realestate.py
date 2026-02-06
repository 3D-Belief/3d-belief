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


Stage = Literal["train", "test",  "unit"]

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

class RealEstate10kDatasetOM(Dataset):
    examples: List[Path]
    pose_file: Path
    stage: Stage
    to_tensor: tf.ToTensor
    overfit_to_index: Optional[int]
    num_target: int
    context_min_distance: int
    context_max_distance: int

    z_near: float = 0.1
    z_far: float = 60.0
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
        pose_root: Optional[Union[str, Path]] = None,
        image_size: Optional[int] = 64,
        adjacent_angle: Optional[float] = np.pi/4,
        use_depth_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.overfit_to_index = overfit_to_index
        self.num_context = num_context
        self.num_target = num_target
        self.context_min_distance = context_min_distance
        self.context_max_distance = context_max_distance
        self.adjacent_angle = adjacent_angle
        self.use_depth_supervision = use_depth_supervision
        self.image_size = image_size
        self.intermediate = intermediate
        self.num_intermediate = num_intermediate
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
        print("NUM IMAGES", self.len)

    def read_image(self, rgb_files, id):
        rgb_file = rgb_files[id]
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
            seed = hash((index, self.global_seed)) % (2**32) + time.time_ns() % (2**32)
            self.rng = np.random.default_rng(seed)
            return self[self.rng.integers(0, len(self.all_rgb_files))]

        rgb_files = self.all_rgb_files[scene_idx]
        if self.use_depth_supervision:
            depth_file = self.all_depth_files[scene_idx]
        timestamps = self.all_timestamps[scene_idx]
        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)
        if num_frames < self.num_target + 1:
            return fallback()

        start_idx = self.rng.choice(max(len(rgb_files)-self.context_max_distance, 2), 1)[0]

        if start_idx+1>=len(rgb_files)-1:
            return fallback()

        # z axis of in pose pf the start idx
        _, cam_param = self.read_image(rgb_files, start_idx)
        pose_ctxt = cam_param.c2w_mat
        z_start = pose_ctxt[:, 2][:3]

        for idx in range(start_idx+1, num_frames):
            _, cam_param = self.read_image(rgb_files, idx)
            pose_idx = cam_param.c2w_mat
            z_idx = pose_idx[:, 2][:3]
            angle = rotation_angle(z_start, z_idx)
            end_idx = idx
            if angle > self.adjacent_angle or idx-start_idx > self.context_max_distance:
                end_idx = idx
                break
        
        # print("scene_idx")
        # print(scene_idx)
        # print("start_idx")
        # print(start_idx)
        # print("end_idx")
        # print(end_idx)

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
        if self.use_depth_supervision:
            trgt_depths = []
        trgt_c2w = []
        trgt_intrinsics = []
        for id in trgt_idx:
            rgb, cam_param = self.read_image(rgb_files, id)
            if self.use_depth_supervision:
                depth = self.read_depth(depth_file, id)
            trgt_rgbs.append(rgb)
            if self.use_depth_supervision:
                trgt_depths.append(depth)
            trgt_intrinsics.append(cam_param.intrinsics)
            trgt_c2w.append(cam_param.c2w_mat)
        trgt_c2w = torch.tensor(np.array(trgt_c2w)).float()
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)
        if self.use_depth_supervision:
            trgt_depth = torch.stack(trgt_depths, axis=0)

        # load the ctxt
        ctxt_rgbs = []
        if self.use_depth_supervision:
            ctxt_depths = []
        ctxt_c2w = []
        ctxt_intrinsics = []
        for id in ctxt_idx:
            rgb, cam_param = self.read_image(rgb_files, id)
            if self.use_depth_supervision:
                depth = self.read_depth(depth_file, id)
            ctxt_rgbs.append(rgb)
            if self.use_depth_supervision:
                ctxt_depths.append(depth)
            ctxt_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            ctxt_c2w.append(cam_param.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, axis=0)
        if self.use_depth_supervision:
            ctxt_depth = torch.stack(ctxt_depths, axis=0)
        ctxt_intrinsics = torch.stack(ctxt_intrinsics, axis=0)

        # load the intermediate
        if self.intermediate and (intm_idx is not None):
            intm_rgbs = []
            if self.use_depth_supervision:
                intm_depths = []
            intm_c2w = []
            intm_intrinsics = []
            for id in intm_idx:
                rgb, cam_param = self.read_image(rgb_files, id)
                if self.use_depth_supervision:
                    depth = self.read_depth(depth_file, id)
                intm_rgbs.append(rgb)
                if self.use_depth_supervision:
                    intm_depths.append(depth)
                intm_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
                intm_c2w.append(cam_param.c2w_mat)
            intm_c2w = torch.tensor(np.array(intm_c2w)).float()
            intm_rgb = torch.stack(intm_rgbs, axis=0)
            if self.use_depth_supervision:
                intm_depth = torch.stack(intm_depths, axis=0)
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
            })
        if self.intermediate and (intm_idx is not None):
            ret_dict.update({
                "intm_c2w": torch.einsum(
                    "ijk, ikl -> ijl", intm_inv_ctxt_c2w_repeat, intm_c2w
                ),
                "intm_rgb": self.normalize(intm_rgb),
                "intm_abs_camera_poses": intm_c2w,
            })
            if self.use_depth_supervision:
                ret_dict.update({
                    "intm_depth": intm_depth,
                })
        ret = (
            ret_dict,
            trgt_rgb,  # rearrange(rendered["image"], "c h w -> (h w) c"),
        )
        return ret

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