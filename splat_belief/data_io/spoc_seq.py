from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from torch.utils.data import Dataset
from numpy.random import default_rng
import cv2

from splat_belief.utils.vision_utils import *
from splat_belief.splat.layers import T5Encoder

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

Stage = Literal["train", "test", "unit", "one", "one_test"]


class Camera(object):
    def __init__(self, extrinsics):
        fx, fy, cx, cy = 0.390, 0.385, 0.5, 0.5
        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.w2c_mat = extrinsics
        self.c2w_mat = np.linalg.inv(extrinsics)


class SPOCDatasetSeq(Dataset):
    """
    Seq dataset corresponding to your video-backed SPOCDataset (un-seq).

    - per scene: videos/rgb*.mp4, videos/depth*.mp4, pose/*.npy
    - frame index is pose index (0..len(pose_files)-1)
    - keyframes chosen by (angle > adjacent_angle) OR (translation > adjacent_distance)
    - each keyframe has num_intermediate sampled frames from the segment since last keyframe
    """
    z_near: float = 0.01
    z_far: float = 50.0
    z_filter: float = 19.0
    image_size: int = 64
    background_color: torch.tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    avg_key_frame_interval = 8  # heuristic, same spirit as your old seq version

    def __init__(
        self,
        root: Union[str, Path],
        num_context: int,
        num_target: int,  # kept for signature consistency, not used (seq uses fixed-length keyframe list)
        context_min_distance: int,
        context_max_distance: int,
        stage: Stage = "train",
        num_intermediate: Optional[int] = 3,
        language_encoder: Optional[T5Encoder] = None,
        overfit_to_index: Optional[int] = None,
        max_scenes: Optional[int] = None,
        image_size: Optional[int] = 64,
        adjacent_angle: float = 0.523,      # ~30 deg
        adjacent_distance: float = 1.0,     # meters (pose translation space)
        use_depth_supervision: bool = True,
        depth_scale: float = 1000.0,        # your un-seq uses /1000.0 (mm->m)
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
        self.depth_scale = depth_scale

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
        else:
            # fall back to a dummy zero vector if you ever call without lang encoder
            self.lang_identity = torch.zeros(1)

        self.len = 0
        all_rgb_files = []
        all_depth_files = []
        all_pose_files = []
        self.scene_path_list = []

        for scene_path in scene_path_list:
            rgb_files = sorted(scene_path.glob("videos/rgb*.mp4"))
            depth_files = sorted(scene_path.glob("videos/depth*.mp4"))
            pose_files = sorted(scene_path.glob("pose/*.npy"))

            # same indexing assumption as your un-seq: pose index == frame index
            self.len += len(pose_files)

            # sort pose files by timestamp embedded in filename (your un-seq does [-2])
            timestamps = [int(p.name.split(".")[0].split("_")[-2]) for p in pose_files]
            sorted_ids = np.argsort(timestamps)

            all_rgb_files.append(np.array(rgb_files))            # you use rgb_files[0]
            all_depth_files.append(np.array(depth_files))        # you use depth_files[0]
            all_pose_files.append(np.array(pose_files)[sorted_ids])
            self.scene_path_list.append(scene_path)

        self.indices = torch.arange(0, len(self.scene_path_list))
        self.all_rgb_files = all_rgb_files
        self.all_depth_files = all_depth_files
        self.all_pose_files = all_pose_files

        print("[SPOC-SEQ] Scenes", len(self.scene_path_list))
        print("[SPOC-SEQ] length dataset (pose frames)", self.len)

    def __len__(self) -> int:
        # NOTE: matches your current behavior (num scenes). If you want per-frame length, use self.len.
        return len(self.all_rgb_files)

    # ----------------------------
    # Video capture caching helpers
    # ----------------------------
    @lru_cache(maxsize=128)
    def _get_caps(self, rgb_path: str, depth_path: str):
        cap_rgb = cv2.VideoCapture(rgb_path)
        cap_depth = cv2.VideoCapture(depth_path)
        if not cap_rgb.isOpened():
            raise RuntimeError(f"Failed to open RGB video: {rgb_path}")
        if not cap_depth.isOpened():
            raise RuntimeError(f"Failed to open depth video: {depth_path}")
        return cap_rgb, cap_depth

    def _read_rgb_depth_from_video(self, rgb_files, depth_files, frame_id: int):
        rgb_path = str(rgb_files[0])
        depth_path = str(depth_files[0])
        cap_rgb, cap_depth = self._get_caps(rgb_path, depth_path)

        # RGB
        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap_rgb.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read RGB frame {frame_id} from {rgb_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = torch.tensor(frame.astype(np.float32)).permute(2, 0, 1) / 255.0
        rgb = F.interpolate(
            rgb.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            antialias=True,
        )[0]

        # Depth (as stored in mp4; you used /1000 and then took channel 0)
        cap_depth.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, dframe = cap_depth.read()
        if not ok or dframe is None:
            raise RuntimeError(f"Failed to read depth frame {frame_id} from {depth_path}")
        dframe = dframe.astype(np.float32) / float(self.depth_scale)  # mm->m if scale=1000
        # take first channel as you did
        depth = torch.tensor(dframe[..., 0], dtype=torch.float32).unsqueeze(0)
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            antialias=True,
        )[0]

        depth_mask = depth < self.z_filter
        return rgb, depth, depth_mask

    def read_frame(self, rgb_files, depth_files, pose_files, frame_id: int):
        rgb, depth, depth_mask = self._read_rgb_depth_from_video(rgb_files, depth_files, frame_id)

        pose_file = pose_files[frame_id]
        extrinsics = np.load(pose_file)
        extrinsics = np.linalg.inv(extrinsics)
        conversion = np.diag([1, -1, -1, 1])
        extrinsics = conversion @ extrinsics
        cam_param = Camera(extrinsics)

        return rgb, depth, depth_mask, cam_param

    def __getitem__(self, index: int):
        seed = hash((index, self.global_seed)) % (2**32) + time.time_ns() % (2**32)
        self.rng = np.random.default_rng(seed)

        scene_idx = self.rng.integers(0, len(self.all_rgb_files))
        if self.overfit_to_index is not None:
            scene_idx = self.overfit_to_index

        def fallback():
            seed2 = hash((index, self.global_seed)) % (2**32) + time.time_ns() % (2**32)
            self.rng = np.random.default_rng(seed2)
            return self[self.rng.integers(0, len(self.all_rgb_files))]

        rgb_files = self.all_rgb_files[scene_idx]
        depth_files = self.all_depth_files[scene_idx]
        pose_files = self.all_pose_files[scene_idx]
        num_frames = len(pose_files)
        if num_frames < 2:
            return fallback()

        # how many keyframes we want (same pattern as your old seq version)
        # NOTE: your old code used: context_max_distance = self.context_max_distance * self.num_context
        num_keyframes = self.context_max_distance * self.num_context

        # start index heuristic (avoid going too close to end)
        max_start = max(2, num_frames - num_keyframes * self.avg_key_frame_interval)
        start_idx = int(self.rng.integers(1, max_start))

        ids = list(range(start_idx, num_frames))
        first_id = ids.pop(0)

        # read first keyframe
        rgb0, depth0, depth_mask0, cam0 = self.read_frame(rgb_files, depth_files, pose_files, first_id)

        rgbs = [rgb0]
        depths = [depth0]
        depth_masks = [depth_mask0]
        pose_c2w = [cam0.c2w_mat]
        intrinsics = [torch.tensor(np.array(cam0.intrinsics)).float()]

        inv_pose_first = inverse_transformation(torch.tensor(np.array([cam0.c2w_mat])).float()[0])

        # intms[0] mirrors your old logic: seed with repeats of the first frame
        intm_keys = ["rgbs", "depth", "depth_mask", "intrinsics", "pose"]
        intms = [{
            "rgbs": [self.normalize(torch.stack([rgb0], axis=0))] * self.num_intermediate,
            "depth": [torch.stack([depth0], axis=0)] * self.num_intermediate,
            "depth_mask": [torch.stack([depth_mask0], axis=0)] * self.num_intermediate,
            "intrinsics": [torch.stack([torch.tensor(np.array(cam0.intrinsics)).float()], axis=0)[0]] * self.num_intermediate,
            "pose": [torch.einsum(
                "ijk, ikl -> ijl",
                inv_pose_first.unsqueeze(0).repeat(1, 1, 1),
                torch.tensor(np.array([cam0.c2w_mat])).float()
            )] * self.num_intermediate,
        }]

        z_previous = cam0.c2w_mat[:, 2][:3]
        t_previous = cam0.c2w_mat[:, 3][:3]

        current_intm = {k: [] for k in intm_keys}

        for fid in ids:
            rgb, depth, depth_mask, cam = self.read_frame(rgb_files, depth_files, pose_files, fid)

            current_pose = cam.c2w_mat
            z_idx = current_pose[:, 2][:3]
            t_idx = current_pose[:, 3][:3]

            angle = rotation_angle(z_previous, z_idx)
            dist = np.linalg.norm(t_idx - t_previous)

            # accumulate potential intermediates
            current_intm["rgbs"].append(rgb)
            current_intm["depth"].append(depth)
            current_intm["depth_mask"].append(depth_mask)
            current_intm["intrinsics"].append(torch.tensor(np.array(cam.intrinsics)).float())
            current_intm["pose"].append(cam.c2w_mat)

            # commit a new keyframe if we moved enough
            if angle > self.adjacent_angle or dist > self.adjacent_distance:
                z_previous = z_idx
                t_previous = t_idx

                rgbs.append(rgb)
                depths.append(depth)
                depth_masks.append(depth_mask)
                intrinsics.append(torch.tensor(np.array(cam.intrinsics)).float())
                pose_c2w.append(cam.c2w_mat)

                # sample num_intermediate from current_intm
                if len(current_intm["rgbs"]) == 0:
                    # degenerate: no candidates, repeat current keyframe
                    chosen = [0] * self.num_intermediate
                    tmp_rgbs = [rgb] * self.num_intermediate
                    tmp_depths = [depth] * self.num_intermediate
                    tmp_masks = [depth_mask] * self.num_intermediate
                    tmp_intr = [torch.tensor(np.array(cam.intrinsics)).float()] * self.num_intermediate
                    tmp_pose = [cam.c2w_mat] * self.num_intermediate
                else:
                    chosen = self.rng.choice(len(current_intm["rgbs"]), size=self.num_intermediate, replace=True)
                    tmp_rgbs = [current_intm["rgbs"][i] for i in chosen]
                    tmp_depths = [current_intm["depth"][i] for i in chosen]
                    tmp_masks = [current_intm["depth_mask"][i] for i in chosen]
                    tmp_intr = [current_intm["intrinsics"][i] for i in chosen]
                    tmp_pose = [current_intm["pose"][i] for i in chosen]

                # normalize / shape them and convert pose to relative-to-first
                tmp_rgbs = [self.normalize(torch.stack([x], axis=0)) for x in tmp_rgbs]
                tmp_depths = [torch.stack([x], axis=0) for x in tmp_depths]
                tmp_masks = [torch.stack([x], axis=0) for x in tmp_masks]
                tmp_intr = [torch.stack([x], axis=0)[0] for x in tmp_intr]

                abs_pose = [torch.tensor(np.array([p])).float() for p in tmp_pose]
                inv_repeat = [inv_pose_first.unsqueeze(0).repeat(1, 1, 1) for _ in abs_pose]
                rel_pose = [torch.einsum("ijk, ikl -> ijl", invr, ap) for invr, ap in zip(inv_repeat, abs_pose)]

                intms.append({
                    "rgbs": tmp_rgbs,
                    "depth": tmp_depths,
                    "depth_mask": tmp_masks,
                    "intrinsics": tmp_intr,
                    "pose": rel_pose,
                })

                current_intm = {k: [] for k in intm_keys}

            if len(rgbs) >= num_keyframes:
                break

        # cycle padding to fixed length
        cur_len = len(rgbs)
        if cur_len < num_keyframes:
            pad_count = num_keyframes - cur_len
            n = cur_len
            if n > 1:
                pad_indices = list(range(n - 2, -1, -1)) + list(range(1, n))
            else:
                pad_indices = [0]
            for i in range(pad_count):
                j = pad_indices[i % len(pad_indices)]
                rgbs.append(rgbs[j])
                depths.append(depths[j])
                depth_masks.append(depth_masks[j])
                intrinsics.append(intrinsics[j])
                pose_c2w.append(pose_c2w[j])
                intms.append(intms[j])

        assert len(rgbs) == num_keyframes
        assert len(pose_c2w) == num_keyframes
        assert len(intms) == num_keyframes

        # build render_poses relative to first keyframe
        abs_camera_poses = [torch.tensor(np.array([c2w])).float() for c2w in pose_c2w]
        inv_camera_poses = [inverse_transformation(c2w[0]) for c2w in abs_camera_poses]
        inv0 = inv_camera_poses[0]
        inv0_repeat = [inv0.unsqueeze(0).repeat(1, 1, 1) for _ in abs_camera_poses]
        render_poses = [torch.einsum("ijk, ikl -> ijl", invr, ap) for invr, ap in zip(inv0_repeat, abs_camera_poses)]

        ret = {
            "render_poses": render_poses,
            "rgbs": [self.normalize(torch.stack([rgb], axis=0)) for rgb in rgbs],
            "abs_camera_poses": abs_camera_poses,
            "intrinsics": [torch.stack([intr], axis=0)[0] for intr in intrinsics],
            "near": self.z_near,
            "far": self.z_far,
            "idx": torch.tensor([index]),
            "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
            "lang": self.lang_identity,
            "intms": intms,
        }

        if self.use_depth_supervision:
            ret.update({
                "depth": [torch.stack([d], axis=0) for d in depths],
                "depth_mask": [torch.stack([m], axis=0) for m in depth_masks],
            })

        assert len(ret["render_poses"]) == len(ret["rgbs"]) == len(ret["abs_camera_poses"]) == len(ret["intrinsics"])

        # match your old seq return structure
        return (
            ret,
            [torch.stack([rgb], axis=0) for rgb in rgbs],
            len(ids),  # how many raw frames were scanned from start_idx to stop
        )

    def data_for_temporal(self, video_idx: int, frames_render: Union[int, List] = 20):
        scene_idx = video_idx
        rgb_files = self.all_rgb_files[scene_idx]
        depth_files = self.all_depth_files[scene_idx]
        pose_files = self.all_pose_files[scene_idx]

        num_frames = len(pose_files)
        if num_frames < 2:
            raise ValueError(f"Scene {scene_idx} has too few frames: {num_frames}")

        # get a list of frame indices
        if isinstance(frames_render, (int, np.integer)):
            ids = select_random_sequence(self.rng, num_frames, int(frames_render))
            start_idx = int(ids[0])
            end_idx = int(ids[-1])
        else:
            start_idx = int(frames_render[0])
            end_idx = int(frames_render[1])
            assert 0 <= start_idx < num_frames and 0 <= end_idx < num_frames
            ids = np.arange(start_idx, end_idx + 1)

        rgbs = []
        depths = []
        depth_masks = []
        pose_c2w = []
        intrinsics = []

        for fid in ids:
            rgb, depth, depth_mask, cam_param = self.read_frame(rgb_files, depth_files, pose_files, int(fid))
            rgbs.append(rgb)
            if self.use_depth_supervision:
                depths.append(depth)
                depth_masks.append(depth_mask)
            intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            pose_c2w.append(cam_param.c2w_mat)

        print(f"start_idx: {ids[0]}, end_idx: {ids[-1]}, num_frames_render: {len(ids)}")

        abs_camera_poses = [torch.tensor(np.array([c2w])).float() for c2w in pose_c2w]

        # render_poses: each pose expressed in the coordinate frame of the first pose
        inv_camera_poses = [inverse_transformation(c2w[0]) for c2w in abs_camera_poses]
        inv0 = inv_camera_poses[0]
        inv0_repeat = [inv0.unsqueeze(0).repeat(1, 1, 1) for _ in abs_camera_poses]
        render_poses = [
            torch.einsum("ijk, ikl -> ijl", invr, ap)
            for invr, ap in zip(inv0_repeat, abs_camera_poses)
        ]

        ret = {
            "render_poses": render_poses,
            "rgbs": [self.normalize(torch.stack([rgb], axis=0)) for rgb in rgbs],
            "abs_camera_poses": abs_camera_poses,
            "intrinsics": [torch.stack([intr], axis=0)[0] for intr in intrinsics],
            "near": self.z_near,
            "far": self.z_far,
            "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
            "lang": self.lang_identity,
        }

        if self.use_depth_supervision:
            ret.update({
                "depth": [torch.stack([d], axis=0) for d in depths],
                "depth_mask": [torch.stack([m], axis=0) for m in depth_masks],
            })

        assert len(render_poses) == len(ret["rgbs"]) == len(ret["abs_camera_poses"]) == len(ret["intrinsics"])
        if self.use_depth_supervision:
            assert len(ret["depth"]) == len(ret["depth_mask"]) == len(ret["rgbs"])

        return (
            ret,
            [torch.stack([rgb], axis=0) for rgb in rgbs],  # raw (unnormalized) rgbs
            start_idx,
            end_idx,
        )