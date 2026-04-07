"""
ProcTHOR dataset for 3D-belief with scene graph conditioning.
Loads RGB (from mp4), depth (npz), poses (npz), and scene graphs (JSON).
"""
from pathlib import Path
from typing import List, Optional, Union

import cv2
import json
import numpy as np
import time
import torch
import torch.nn.functional as F
from numpy.random import default_rng
from torch.utils.data import Dataset

from splat_belief.utils.vision_utils import normalize_to_neg_one_to_one
from splat_belief.utils.procthor_utils import (
    load_vocabulary,
    parse_scene_graph,
    collect_room_bounds,
    NODE_TYPE_PAD,
)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

Stage = Literal["train", "unit"]


class Camera:
    """Camera parameters matching SPOC/ProcTHOR convention."""
    def __init__(self, extrinsics: np.ndarray):
        fx, fy, cx, cy = 0.390, 0.385, 0.5, 0.5
        self.intrinsics = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32,
        )
        self.w2c_mat = extrinsics
        self.c2w_mat = np.linalg.inv(extrinsics)


def _rotation_angle(z1: np.ndarray, z2: np.ndarray) -> float:
    """Angle between two direction vectors (radians)."""
    cos = np.clip(np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2) + 1e-8), -1, 1)
    return np.arccos(cos)


class ProcTHORDataset(Dataset):
    """
    ProcTHOR POC dataset with scene graph conditioning.

    Each episode directory contains:
        - rgb_trajectory.mp4
        - all_depths.npz  (key: 'depths', shape [N, 240, 240])
        - all_poses.npz   (key: 'poses', shape [N, 4, 4])
        - all_scene_graphs.json  (list of N frame dicts)
        - trajectory_metadata.json
    """

    z_near: float = 0.01
    z_far: float = 50.0
    z_filter: float = 19.0

    def __init__(
        self,
        root: Union[str, Path],
        vocab_dir: str,
        num_context: int = 1,
        num_target: int = 1,
        context_min_distance: int = 1,
        context_max_distance: int = 30,
        stage: Stage = "train",
        image_size: int = 64,
        adjacent_angle: float = np.pi / 4,
        adjacent_distance: float = 1.0,
        use_depth_supervision: bool = True,
        intermediate: bool = False,
        num_intermediate: Optional[int] = 3,
        max_scenes: Optional[int] = None,
        overfit_to_index: Optional[int] = None,
        near_threshold: float = 2.0,
        max_nodes: int = 128,
        max_edges: int = 512,
        include_walls: bool = False,
        wall_height_default: float = 2.5,
        wall_thickness: float = 0.15,
        pose_source: str = "gt",
        predicted_poses_filename: str = "predicted_poses.npz",
    ) -> None:
        super().__init__()
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
        self.overfit_to_index = overfit_to_index
        self.stage = stage
        self.near_threshold = near_threshold
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.include_walls = include_walls
        self.wall_height_default = wall_height_default
        self.wall_thickness = wall_thickness
        self.pose_source = pose_source
        self.predicted_poses_filename = predicted_poses_filename

        self.normalize = normalize_to_neg_one_to_one
        self.rng = default_rng()
        self.global_seed = 42

        # Load vocabulary
        self.type_to_id, self.id_to_type = load_vocabulary(vocab_dir)

        # Discover episodes
        image_root = Path(root) / self.stage
        scene_path_list = sorted([p for p in image_root.glob("*/") if p.is_dir()])
        if max_scenes is not None:
            scene_path_list = scene_path_list[:max_scenes]

        self.scene_path_list = []
        self.num_frames_per_scene = []

        for scene_path in scene_path_list:
            rgb_file = scene_path / "rgb_trajectory.mp4"
            depth_file = scene_path / "all_depths.npz"
            pose_file = scene_path / "all_poses.npz"
            sg_file = scene_path / "all_scene_graphs.json"

            required = rgb_file.exists() and depth_file.exists() and pose_file.exists() and sg_file.exists()
            if self.pose_source == "predicted":
                pred_pose_file = scene_path / self.predicted_poses_filename
                required = required and pred_pose_file.exists()

            if required:
                poses_data = np.load(pose_file)
                num_frames = poses_data["poses"].shape[0]
                self.num_frames_per_scene.append(num_frames)
                self.scene_path_list.append(scene_path)

        print(f"[ProcTHOR] {len(self.scene_path_list)} scenes (pose_source={self.pose_source}), vocab size {len(self.id_to_type)}")

    # -------------------------------------------------------------------------
    # Frame reading (mirrors SPOC pattern)
    # -------------------------------------------------------------------------

    def read_frame(self, scene_path: Path, frame_id: int):
        """Read RGB, depth, depth_mask, and camera for a single frame."""
        # RGB from video
        rgb_file = scene_path / "rgb_trajectory.mp4"
        cap = cv2.VideoCapture(str(rgb_file))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_id} from {rgb_file}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = torch.tensor(frame.astype(np.float32)).permute(2, 0, 1) / 255.0
        rgb = F.interpolate(
            rgb.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            antialias=True,
        )[0]

        # Depth from npz
        depth_file = scene_path / "all_depths.npz"
        depths_data = np.load(depth_file)
        depth_val = depths_data["depths"][frame_id]
        depth = torch.tensor(depth_val, dtype=torch.float32).unsqueeze(0)
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            antialias=True,
        )[0]
        depth_mask = (depth < self.z_filter)

        # Pose from npz
        if self.pose_source == "predicted":
            pred_file = scene_path / self.predicted_poses_filename
            pred_data = np.load(pred_file)
            # Predicted poses are already c2w in 3d-belief convention
            c2w_3dbelief = pred_data["poses"][frame_id]
            w2c_3dbelief = np.linalg.inv(c2w_3dbelief)
            cam_param = Camera(w2c_3dbelief.astype(np.float32))
            if "intrinsics" in pred_data:
                cam_param.intrinsics = pred_data["intrinsics"][frame_id].astype(np.float32)
        else:
            pose_file = scene_path / "all_poses.npz"
            poses_data = np.load(pose_file)
            extrinsics = poses_data["poses"][frame_id]  # c2w
            extrinsics = np.linalg.inv(extrinsics)       # w2c
            conversion = np.diag([1, -1, -1, 1])
            extrinsics = conversion @ extrinsics
            cam_param = Camera(extrinsics)
        return rgb, depth, depth_mask, cam_param

    def read_scene_graph(self, scene_path: Path, frame_id: int) -> dict:
        """Read and parse scene graph for a single frame."""
        sg_file = scene_path / "all_scene_graphs.json"
        with open(sg_file) as f:
            sg_data = json.load(f)
        frame_data = sg_data[frame_id]
        # Pre-compute room bounds across the full episode for door inference
        room_bounds = collect_room_bounds(sg_data) if self.include_walls else None
        return parse_scene_graph(
            frame_data,
            self.type_to_id,
            near_threshold=self.near_threshold,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
            include_walls=self.include_walls,
            wall_height_default=self.wall_height_default,
            wall_thickness=self.wall_thickness,
            room_bounds=room_bounds,
        )

    # -------------------------------------------------------------------------
    # Dataset interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.scene_path_list)

    def __getitem__(self, index: int):
        seed = hash((index, self.global_seed)) % (2**32) + time.time_ns() % (2**32)
        self.rng = np.random.default_rng(seed)
        scene_idx = self.rng.integers(0, len(self.scene_path_list))
        if self.overfit_to_index is not None:
            scene_idx = self.overfit_to_index

        def fallback():
            seed = hash((index, self.global_seed)) % (2**32) + time.time_ns() % (2**32)
            self.rng = np.random.default_rng(seed)
            return self[self.rng.integers(0, len(self.scene_path_list))]

        scene_path = self.scene_path_list[scene_idx]
        num_frames = self.num_frames_per_scene[scene_idx]

        if num_frames < 2:
            return fallback()

        # Pick context frame
        start_idx = self.rng.choice(max(num_frames - self.context_max_distance, 2), 1)[0]
        if start_idx + 1 >= num_frames - 1:
            return fallback()

        # Find target frame with sufficient viewpoint change (same logic as SPOC)
        _, _, _, cam_param = self.read_frame(scene_path, start_idx)
        pose_ctxt = cam_param.c2w_mat
        z_start = pose_ctxt[:, 2][:3]
        t_start = pose_ctxt[:, 3][:3]

        end_idx = start_idx + 1
        for idx in range(start_idx + 1, num_frames):
            _, _, _, cam_param = self.read_frame(scene_path, idx)
            pose_idx = cam_param.c2w_mat
            z_idx = pose_idx[:, 2][:3]
            t_idx = pose_idx[:, 3][:3]
            angle = _rotation_angle(z_start, z_idx)
            dist = np.linalg.norm(t_idx - t_start)
            end_idx = idx
            if (
                angle > self.adjacent_angle
                or dist > self.adjacent_distance
                or idx - start_idx > self.context_max_distance
            ):
                break

        # Random flip context/target
        if self.rng.choice([True, False]):
            start_idx, end_idx = end_idx, start_idx

        ctxt_idx = [start_idx]
        trgt_idx = [end_idx]

        # intermediate frames
        if self.intermediate:
            start = min(trgt_idx[0], ctxt_idx[0])
            end = max(trgt_idx[0], ctxt_idx[0])
            available_choices = np.arange(start, end)
            intm_idx = self.rng.choice(available_choices, self.num_intermediate, replace=True)

        # Use the target frame's scene graph (represents what we want to generate)
        # Could also use a union — for now use the max(ctxt, trgt) frame since later frames see more
        sg_frame_id = max(ctxt_idx[0], trgt_idx[0])

        # Load target views
        trgt_rgbs, trgt_depths, trgt_depth_masks, trgt_c2w_list = [], [], [], []
        trgt_intrinsics_list = []
        for fid in trgt_idx:
            rgb, depth, depth_mask, cam = self.read_frame(scene_path, fid)
            trgt_rgbs.append(rgb)
            trgt_depths.append(depth)
            trgt_depth_masks.append(depth_mask)
            trgt_intrinsics_list.append(cam.intrinsics)
            trgt_c2w_list.append(cam.c2w_mat)
        trgt_c2w = torch.tensor(np.array(trgt_c2w_list)).float()
        trgt_rgb = torch.stack(trgt_rgbs, dim=0)
        trgt_depth = torch.stack(trgt_depths, dim=0)
        trgt_depth_mask = torch.stack(trgt_depth_masks, dim=0)

        # Load context views
        ctxt_rgbs, ctxt_depths, ctxt_depth_masks, ctxt_c2w_list = [], [], [], []
        ctxt_intrinsics_list = []
        for fid in ctxt_idx:
            rgb, depth, depth_mask, cam = self.read_frame(scene_path, fid)
            ctxt_rgbs.append(rgb)
            ctxt_depths.append(depth)
            ctxt_depth_masks.append(depth_mask)
            ctxt_intrinsics_list.append(torch.tensor(cam.intrinsics).float())
            ctxt_c2w_list.append(cam.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w_list)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, dim=0)
        ctxt_depth = torch.stack(ctxt_depths, dim=0)
        ctxt_depth_mask = torch.stack(ctxt_depth_masks, dim=0)
        ctxt_intrinsics = torch.stack(ctxt_intrinsics_list, dim=0)

        # Load intermediate views
        if self.intermediate and (intm_idx is not None):
            intm_rgbs, intm_depths, intm_depth_masks, intm_c2w_list = [], [], [], []
            intm_intrinsics_list = []
            for fid in intm_idx:
                rgb, depth, depth_mask, cam = self.read_frame(scene_path, fid)
                intm_rgbs.append(rgb)
                intm_depths.append(depth)
                intm_depth_masks.append(depth_mask)
                intm_intrinsics_list.append(torch.tensor(cam.intrinsics).float())
                intm_c2w_list.append(cam.c2w_mat)
            intm_c2w = torch.tensor(np.array(intm_c2w_list)).float()
            intm_rgb = torch.stack(intm_rgbs, dim=0)
            intm_depth = torch.stack(intm_depths, dim=0)
            intm_depth_mask = torch.stack(intm_depth_masks, dim=0)
            intm_intrinsics = torch.stack(intm_intrinsics_list, dim=0)

        # Normalize poses relative to first context frame
        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])
        ctxt_inv = inv_ctxt_c2w.unsqueeze(0).repeat(len(ctxt_idx), 1, 1)
        trgt_inv = inv_ctxt_c2w.unsqueeze(0).repeat(len(trgt_idx), 1, 1)
        if self.intermediate and (intm_idx is not None):
            intm_inv = inv_ctxt_c2w.unsqueeze(0).repeat(self.num_intermediate, 1, 1)

        # Parse scene graph
        sg_tensors = self.read_scene_graph(scene_path, sg_frame_id)

        ret_dict = {
            "ctxt_c2w": torch.einsum("ijk, ikl -> ijl", ctxt_inv, ctxt_c2w),
            "trgt_c2w": torch.einsum("ijk, ikl -> ijl", trgt_inv, trgt_c2w),
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
            # Scene graph tensors
            **sg_tensors,
        }

        if self.use_depth_supervision:
            ret_dict.update({
                "ctxt_depth": ctxt_depth,
                "trgt_depth": trgt_depth,
                "ctxt_depth_mask": ctxt_depth_mask,
                "trgt_depth_mask": trgt_depth_mask,
            })
        if self.intermediate and (intm_idx is not None):
            ret_dict.update({
                "intm_c2w": torch.einsum("ijk, ikl -> ijl", intm_inv, intm_c2w),
                "intm_rgb": self.normalize(intm_rgb),
                "intm_abs_camera_poses": intm_c2w,
            })
            if self.use_depth_supervision:
                ret_dict.update({
                    "intm_depth": intm_depth,
                    "intm_depth_mask": intm_depth_mask,
                })

        return (ret_dict, trgt_rgb)
