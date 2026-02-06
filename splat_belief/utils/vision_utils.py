import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from typing import Optional, Callable, Any
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import torch
import copy
import json
import colorsys
import numbers
import os
import numpy as np
import cv2
from einops import rearrange
import random
from typing import Union


def normalize(a):
    return (a - a.min()) / (a.max() - a.min())


def jet_depth(depth):
    # depth (B, H, W)
    # normalize depth to [0,1]
    depth = normalize(depth)
    depth = plt.cm.jet(depth.numpy())[..., :3]  # (B, H, W, 3)
    return depth


def put_optical_flow_arrows_on_image(
    image, optical_flow_image, threshold=1.0, skip_amount=30
):
    # Don't affect original image
    # image = image.copy()

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(
        np.meshgrid(
            range(optical_flow_image.shape[1]), range(optical_flow_image.shape[0])
        ),
        2,
    )
    flow_end = (
        optical_flow_image[flow_start[:, :, 1], flow_start[:, :, 0], :1] * 3
        + flow_start
    ).astype(np.int32)

    # Threshold values
    norm = np.linalg.norm(flow_end - flow_start, axis=2)
    norm[norm < threshold] = 0

    # Draw all the nonzero values
    nz = np.nonzero(norm)
    for i in range(0, len(nz[0]), skip_amount):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(
            image,
            pt1=tuple(flow_start[y, x]),
            pt2=tuple(flow_end[y, x]),
            color=(0, 255, 0),
            thickness=1,
            tipLength=0.2,
        )
    return image


def trans_t(t):
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1],], dtype=torch.float32,
    )


def rot_phi(phi):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rot_theta(th):
    return torch.tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def pose_spherical(theta, phi, radius):
    """
    Spherical rendering poses, from NeRF
    """
    c2w = trans_t(-radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    # c2w[2, -1] += radius
    return c2w


def render_spherical(model_input, model, resolution, n):
    radius = (1.2 + 4.0) * 0.5

    # Use 360 pose sequence from NeRF
    render_poses = torch.stack(
        [
            torch.einsum(
                "ij, jk -> ik",
                model_input["ctxt_c2w"][0].cpu(),
                pose_spherical(angle, -0.0, radius).cpu(),
            )
            # pose_spherical(angle, -10., radius)
            for angle in np.linspace(-180, 180, n + 1)[:-1]
        ],
        0,
    )  # (NV, 4, 4)

    # torch.set_printoptions(precision=2)
    frames = []

    # print(model_input.keys())
    for k in ["x_pix", "intrinsics", "ctxt_rgb", "ctxt_c2w", "idx", "z_near", "z_far"]:
        if k in model_input:
            model_input[k] = model_input[k][:1]

    for i in range(n):
        model_input["trgt_c2w"] = render_poses[i : i + 1].cuda()

        with torch.no_grad():
            rgb_pred, depth_pred, _ = model(model_input)

        rgb_pred = (
            rgb_pred.cpu().view(*(1, resolution[1], resolution[2], 3)).detach().numpy()
        )
        frames.append(rgb_pred)
    return frames


def exists(x):
    return x is not None


def get_constant_hyperparameter_schedule_with_warmup(
    num_warmup_steps: int, last_epoch: int = -1
):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.
    Args:
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return lr_lambda


def get_constant_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1
):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# def normalize_to_neg_one_to_one(img):
#     return torch.clamp(img * 2 - 1, -1.0, 1.0)

# def unnormalize_to_zero_to_one(t):
#     return torch.clamp((t + 1) * 0.5, 0.0, 1.0)

def to_gpu(ob, device):
    if isinstance(ob, dict):
        return {k: to_gpu(v, device) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(to_gpu(k, device) for k in ob)
    elif isinstance(ob, list):
        return [to_gpu(k, device) for k in ob]
    else:
        try:
            return ob.to(device)
        except Exception:
            return ob


from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor


@torch.no_grad()
def interpolate_pose(
    initial: Float[Tensor, "4 4"], final: Float[Tensor, "4 4"], t: float,
) -> Float[Tensor, "4 4"]:
    # Get the relative rotation.
    r_initial = initial[:3, :3]
    r_final = final[:3, :3]
    r_relative = r_final @ r_initial.T
    r_relative = r_relative.float()

    # Convert it to axis-angle to interpolate it.
    r_relative = R.from_matrix(r_relative.cpu().numpy()).as_rotvec()
    r_relative = R.from_rotvec(r_relative * t).as_matrix()
    r_relative = torch.tensor(r_relative, dtype=final.dtype, device=final.device)
    r_interpolated = r_relative @ r_initial

    # Interpolate the position.
    t_initial = initial[:3, 3]
    t_final = final[:3, 3]
    t_interpolated = t_initial + (t_final - t_initial) * t

    # Assemble the result.
    result = torch.zeros_like(initial)
    result[3, 3] = 1
    result[:3, :3] = r_interpolated
    result[:3, 3] = t_interpolated
    return result


def add_wobble(t, radius=0.2):
    angle = 2 * np.pi * t
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    # torch make array [x,y,0]
    return torch.tensor([x, y, 0.0], device=radius.device, dtype=radius.dtype)


@torch.no_grad()
def interpolate_pose_wobble(
    initial: Float[Tensor, "4 4"],
    final: Float[Tensor, "4 4"],
    t: float,
    wobble: bool = True,
) -> Float[Tensor, "4 4"]:
    # Get the relative rotation.
    r_initial = initial[:3, :3]
    r_final = final[:3, :3]
    r_relative = r_final @ r_initial.T
    r_relative = r_relative.float()

    # Convert it to axis-angle to interpolate it.
    r_relative = R.from_matrix(r_relative.cpu().numpy()).as_rotvec()
    r_relative = R.from_rotvec(r_relative * t).as_matrix()
    r_relative = torch.tensor(r_relative, dtype=final.dtype, device=final.device)
    r_interpolated = r_relative @ r_initial

    # Interpolate the position.
    t_initial = initial[:3, 3]
    t_final = final[:3, 3]
    dir = t_final - t_initial
    t_interpolated = t_initial + (dir) * t

    if wobble:
        radius = torch.norm(dir) * 0.05
        t_wobble = add_wobble(t, radius)
        t_interpolated += t_wobble

    # Assemble the result.
    result = torch.zeros_like(initial)
    result[3, 3] = 1
    result[:3, :3] = r_interpolated
    result[:3, 3] = t_interpolated
    return result

def select_random_sequence(rng, list_length, sequence_length, use_first_frame_prob=0, start_frame_id=1):
    """Returns a list of contiguous random indices, shortened if needed."""

    actual_length = min(sequence_length, list_length)
    max_start = list_length - actual_length

    if rng.random() < use_first_frame_prob or max_start <= 0:
        start_index = min(start_frame_id, list_length - actual_length)
    else:
        start_index = rng.integers(0, max_start + 1)

    return np.arange(start_index, start_index + actual_length)

def inverse_transformation(T: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Computes the inverse of a 4x4 transformation matrix.
    
    Args:
        T (torch.Tensor or np.ndarray): A 4x4 transformation matrix.
        
    Returns:
        torch.Tensor or np.ndarray: The inverse of the transformation matrix,
                                    in the same type as the input.
    """
    if isinstance(T, np.ndarray):
        assert T.shape == (4, 4), "Input matrix must be of shape (4, 4)"
        
        R = T[:3, :3]
        t = T[:3, 3]
        
        R_inv = R.T
        t_inv = -R_inv @ t
        
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        
        return T_inv

    elif isinstance(T, torch.Tensor):
        assert T.shape == (4, 4), "Input matrix must be of shape (4, 4)"
        
        R = T[:3, :3]
        t = T[:3, 3]
        
        R_inv = R.T
        t_inv = -R_inv @ t
        
        T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        
        return T_inv

    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")

@torch.no_grad()
def interpolate_intrinsics(
    initial: Float[Tensor, "3 3"], final: Float[Tensor, "3 3"], t: float,
) -> Float[Tensor, "3 3"]:
    return initial + (final - initial) * t


import functools
import torch.nn as nn


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def prepare_video_viz(frames):
    for f in range(len(frames)):
        frames[f] = rearrange(frames[f], "b h w c -> h (b w) c")
    return frames


class JsonLogger:
    def __init__(self, path: str, 
            filter_fn: Optional[Callable[[str,Any],bool]]=None):
        if filter_fn is None:
            filter_fn = lambda k,v: isinstance(v, numbers.Number)

        # default to append mode
        self.path = path
        self.filter_fn = filter_fn
        self.file = None
        self.last_log = None
    
    def start(self):
        # use line buffering
        try:
            self.file = file = open(self.path, 'r+', buffering=1)
        except FileNotFoundError:
            self.file = file = open(self.path, 'w+', buffering=1)

        # Move the pointer (similar to a cursor in a text editor) to the end of the file
        pos = file.seek(0, os.SEEK_END)

        # Read each character in the file one at a time from the last
        # character going backwards, searching for a newline character
        # If we find a new line, exit the search
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        # now the file pointer is at one past the last '\n'
        # and pos is at the last '\n'.
        last_line_end = file.tell()
        
        # find the start of second last line
        pos = max(0, pos-1)
        file.seek(pos, os.SEEK_SET)
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        # now the file pointer is at one past the second last '\n'
        last_line_start = file.tell()

        if last_line_start < last_line_end:
            # has last line of json
            last_line = file.readline()
            self.last_log = json.loads(last_line)
        
        # remove the last incomplete line
        file.seek(last_line_end)
        file.truncate()
    
    def stop(self):
        self.file.close()
        self.file = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def log(self, data: dict):
        filtered_data = dict(
            filter(lambda x: self.filter_fn(*x), data.items()))
        # save current as last log
        self.last_log = filtered_data
        for k, v in filtered_data.items():
            if isinstance(v, numbers.Integral):
                filtered_data[k] = int(v)
            elif isinstance(v, numbers.Number):
                filtered_data[k] = float(v)
        buf = json.dumps(filtered_data)
        # ensure one line per json
        buf = buf.replace('\n','') + '\n'
        self.file.write(buf)
    
    def get_last_log(self):
        return copy.deepcopy(self.last_log)

def rotation_angle(initial_direction, target_direction):
    # Normalize the vectors
    initial_direction_normalized = initial_direction / np.linalg.norm(initial_direction)
    target_direction_normalized = target_direction / np.linalg.norm(target_direction)
    
    # Find the rotation angle (arc cosine of the dot product)
    cos_angle = np.dot(initial_direction_normalized, target_direction_normalized)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return angle

def semantic_to_color(
    mask: np.ndarray | torch.Tensor,
    max_labels: int = 256,
    seed: int = 42
) -> np.ndarray | torch.Tensor:
    """
    Convert a semantic mask [1, H, W] or [H, W] into an RGB image [H, W, 3].
    Supports both numpy.ndarray and torch.Tensor inputs.
    Each label gets a consistent and unique color across images.
    """
    is_tensor = torch.is_tensor(mask)

    if is_tensor:
        device = mask.device
        mask_proc = mask
        if mask_proc.ndim == 3 and mask_proc.shape[0] == 1:
            mask_proc = mask_proc[0]
        mask_int = mask_proc.long()

        gen = torch.Generator(device=device).manual_seed(seed)
        permuted_ids = torch.randperm(max_labels, generator=gen, device=device)

        palette = torch.zeros((max_labels, 3), dtype=torch.uint8, device=device)
        for idx, lbl in enumerate(permuted_ids.tolist()):
            h = idx / max_labels
            r_, g_, b_ = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            palette[lbl] = torch.tensor(
                [int(255 * r_), int(255 * g_), int(255 * b_)],
                dtype=torch.uint8,
                device=device
            )

        mask_clipped = mask_int.clamp(0, max_labels - 1)
        return palette[mask_clipped]

    else:
        mask_np = mask
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        mask_np = mask_np.astype(int)

        rng = np.random.default_rng(seed)
        permuted_ids = rng.permutation(max_labels)

        palette = np.zeros((max_labels, 3), dtype=np.uint8)
        for idx, lbl in enumerate(permuted_ids):
            h = idx / max_labels
            r_, g_, b_ = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            palette[lbl] = [int(255 * r_), int(255 * g_), int(255 * b_)]

        mask_clipped = np.clip(mask_np, 0, max_labels - 1)
        return palette[mask_clipped]

def viz_feat(feat):
    _, _, h, w = feat.shape
    feat = feat.squeeze(0).permute((1,2,0))
    projected_featmap = feat.reshape(-1, feat.shape[-1]).cpu()

    pca = PCA(n_components=3)
    pca.fit(projected_featmap)
    pca_features = pca.transform(projected_featmap)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    res_pred = Image.fromarray(pca_features.reshape(h, w, 3).astype(np.uint8))
    return res_pred