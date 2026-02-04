import os
import numpy as np
from typing import Optional, List, Tuple, Callable, Union
from PIL import Image
import torch
from torch import Tensor
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def get_yaw_from_pose(pose_matrix):
    raw_z_axis = pose_matrix[:3, 2]
    # -Z
    forward = -raw_z_axis
    # X-Z plane (atan2(x, z))
    yaw = np.arctan2(forward[0], forward[2])
    return yaw

def angle_difference(theta1, theta2):
    delta_theta = theta2 - theta1    
    delta_theta = delta_theta - 2 * np.pi * np.floor((delta_theta + np.pi) / (2 * np.pi))    
    return delta_theta

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def get_delta_np(actions):
    # append zeros to first action (unbatched)
    ex_actions = np.concatenate((np.zeros((1, actions.shape[1])), actions), axis=0)
    delta = ex_actions[1:] - ex_actions[:-1]
    
    return delta

def pose_lh2rh(pose: np.ndarray) -> np.ndarray:
    """Convert a left-handed (e.g. Habitat) 4x4 pose to right-handed (e.g. Open3D)."""
    F = np.diag([1, 1, -1, 1])
    return F @ pose @ F

def pose_gl_cam2world_to_open3d_cam2world(T_gl_c2w: np.ndarray) -> np.ndarray:
    """OpenGL cam->world  (X right, Y up, Z backward)  → Open3D/OpenCV cam->world (X right, Y down, Z forward)."""
    D_h = np.diag([1.0, -1.0, -1.0, 1.0])
    T_gl_c2w = np.asarray(T_gl_c2w, dtype=np.float64)
    return T_gl_c2w @ D_h

def pose_gl_world2cam_to_open3d_world2cam(E_gl_w2c: np.ndarray) -> np.ndarray:
    """OpenGL world->cam  → Open3D/OpenCV world->cam."""
    E_gl_w2c = np.asarray(E_gl_w2c, dtype=np.float64)
    D_h = np.diag([1.0, -1.0, -1.0, 1.0])
    return D_h @ E_gl_w2c

def points_gl_to_open3d(points: np.ndarray) -> np.ndarray:
    """Convert 3D points from OpenGL (X right, Y up, Z backward) to Open3D/OpenCV (X right, Y down, Z forward)."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1 and points.shape[0] == 3:
        return points * np.array([1.0, -1.0, -1.0])
    elif points.ndim == 2 and points.shape[1] == 3:
        return points * np.array([1.0, -1.0, -1.0])
    else:
        raise ValueError("points_gl_to_open3d: input must be Nx3 or 3")

def points_open3d_to_gl(points: np.ndarray) -> np.ndarray:
    """Convert 3D points from Open3D/OpenCV (X right, Y down, Z forward) to OpenGL (X right, Y up, Z backward)."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1 and points.shape[0] == 3:
        return points * np.array([1.0, -1.0, -1.0])
    elif points.ndim == 2 and points.shape[1] == 3:
        return points * np.array([1.0, -1.0, -1.0])
    else:
        raise ValueError("points_open3d_to_gl: input must be Nx3 or 3")

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def to_pil(
    arr: Union[np.ndarray, "torch.Tensor"],
    *,
    assume_range: Optional[str] = None,  # "0_1" or "0_255" or None to auto
    kind: Optional[str] = None           # "rgb" | "rgba" | "gray" | "depth" | "normals" | None(auto)
) -> Image.Image:
    """
    Convert numpy/torch array to a PIL.Image with sensible defaults.

    - RGB/RGBA:
        * float in [0,1] -> scale to 0..255 uint8
        * float in [0,255] or uint16 -> clip to 0..255 and cast uint8
    - GRAY:
        * same as above; returns mode "L"
    - DEPTH:
        * 2D float meters -> 16-bit PNG ("I;16") after scaling clamped meters
    - NORMALS:
        * [-1,1] -> map to [0,255] uint8 RGB
    """
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except Exception:
        pass

    if arr is None:
        raise ValueError("to_pil: got None")

    arr = np.asarray(arr)  # ensure numpy
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]  # squeeze HxWx1 -> HxW

    # Auto-detect kind if not provided
    if kind is None:
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            kind = "rgb" if arr.shape[2] == 3 else "rgba"
        elif arr.ndim == 2:
            kind = "gray"
        elif arr.ndim == 3 and arr.shape[2] == 1:
            kind = "depth"
            arr = np.squeeze(arr, axis=2)  # Remove channel dimension
        else:
            # fallback: try treat as rgb-like if last dim ==3
            if arr.ndim >= 3 and arr.shape[-1] == 3:
                kind = "rgb"
                arr = np.reshape(arr, (*arr.shape[:-1], 3))
            else:
                raise ValueError(f"Unsupported shape for to_pil: {arr.shape}")

    if kind == "normals":
        # map [-1,1] -> [0,255]
        arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
        kind = "rgb"
        assume_range = "0_1"

    if kind == "depth":
        if arr.ndim != 2:
            raise ValueError("Depth image must be HxW.")
        depth_m = np.asarray(arr, dtype=np.float32)
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        return Image.fromarray(depth_mm, mode="I;16")

    if kind in ("rgb", "rgba"):
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("RGB/RGBA image must be HxWx3 or HxWx4")
        # Determine scaling
        a = arr.astype(np.float32)
        if assume_range == "0_1" or (assume_range is None and a.max() <= 1.5):
            a = np.clip(a * 255.0, 0.0, 255.0)
        else:
            a = np.clip(a, 0.0, 255.0)
        a = a.astype(np.uint8)
        return Image.fromarray(a, mode="RGB" if arr.shape[2] == 3 else "RGBA")

    if kind == "gray":
        a = arr.astype(np.float32)
        if assume_range == "0_1" or (assume_range is None and a.max() <= 1.5):
            a = np.clip(a * 255.0, 0.0, 255.0)
        else:
            a = np.clip(a, 0.0, 255.0)
        return Image.fromarray(a.astype(np.uint8), mode="L")

    raise ValueError(f"Unknown kind: {kind}")

def _as_4x4(E: torch.Tensor) -> torch.Tensor:
    if E.shape == (4, 4):
        return E
    if E.shape == (3, 4):
        pad = torch.tensor([[0, 0, 0, 1]], device=E.device, dtype=E.dtype)
        return torch.cat([E, pad], dim=0)
    raise ValueError(f"Extrinsic must be (3,4) or (4,4); got {E.shape}")

def _to_hw3(points: torch.Tensor):
    """Accepts (H,W,3) or (3,H,W) torch -> returns (H*W,3) np and (H,W)."""
    if points.ndim != 3:
        raise ValueError(f"_to_hw3 expects 3D tensor; got {points.shape}")
    if points.shape[0] == 3:
        points = points.permute(1, 2, 0)  # (H,W,3)
    H, W, C = points.shape
    assert C == 3
    return points.reshape(-1, 3).detach().cpu().numpy(), (H, W)

def _image_to_hw3(image: torch.Tensor) -> np.ndarray:
    """Accepts (3,H,W) or (H,W,3) torch -> (H*W,3) np in [0,1]."""
    if image.ndim != 3:
        raise ValueError(f"_image_to_hw3 expects 3D tensor; got {image.shape}")
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)  # (H,W,3)
    img = image.detach().cpu().float().numpy()
    if img.max() > 1.0:
        img = img / 255.0
    H, W, _ = img.shape
    return img.reshape(-1, 3)

def _flatten_at_least_hw(x: torch.Tensor, H: int, W: int) -> np.ndarray:
    """
    Flatten any per-frame tensor and ensure it has AT LEAST H*W elements.
    Returns a 1D numpy array (length >= H*W). Caller will slice to H*W.
    """
    x_np = x.detach().cpu().numpy()
    flat = x_np.reshape(-1)  # row-major flatten
    need = H * W
    if flat.size < need:
        raise ValueError(
            f"Per-frame array too small: need >= {need} (=H*W) but got {flat.size}. "
            f"Original per-frame shape: {tuple(x.shape)}; expected something like (H,W) or (H,W,1)."
        )
    return flat

def _depth_valid_mask_per_frame(dm_t: torch.Tensor, H: int, W: int) -> np.ndarray:
    """
    Build validity mask from a per-frame depth map of arbitrary layout,
    as long as it has at least H*W elements.
    """
    flat = _flatten_at_least_hw(dm_t, H, W)[: H * W]
    return np.isfinite(flat) & (flat > 0)

def _conf_flat_per_frame(cm_t: torch.Tensor, H: int, W: int) -> np.ndarray:
    """
    Flatten a per-frame confidence map of arbitrary layout to length H*W.
    """
    flat = _flatten_at_least_hw(cm_t, H, W)[: H * W]
    return flat

def _mask_take(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return arr
    return arr[mask]

def _ensure_time_dim(x, expect_desc: str):
    """
    Ensure tensors have leading T or are expanded to T=1.
    Returns (xT, T) where xT has T leading dim (for numpy or torch).
    """
    if x is None:
        return None, 0
    if isinstance(x, np.ndarray):
        if x.ndim == 4:           # (T,H,W,3)
            return x, x.shape[0]
        if x.ndim == 3:           # (H,W,3)
            return x[None], 1
        raise ValueError(f"{expect_desc}: unexpected np shape {x.shape}")
    if isinstance(x, torch.Tensor):
        if x.ndim >= 4:
            return x, x.shape[0]  # e.g., (T,3,H,W) or (T,H,W,1) or (T,1,H,W)...
        if x.ndim == 3:
            return x.unsqueeze(0), 1
        raise ValueError(f"{expect_desc}: unexpected torch shape {x.shape}")
    raise ValueError(f"{expect_desc}: unsupported type {type(x)}")

def _slice_per_frame(x, t):
    """Slice the t-th frame across various shapes/numpy/torch without reordering."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x[t] if x.ndim >= 4 else x  # (T,...) or already per-frame
    if isinstance(x, torch.Tensor):
        return x[t] if x.ndim >= 4 else x  # (T,...) or already per-frame
    raise ValueError(f"Unsupported type in _slice_per_frame: {type(x)}")

def _get_T_c1_for_frame(ref_E_first_cw: torch.Tensor,
                        points_are_world_frame: bool,
                        current_E_cw_t: torch.Tensor | None) -> torch.Tensor:
    T_c1w = _as_4x4(ref_E_first_cw)  # first camera-from-world
    if points_are_world_frame:
        return T_c1w
    if current_E_cw_t is None:
        raise ValueError("current_extrinsic_cw is required when points_are_world_frame=False.")
    T_ckw = _as_4x4(current_E_cw_t)
    T_wck = torch.linalg.inv(T_ckw)
    return T_c1w @ T_wck

def _transform_points_np(pts_flat: np.ndarray, T_4x4: torch.Tensor) -> np.ndarray:
    if pts_flat.size == 0:
        return pts_flat
    T_np = T_4x4.detach().cpu().float().numpy()
    pts_h = np.concatenate([pts_flat, np.ones((pts_flat.shape[0], 1))], axis=1)
    return (pts_h @ T_np.T)[:, :3]

def _stack_points_colors(all_pts: list[np.ndarray], all_cols: list[np.ndarray | None]):
    pts = np.concatenate(all_pts, axis=0) if len(all_pts) else np.empty((0,3), dtype=np.float32)
    cols = None
    if any(c is not None for c in all_cols):
        cols = np.concatenate(
            [c if c is not None else np.zeros((ap.shape[0],3))
             for ap, c in zip(all_pts, all_cols)],
            axis=0
        ).clip(0,1)
    return pts, cols

def _move_time_from_dim1_to_dim0(x, T_expected: int):
    """
    If x is shaped (1,T,...) move T to the leading dim -> (T,...).
    Works for torch or numpy. Otherwise returns x unchanged.
    """
    if x is None:
        return None
    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim >= 4:
        if x.shape[0] == 1 and x.shape[1] == T_expected:
            return x.squeeze(0)
    return x

def _align_time_dim_like(pmT, xT):
    """Align xT's leading time dimension to match pmT (which is (T,...))."""
    if xT is None:
        return None
    T_pm = pmT.shape[0]
    return _move_time_from_dim1_to_dim0(xT, T_pm)

def pad34_to44(T34: np.ndarray) -> np.ndarray:
    T44 = np.eye(4, dtype=T34.dtype)
    T44[:3, :4] = T34
    return T44

def cam_center_from_w2c_cv(T_w2c: np.ndarray) -> np.ndarray:
    """Camera center from OpenCV w2c (3x4 or 4x4)."""
    if T_w2c.shape == (3, 4):
        T_w2c = pad34_to44(T_w2c)
    R, t = T_w2c[:3, :3], T_w2c[:3, 3]
    return (-R.T @ t)

def cam_center_from_c2w_cv(T_c2w: np.ndarray) -> np.ndarray:
    """Camera center from OpenCV c2w (4x4)."""
    return T_c2w[:3, 3]

def to_np_keep(x):
    if isinstance(x, torch.Tensor):
        return x.squeeze(0).detach().cpu().numpy()
    return x

def umeyama_sim3(X: np.ndarray, Y: np.ndarray, with_scale: bool = True):
    """
    Find s, R, t such that Y ≈ s * R * X + t (X,Y: Nx3).
    Returns (s, R, t).
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY
    Sigma = (Yc.T @ Xc) / X.shape[0]
    U, D, Vt = np.linalg.svd(Sigma)
    Sfix = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        Sfix[2, 2] = -1
    R = U @ Sfix @ Vt
    if with_scale:
        varX = (Xc ** 2).sum() / X.shape[0]
        s = np.trace(np.diag(D) @ Sfix) / varX
    else:
        s = 1.0
    t = muY - s * (R @ muX)
    return s, R, t

def flatten_points_colors(world_pts_HW3: np.ndarray,
                           img_HW3: np.ndarray,
                           mask_HW: Optional[np.ndarray] = None,
                           stride: int = 1):
    """
    Flatten HxWx3 world points and matching HxWx3 image colors to Nx3.
    Optionally apply a boolean mask and a stride for downsampling.
    """
    H, W = world_pts_HW3.shape[:2]
    if mask_HW is None:
        mask_HW = np.ones((H, W), dtype=bool)

    take = np.zeros_like(mask_HW, dtype=bool)
    take[0:H:stride, 0:W:stride] = True
    take &= mask_HW

    pts = world_pts_HW3[take]          # (N, 3)
    cols = img_HW3[take]               # (N, 3)

    # Normalize colors to [0,1]
    if cols.dtype.kind in ("u", "i"):
        cols = cols.astype(np.float32) / 255.0
    else:
        cols = np.clip(cols.astype(np.float32), 0.0, 1.0)

    return pts.astype(np.float32), cols.astype(np.float32)

def _as_tensor3x3(x, device, dtype):
    if isinstance(x, torch.Tensor):
        K = x
    else:
        K = torch.tensor(x, device=device, dtype=dtype)
    assert K.shape[-2:] == (3,3), f"Expected 3x3 intrinsics, got {tuple(K.shape)}"
    return K

def _depth_scale_from_intrinsics(K_est_seq, K_gt):
    """
    K_est_seq: torch.Tensor (T,3,3) or (3,3)
    K_gt: torch.Tensor (3,3)
    Returns a scalar float scale s.
    """
    if K_est_seq.ndim == 2:
        K_est_seq = K_est_seq.unsqueeze(0)  # (1,3,3)

    fxg, fyg = float(K_gt[0,0]), float(K_gt[1,1])
    s_list = []
    for t in range(K_est_seq.shape[0]):
        fx_e = float(K_est_seq[t, 0, 0])
        fy_e = float(K_est_seq[t, 1, 1])
        # geometric-mean focal ratio
        s_t = ((fxg * fyg) / max(1e-8, fx_e * fy_e)) ** 0.5
        s_list.append(s_t)
    # robust global scale
    return float(np.median(s_list))

def interpolate_pose_wobble(
    initial: Tensor,
    final: Tensor,
    t: float,
    wobble: bool = True,
) -> Tensor:
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

def square_image(image):
    # consider both rgb and depth images
    height, width = image.shape[:2]
    size = min(height, width)
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    return image[start_y:start_y+size, start_x:start_x+size]

# def pose_robot_to_opencv(T_robot: np.ndarray) -> np.ndarray:
#     # T_cv = A * T_robot * A^{-1} ; A^{-1} uses R^T
#     R_cv_robot = np.array([
#         [0, -1,  0],
#         [0,  0, -1],
#         [1,  0,  0],
#     ], dtype=float)
#     A = np.eye(4)
#     A[:3, :3] = R_cv_robot
#     A_inv = np.eye(4)
#     A_inv[:3, :3] = R_cv_robot.T
#     return A @ T_robot @ A_inv

# def pose_opencv_to_robot(T_cv: np.ndarray) -> np.ndarray:
#     R_cv_robot = np.array([
#         [0, -1,  0],
#         [0,  0, -1],
#         [1,  0,  0],
#     ], dtype=float)

#     A = np.eye(4)
#     A[:3, :3] = R_cv_robot

#     A_inv = np.eye(4)
#     A_inv[:3, :3] = R_cv_robot.T  # inverse of rotation

#     return A_inv @ T_cv @ A

def pose_robot_to_opencv(T_robot: np.ndarray) -> np.ndarray:
    # T_cv = A * T_robot * A^{-1} ; A^{-1} uses R^T
    # F = np.diag([1, 1, -1, 1])
    # T_robot =  F @ T_robot @ F
    R_cv_robot = np.array([
        [0, -1,  0],
        [0,  0, -1],
        [-1,  0,  0],
    ], dtype=float)
    A = np.eye(4)
    A[:3, :3] = R_cv_robot
    A_inv = np.eye(4)
    A_inv[:3, :3] = R_cv_robot.T
    return A @ T_robot @ A_inv

def pose_opencv_to_robot(T_cv: np.ndarray) -> np.ndarray:
    R_cv_robot = np.array([
        [0, -1,  0],
        [0,  0, -1],
        [-1,  0,  0],
    ], dtype=float)

    A = np.eye(4)
    A[:3, :3] = R_cv_robot

    A_inv = np.eye(4)
    A_inv[:3, :3] = R_cv_robot.T  # inverse of rotation
    
    ret = A_inv @ T_cv @ A
    # ret = A_inv @ T_cv
    # F = np.diag([1, 1, -1, 1])
    # ret = F @ ret @ F
    return ret


def _draw_frame(ax, T, label=None, axis_len=0.1, lw=2.0):
    T = np.asarray(T, dtype=float)
    assert T.shape == (4, 4)

    R = T[:3, :3]
    t = T[:3, 3]

    x_end = t + axis_len * R[:, 0]
    y_end = t + axis_len * R[:, 1]
    z_end = t + axis_len * R[:, 2]

    ax.plot([t[0], x_end[0]], [t[1], x_end[1]], [t[2], x_end[2]], color="r", linewidth=lw)
    ax.plot([t[0], y_end[0]], [t[1], y_end[1]], [t[2], y_end[2]], color="g", linewidth=lw)
    ax.plot([t[0], z_end[0]], [t[1], z_end[1]], [t[2], z_end[2]], color="b", linewidth=lw)

    ax.scatter([t[0]], [t[1]], [t[2]], s=30, color="k")

    if label is not None:
        ax.text(t[0], t[1], t[2], f" {label}", fontsize=10)


def _set_axes_equal(ax):
    """Make 3D axes have equal scale so frames don't look skewed."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xmid = (xlim[0] + xlim[1]) / 2.0
    ymid = (ylim[0] + ylim[1]) / 2.0
    zmid = (zlim[0] + zlim[1]) / 2.0

    xr = abs(xlim[1] - xlim[0])
    yr = abs(ylim[1] - ylim[0])
    zr = abs(zlim[1] - zlim[0])
    r = max(xr, yr, zr) / 2.0

    ax.set_xlim3d(xmid - r, xmid + r)
    ax.set_ylim3d(ymid - r, ymid + r)
    ax.set_zlim3d(zmid - r, zmid + r)


def plot_two_poses(T1, T2, save_path, axis_len=0.1, elev=20, azim=45, dpi=200):
    """
    Plot two 4x4 poses on the same 3D coordinate system and save to save_path.
    """
    T1 = np.asarray(T1, dtype=float)
    T2 = np.asarray(T2, dtype=float)
    assert T1.shape == (4, 4) and T2.shape == (4, 4), "Both poses must be (4,4)."

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    _draw_frame(ax, T1, label="T1", axis_len=axis_len)
    _draw_frame(ax, T2, label="T2", axis_len=axis_len)

    # expand limits to include both origins + some margin
    pts = np.stack([T1[:3, 3], T2[:3, 3]], axis=0)
    pad = axis_len * 1.5
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_zlim(pts[:, 2].min() - pad, pts[:, 2].max() + pad)
    _set_axes_equal(ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("Two Poses")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)

def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0,  s],
                     [ 0, 1,  0],
                     [-s, 0,  c]], dtype=float)

def flip_yaw_in_Twc(T_wc: np.ndarray) -> np.ndarray:
    """
    Flip yaw sign for a camera pose T_wc (camera -> world),
    assuming yaw is about world +Y and forward is OpenCV +Z.
    Keeps translation unchanged and preserves the remaining rotation
    in the yaw-removed frame (e.g., roll).
    """
    T_wc = np.asarray(T_wc, dtype=float)
    assert T_wc.shape == (4, 4)

    R = T_wc[:3, :3]
    t = T_wc[:3, 3].copy()

    # camera forward (OpenCV +Z) expressed in world
    f = R[:, 2]

    # yaw about world Y (using world X,Z components of forward)
    yaw = np.arctan2(f[0], f[2])  # atan2(x, z)

    # flip yaw direction
    yaw_new = -yaw
    R_new = Ry(yaw_new)

    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t
    return T_new