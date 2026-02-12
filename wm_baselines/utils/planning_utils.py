import numpy as np
from typing import Optional, List, Tuple, Callable
import numpy as np
import heapq
import math
import skimage.draw
from scipy.ndimage import distance_transform_edt, binary_dilation, convolve
from wm_baselines.world_model.base_world_model import BaseWorldModel

KERNEL_8 = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)

def rotation_angle(initial_direction, target_direction):
    # Normalize the vectors
    initial_direction_normalized = initial_direction / np.linalg.norm(initial_direction)
    target_direction_normalized = target_direction / np.linalg.norm(target_direction)
    
    # Find the rotation angle (arc cosine of the dot product)
    cos_angle = np.dot(initial_direction_normalized, target_direction_normalized)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return angle

def _bresenham_segment_is_free(
    occ, p1: Tuple[float,float,float], p2: Tuple[float,float,float]
) -> bool:
    """Collision check along the xz segment using the occupancy grid."""
    (r1, c1) = occ._world_to_grid((p1[0], p1[2]))
    (r2, c2) = occ._world_to_grid((p2[0], p2[2]))
    rr, cc = skimage.draw.line(r1, c1, r2, c2)
    for r, c in zip(rr, cc):
        if not occ._in_bounds((r, c)) or occ.occupancy[r, c] == 1:
            return False
    return True

def _shortcut_path(
    pts: np.ndarray,
    is_free: Optional[Callable[[Tuple[float,float,float], Tuple[float,float,float]], bool]] = None,
    max_passes: int = 2
) -> np.ndarray:
    """
    Greedy 'shortcut' smoothing: try to replace (i..j) with a straight segment if collision-free.
    pts: (N,3)
    """
    if is_free is None or len(pts) <= 2:
        return pts

    pts = pts.copy()
    for _ in range(max_passes):
        i = 0
        out = [pts[0]]
        while i < len(pts) - 1:
            j = len(pts) - 1
            found = False
            while j > i + 1:
                if is_free(tuple(pts[i]), tuple(pts[j])):
                    # skip interior points
                    out.append(pts[j])
                    i = j
                    found = True
                    break
                j -= 1
            if not found:
                out.append(pts[i + 1])
                i += 1
        pts = np.asarray(out)
    return pts

def _dedupe_polyline(P: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Remove consecutive duplicates from (N,3) polyline."""
    if P.shape[0] <= 1:
        return P
    keep = [0]
    for i in range(1, P.shape[0]):
        if np.linalg.norm(P[i] - P[keep[-1]]) > tol:
            keep.append(i)
    return P[keep]

def _catmull_rom_spline(
    P: np.ndarray,
    samples_per_seg: int = 20,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Centripetal Catmull–Rom via Hermite form.
    - De-duplicates control points.
    - Computes chord-length parameters t[i] with eps guards.
    - Falls back to linear on degenerate segments.
    """
    assert P.ndim == 2 and P.shape[1] == 3
    P = _dedupe_polyline(P, tol=1e-8)
    N = P.shape[0]
    if N == 1:
        return P.copy()
    if N == 2:
        # simple linear upsample
        t = np.linspace(0, 1, max(2, samples_per_seg), endpoint=True)
        return (1 - t)[:, None] * P[0] + t[:, None] * P[1]

    # build centripetal chord-length params
    eps = 1e-12
    tvals = [0.0]
    for i in range(N - 1):
        dt = (np.linalg.norm(P[i+1] - P[i]) ** alpha) + eps
        tvals.append(tvals[-1] + dt)
    t = np.array(tvals, dtype=float)

    out = []
    for i in range(N - 1):
        # neighbor indices with clamping
        i0 = max(0, i - 1)
        i1 = i
        i2 = i + 1
        i3 = min(N - 1, i + 2)

        P0, P1, P2, P3 = P[i0], P[i1], P[i2], P[i3]
        t0, t1, t2, t3 = t[i0], t[i1], t[i2], t[i3]

        # tangents (finite differences in parametrization space)
        denom1 = max(t2 - t0, eps)
        denom2 = max(t3 - t1, eps)
        m1 = (P2 - P0) / denom1
        m2 = (P3 - P1) / denom2

        # u in [0,1] over this segment
        S = samples_per_seg
        # include endpoint on final segment to avoid missing last point
        endpoint = (i == N - 2)
        u = np.linspace(0.0, 1.0, S, endpoint=endpoint)[:, None]

        # cubic Hermite basis
        u2 = u * u
        u3 = u2 * u
        h00 =  2*u3 - 3*u2 + 1
        h10 =      u3 - 2*u2 + u
        h01 = -2*u3 + 3*u2
        h11 =      u3 -    u2

        seg = h00*P1 + h10*(t2 - t1)*m1 + h01*P2 + h11*(t2 - t1)*m2
        out.append(seg)

    C = np.vstack(out)

    # remove any accidental NaNs (shouldn't happen now) and dedupe again
    C = C[~np.isnan(C).any(axis=1)]
    C = _dedupe_polyline(C, tol=1e-9)
    return C if C.shape[0] > 0 else P.copy()

def _equal_arclength_resample(curve: np.ndarray, ds: float) -> np.ndarray:
    """
    Resample curve (M,3) to ~equal spacing ds.
    Robust to NaNs, zero-length, and tiny totals.
    """
    assert curve.ndim == 2 and curve.shape[1] == 3
    ds = float(ds)
    if not np.isfinite(ds) or ds <= 0:
        raise ValueError(f"ds must be > 0, got {ds}")

    C = curve[~np.isnan(curve).any(axis=1)]
    C = _dedupe_polyline(C, tol=1e-9)
    if C.shape[0] == 0:
        return curve[:1]  # fall back
    if C.shape[0] == 1:
        return C.copy()

    diffs = np.diff(C, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    total = float(seglen.sum())

    if not np.isfinite(total) or total < 1e-9:
        # Degenerate: return the endpoints
        return C[[0, -1]]

    # choose count robustly: at least 2 points including end
    n_samples = int(max(2, math.floor(total / ds + 1e-9) + 1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    s_new = np.linspace(0.0, total, n_samples, endpoint=True)

    # linear interpolate per axis
    x = np.interp(s_new, s, C[:, 0])
    y = np.interp(s_new, s, C[:, 1])
    z = np.interp(s_new, s, C[:, 2])
    X = np.stack([x, y, z], axis=1)
    return X

def _yaw_from_tangent(tan: np.ndarray) -> float:
    """Yaw about +Y from xz-projection of the tangent."""
    dx, dy, dz = tan
    return math.atan2(dz, dx)  # heading in xz plane

def _poses_from_xyz(
    xyz: np.ndarray,
    ref_pose: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Compute 4x4 SE(3) poses with yaw from path tangent (roll=pitch=0).
    If ref_pose is provided, each pose is multiplied with ref_pose to give
    full world-frame pose.

    Args:
        xyz: (N,3) path positions
        ref_pose: optional (4,4) reference transform (world <- local)

    Returns:
        list of (4,4) SE(3) numpy arrays
    """
    N = xyz.shape[0]
    # finite-diff tangent
    tangents = np.zeros_like(xyz)
    tangents[1:-1] = xyz[2:] - xyz[:-2]
    tangents[0]    = xyz[1] - xyz[0]
    tangents[-1]   = xyz[-1] - xyz[-2]

    poses = []
    for p, t in zip(xyz, tangents):
        if np.linalg.norm(t) < 1e-9:
            yaw = 0.0
        else:
            yaw = _yaw_from_tangent(t)

        cy, sy = math.cos(yaw), math.sin(yaw)
        R = np.array([
            [ cy, 0.0,  sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0,  cy],
        ], dtype=np.float32)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = p.astype(np.float32)

        if ref_pose is not None:
            T = ref_pose @ T  # compose with reference transform T_w_r*T_r_local

        poses.append(T)
    return poses

def _vectorized_frontiers(occ: np.ndarray) -> np.ndarray:
    """Frontier mask: free cells adjacent to unknown."""
    free = (occ == 0)
    unknown = (occ == -1)
    unk_nbrs = convolve(unknown.astype(np.uint8), KERNEL_8, mode="nearest") > 0
    return free & unk_nbrs

def _frustum_gain(occ, rows, cols, yaw, fx=None, cx=None, max_range_m=3.0, res=0.05, w=None):
    """
    Approx expected info gain from a candidate (rows, cols) center using a 2D fan (FOV) ray-cast.
    If intrinsics (fx,cx,w) are supplied, derive HFOV; else use ~90° by default.
    """
    if fx is not None and cx is not None and w is not None:
        h_fov = 2 * math.atan(w / (2.0 * fx))
    else:
        h_fov = math.radians(90.0)
    r_max = int(round(max_range_m / res))
    N = max(int(round(h_fov * max_range_m / res)), 60)
    ang = np.linspace(yaw - h_fov/2, yaw + h_fov/2, N, endpoint=True)
    # grid-centric angle convention: x to the right (cols), z downward (rows)
    ang = ang + math.pi/2
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    ds = (np.arange(1, r_max+1) * res).astype(np.float32)
    rr = np.clip((rows + (ds[:,None]*sin_a)/res).astype(int), 0, occ.shape[0]-1)
    cc = np.clip((cols + (ds[:,None]*cos_a)/res).astype(int), 0, occ.shape[1]-1)
    ray = occ[rr, cc]                # (r_max, N)
    blocked = (ray == 1)
    first = np.where(blocked.any(0), blocked.argmax(0), r_max)
    # unknown up to the first hit (exclusive)
    gain = 0
    for j in range(N):
        end = int(first[j])
        if end > 0:
            seg = ray[:end, j]
            gain += int((seg == -1).sum())
    return gain

def _segment_is_free_grid(occ, p1_rc: Tuple[int,int], p2_rc: Tuple[int,int]) -> bool:
    """Check a grid segment (r,c)->(r,c) for collision."""
    r1, c1 = p1_rc
    r2, c2 = p2_rc
    for r, c in zip(*skimage.draw.line(r1, c1, r2, c2)):
        if not occ._in_bounds((r, c)) or occ.occupancy[r, c] == 1:
            return False
    return True

def _segment_is_free_world(occ, p1_xz: Tuple[float,float], p2_xz: Tuple[float,float]) -> bool:
    """Check a world segment (x,z)->(x,z) for collision via grid samples."""
    (r1, c1) = occ._world_to_grid(p1_xz)
    (r2, c2) = occ._world_to_grid(p2_xz)
    return _segment_is_free_grid(occ, (r1, c1), (r2, c2))

def _waypoint_y(occ, r: int, c: int, use_height_map: bool) -> float:
    if use_height_map and (occ.height_map is not None):
        return float(occ.height_map[r, c])
    return 0.0

def _rc_to_xyz_center(occ, r: int, c: int, use_height_map: bool) -> np.ndarray:
    x, z = occ._grid_to_world((r, c))
    y = _waypoint_y(occ, r, c, use_height_map)
    return np.array([x, y, z], dtype=np.float32)

def goals_and_forwards_to_poses(goals: List[np.ndarray], forwards: List[np.ndarray]) -> List[np.ndarray]:
    """Convert list of goals and forwards to list of camera-to-world poses."""
    poses = []
    for goal, forward in zip(goals, forwards):
        goal = np.asarray(goal, dtype=np.float64).reshape(3)
        forward = BaseWorldModel.normalize(np.asarray(forward, dtype=np.float64).reshape(3))

        # Camera Z looks along the forward direction
        z_axis = BaseWorldModel.normalize(forward)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = BaseWorldModel.normalize(np.cross(y_axis, z_axis))

        R = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float64)   # (3,3)
        t = goal.astype(np.float64).reshape(3, 1)

        T_cam2world = np.eye(4, dtype=np.float64)
        T_cam2world[:3, :3] = R
        T_cam2world[:3, 3:4] = t
        poses.append(T_cam2world)
    return poses

def distance_traveled_step(action_name, action_params):
    distance_traveled, angle_turned = 0.0, 0.0
    if action_name == "MoveAhead" or action_name == "MoveBack":
        # Calculate distance between last two positions
        distance = action_params.get('moveMagnitude', 0.0)
        distance_traveled += distance
    elif action_name == "RotateLeft" or action_name == "RotateRight":
        # Track angle turned
        degrees = action_params.get('degrees', 0)
        angle_turned += abs(degrees)
    return distance_traveled, angle_turned

def _robot_fwd_xz(robot_T: np.ndarray, forward_axis: int = 0) -> np.ndarray:
    """Robot forward axis projected into world XZ as a unit 2D vector [fx, fz]."""
    R = robot_T[:3, :3]
    f = R[:, forward_axis].astype(float)  # world-frame axis
    fx, fz = float(f[0]), float(f[2])
    n = math.hypot(fx, fz)
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return np.array([fx / n, fz / n], dtype=float)

def _front_cos_and_bearing(
    robot_T: np.ndarray,
    rx: float, rz: float,
    gx: float, gz: float,
    forward_axis: int = 0
) -> Tuple[float, float]:
    """Return (cos_front, bearing) where bearing is signed angle goal-dir relative to robot heading."""
    fwd = _robot_fwd_xz(robot_T, forward_axis=forward_axis)  # (fx, fz)
    dx, dz = gx - rx, gz - rz
    dn = math.hypot(dx, dz)
    if dn < 1e-9:
        return 1.0, 0.0
    dir_xz = np.array([dx / dn, dz / dn], dtype=float)
    cos_front = float(np.clip(np.dot(fwd, dir_xz), -1.0, 1.0))
    # signed bearing in [-pi, pi]
    cross = fwd[0] * dir_xz[1] - fwd[1] * dir_xz[0]
    bearing = math.atan2(cross, cos_front)
    return cos_front, bearing

