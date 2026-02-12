import numpy as np
import open3d as o3d
import torch, functools
from PIL import Image
from typing import Iterable, Sequence, Tuple, Union, List, Optional
from wm_baselines.agent.perception.occupancy import OccupancyMap
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, SiglipModel
import torchvision.transforms as T
from lpips import LPIPS

try:
    from shapely.geometry import Polygon
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

class Box3D:
    """
    Oriented 3D box with a single yaw (rotation about +Z).
    center: (x, y, z)
    size:   (l, w, h)  # along box local x (length, forward), y (width, left), z (height, up)
    yaw:    radians
    """
    __slots__ = ("x","y","z","l","w","h","yaw")
    def __init__(self, center, size, yaw):
        self.x, self.y, self.z = map(float, center)
        self.l, self.w, self.h = map(float, size)
        self.yaw = float(yaw)

    @property
    def z_bottom(self): return self.z - self.h/2.0
    @property
    def z_top(self):    return self.z + self.h/2.0

def _rect_corners_bev(box: Box3D) -> np.ndarray:
    """Return 4x2 array of (x,y) corners in world coords, CCW."""
    l, w = box.l, box.w
    # local rectangle corners (x,y) in box frame (centered):
    # front-left, front-right, back-right, back-left (CCW)
    X = np.array([[ l/2,  w/2],
                  [ l/2, -w/2],
                  [-l/2, -w/2],
                  [-l/2,  w/2]], dtype=np.float64)
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    R = np.array([[c, -s],
                  [s,  c]])
    Y = (X @ R.T) + np.array([box.x, box.y])
    return Y

def _to_pil(x):
    if isinstance(x, Image.Image): return x
    if isinstance(x, np.ndarray):
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        if x.ndim == 2:  # gray
            x = np.stack([x]*3, axis=-1)
        return Image.fromarray(x)
    raise TypeError("Expected PIL or np.ndarray")

def _to_pil_lpips(x):
    if isinstance(x, Image.Image): return x.convert("RGB")
    if isinstance(x, np.ndarray):
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        if x.ndim == 2:
            x = np.stack([x]*3, axis=-1)
        return Image.fromarray(x).convert("RGB")
    raise TypeError("Expected PIL.Image or np.ndarray")

def _pair_lists(A, B):
    """Allow single image vs list broadcasting; always return two lists of equal length."""
    if isinstance(A, (Image.Image, np.ndarray)): A = [A]
    if isinstance(B, (Image.Image, np.ndarray)): B = [B]
    assert len(A) == len(B), f"Length mismatch: {len(A)} vs {len(B)}"
    return list(A), list(B)

def _apply_resize_pipeline(
    imgs: List[Image.Image],
    processor,            # HF processor (CLIP/SigLIP)
    resize: Optional[int] = None,
    center_crop: bool = True
):
    """
    If resize is None -> use model default via processor.
    If resize is int -> override: resize shortest side to this, keep aspect; optional center crop to a square of this size if center_crop.
    """
    if resize is None:
        return processor(images=imgs, return_tensors="pt")
    # manual resize but still let processor handle normalization
    # Resize shortest side to 'resize' (keep AR)
    resized = []
    for im in imgs:
        w, h = im.size
        if min(w, h) == resize:
            im_r = im
        else:
            if w < h:
                new_w = resize
                new_h = int(round(h * (resize / w)))
            else:
                new_h = resize
                new_w = int(round(w * (resize / h)))
            im_r = im.resize((new_w, new_h), Image.BICUBIC)
        if center_crop:
            # crop center square of size 'resize'
            w2, h2 = im_r.size
            left = max(0, (w2 - resize)//2)
            top  = max(0, (h2 - resize)//2)
            im_r = im_r.crop((left, top, left+resize, top+resize))
        resized.append(im_r)
    return processor(images=resized, return_tensors="pt")

def bev_intersection_area(a: Box3D, b: Box3D) -> float:
    A = _rect_corners_bev(a)
    B = _rect_corners_bev(b)
    if _HAS_SHAPELY:
        poly_a = Polygon(A)
        poly_b = Polygon(B)
        if not poly_a.is_valid or not poly_b.is_valid:
            return 0.0
        return float(poly_a.intersection(poly_b).area)
    elif _HAS_CV2:
        retval, inter = cv2.intersectConvexConvex(A.astype(np.float32), B.astype(np.float32))
        return float(retval) if inter is not None else 0.0
    else:
        # Minimal fallback: approximate by raster (coarse but dependency-free)
        mins = np.floor(np.minimum(A.min(0), B.min(0)) - 0.25).astype(int)
        maxs = np.ceil( np.maximum(A.max(0), B.max(0)) + 0.25).astype(int)
        H, W = (maxs - mins)[1], (maxs - mins)[0]
        if H <= 0 or W <= 0:
            return 0.0
        yy, xx = np.mgrid[0:H, 0:W]
        pts = np.c_[xx.ravel() + mins[0] + 0.5, yy.ravel() + mins[1] + 0.5]

        def inside(poly):
            # winding for convex quad
            v = poly[np.newaxis, :, :] - pts[:, np.newaxis, :]
            cross = np.cross(v, np.roll(v, -1, axis=1))
            return np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)
        m = inside(A) & inside(B)
        return float(m.sum())  # pixel area with 1x1 pixels


def bev_iou(a: Box3D, b: Box3D) -> float:
    inter = bev_intersection_area(a, b)
    area_a = a.l * a.w
    area_b = b.l * b.w
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def iou_3d(a: Box3D, b: Box3D) -> float:
    # BEV intersection footprint:
    inter_area = bev_intersection_area(a, b)
    if inter_area <= 0:
        return 0.0
    # Height overlap (interval intersection on z)
    z1 = max(a.z_bottom, b.z_bottom)
    z2 = min(a.z_top, b.z_top)
    inter_h = max(0.0, z2 - z1)
    inter_vol = inter_area * inter_h
    vol_a = a.l * a.w * a.h
    vol_b = b.l * b.w * b.h
    union = vol_a + vol_b - inter_vol
    return 0.0 if union <= 0 else inter_vol / union


def box_errors(a: Box3D, b: Box3D):
    """Return center L2, per-dim size abs/rel errors, and yaw error in deg (wrapped)."""
    center_err = float(np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z]))
    size_abs = (abs(a.l - b.l), abs(a.w - b.w), abs(a.h - b.h))
    size_rel = tuple(abs(sa - sb) / max(1e-6, sb) for sa, sb in zip((a.l,a.w,a.h), (b.l,b.w,b.h)))
    yaw_diff = (a.yaw - b.yaw + np.pi) % (2*np.pi) - np.pi
    yaw_err_deg = float(abs(yaw_diff) * 180.0 / np.pi)
    return {
        "center_L2": center_err,
        "size_abs": size_abs,    # (dl, dw, dh)
        "size_rel": size_rel,    # (%, %, %)
        "yaw_err_deg": yaw_err_deg,
    }

def box3d_from_aabb(pcd: o3d.geometry.PointCloud) -> Box3D:
    """
    Build a Box3D from an axis-aligned bbox of an Open3D point cloud.
    Yaw is set to 0 by definition (aligned with world axes).
    """
    if len(pcd.points) == 0:
        raise ValueError("Empty point cloud")

    aabb = pcd.get_axis_aligned_bounding_box()
    c = aabb.get_center()                  # (x, y, z)
    extent = aabb.get_extent()             # (dx, dy, dz) side lengths
    # Interpret size as (length=X extent, width=Y extent, height=Z extent)
    return Box3D(center=tuple(c), size=tuple(extent), yaw=0.0)


def box3d_from_obb(pcd: o3d.geometry.PointCloud, up_axis: str = "z") -> Box3D:
    """
    Build a Box3D from an oriented bbox (keeps yaw if you want it).
    Assumes 'yaw' is rotation about the chosen up-axis (default Z-up).
    """
    if len(pcd.points) == 0:
        raise ValueError("Empty point cloud")

    obb = pcd.get_oriented_bounding_box()
    c = obb.center
    extent = obb.extent                     # side lengths along OBB's local axes
    R = obb.R if hasattr(obb, "R") else obb.get_rotation_matrix()  # 3x3

    if up_axis.lower() == "z":
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        # Map size to (l, w, h) ~ (extent along local x,y,z)
        size = (extent[0], extent[1], extent[2])
    elif up_axis.lower() == "y":
        # yaw around +Y; project local axes to world XZ plane
        yaw = float(np.arctan2(R[0, 2], R[0, 0]))  # one reasonable convention
        size = (extent[0], extent[2], extent[1])   # reorder so height=Y
    elif up_axis.lower() == "x":
        yaw = float(np.arctan2(R[2, 1], R[1, 1]))
        size = (extent[1], extent[2], extent[0])   # height=X
    else:
        raise ValueError("up_axis must be one of {'x','y','z'}")

    return Box3D(center=tuple(c), size=tuple(size), yaw=yaw)

def chamfer_np(A, B, squared=True):
    # A: (N,3), B: (M,3)
    diff = A[:, None, :] - B[None, :, :]          # (N,M,3)
    d2 = np.sum(diff*diff, axis=2)                # (N,M)
    if squared:
        da = d2.min(axis=1).mean()
        db = d2.min(axis=0).mean()
    else:
        da = np.sqrt(d2.min(axis=1)).mean()
        db = np.sqrt(d2.min(axis=0)).mean()
    return da + db

def default_chamfer_from_gt(gt_pcd, *, chamfer_fn):
    """
    Construct a scale-aware default Chamfer distance using only gt_pcd.

    We do this by shifting the GT point cloud by one bounding-box diagonal
    along X and computing Chamfer(gt, gt_shifted).

    gt_pcd: array-like of shape [N, 3]
    chamfer_fn: function (pred_pts: np.ndarray, gt_pts: np.ndarray) -> float
    """
    gt = np.asarray(gt_pcd, dtype=np.float32)
    if gt.ndim != 2 or gt.shape[0] == 0:
        # Fallback if something is wrong with gt
        return float("nan")

    bbox_min = gt.min(axis=0)
    bbox_max = gt.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)

    # If degenerate bbox, pick a unit diagonal to avoid zero shift
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0

    # Shift along X by one diagonal (you can bump to 2*diag if you want it even worse)
    offset = np.array([diag, 0.0, 0.0], dtype=np.float32)
    pred_fake = gt + offset

    return chamfer_fn(pred_fake, gt)

@functools.lru_cache(maxsize=4)
def _load_clip(model_name: str, device: str):
    model = CLIPModel.from_pretrained(model_name)
    proc  = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device).eval()
    return model, proc

@torch.no_grad()
def clip_similarity(
    images_a,
    images_b,
    model_name: str = "openai/clip-vit-large-patch14",
    device: Optional[str] = None,
    resize: Optional[int] = None,   # e.g., 224, 336; None uses model default
    center_crop: bool = True,
    dtype: Optional[torch.dtype] = None  # e.g., torch.float16 on GPU
):
    """
    Pairwise cosine similarities for arbitrary-size inputs.
    Returns: {'cosine': Tensor[N], 'emb_a': Tensor[N,D], 'emb_b': Tensor[N,D]}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _load_clip(model_name, device)
    A, B = _pair_lists(images_a, images_b)
    A = [_to_pil(x) for x in A]; B = [_to_pil(x) for x in B]

    batch = _apply_resize_pipeline(A + B, processor, resize=resize, center_crop=center_crop)
    pixel_values = batch["pixel_values"].to(device)
    if dtype is not None:
        pixel_values = pixel_values.to(dtype)
        model = model.to(dtype)

    feats = model.get_image_features(pixel_values=pixel_values)  # [2N, D]
    feats = torch.nn.functional.normalize(feats.float(), dim=-1)

    N = len(A)
    fa, fb = feats[:N], feats[N:]
    cos = torch.sum(fa * fb, dim=-1)  # [-1,1]
    sim01 = (cos + 1.0) * 0.5  
    # make values to be float instead of tensor for easier json serialization
    sim01 = sim01.cpu().numpy().tolist()[0] if len(sim01) == 1 else sim01.cpu().numpy().tolist()
    cos = cos.cpu().numpy().tolist()[0] if len(cos) == 1 else cos.cpu().numpy().tolist()
    fa = fa.cpu().numpy().tolist()[0] if len(fa) == 1 else fa.cpu().numpy().tolist()
    fb = fb.cpu().numpy().tolist()[0] if len(fb) == 1 else fb.cpu().numpy().tolist()
    return {"similarity": sim01, "cosine": cos, "emb_a": fa, "emb_b": fb}

@functools.lru_cache(maxsize=4)
def _load_siglip(model_name: str, device: str):
    model = SiglipModel.from_pretrained(model_name)
    proc  = AutoProcessor.from_pretrained(model_name)
    model = model.to(device).eval()
    return model, proc

@torch.no_grad()
def siglip_similarity(
    images_a,
    images_b,
    model_name: str = "google/siglip-so400m-patch14-384",
    device: Optional[str] = None,
    resize: Optional[int] = None,    # e.g., 224, 384; None uses model default
    center_crop: bool = True,
    dtype: Optional[torch.dtype] = None
):
    """
    Pairwise semantic similarity with SigLIP for arbitrary-size inputs.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _load_siglip(model_name, device)
    A, B = _pair_lists(images_a, images_b)
    A = [_to_pil(x) for x in A]; B = [_to_pil(x) for x in B]

    batch = _apply_resize_pipeline(A + B, processor, resize=resize, center_crop=center_crop)
    pixel_values = batch["pixel_values"].to(device)
    if dtype is not None:
        pixel_values = pixel_values.to(dtype)
        model = model.to(dtype)

    feats = model.get_image_features(pixel_values=pixel_values)  # [2N, D]
    feats = torch.nn.functional.normalize(feats.float(), dim=-1)

    N = len(A)
    fa, fb = feats[:N], feats[N:]
    cos = torch.sum(fa * fb, dim=-1)
    sim01 = (cos + 1.0) * 0.5
    # make values to be float instead of tensor for easier json serialization
    sim01 = sim01.cpu().numpy().tolist()[0] if len(sim01) == 1 else sim01.cpu().numpy().tolist()
    cos = cos.cpu().numpy().tolist()[0] if len(cos) == 1 else cos.cpu().numpy().tolist()
    fa = fa.cpu().numpy().tolist()[0] if len(fa) == 1 else fa.cpu().numpy().tolist()
    fb = fb.cpu().numpy().tolist()[0] if len(fb) == 1 else fb.cpu().numpy().tolist()
    return {"similarity": sim01, "cosine": cos, "emb_a": fa, "emb_b": fb}

@functools.lru_cache(maxsize=4)
def _load_lpips(net: str, device: str):
    # nets: 'vgg' (standard), 'alex', 'squeeze'
    model = LPIPS(net=net).to(device).eval()
    return model

def _prep_lpips_t(img: Image.Image, size: Optional[Tuple[int,int]] = None):
    """
    Convert PIL -> Tensor in [-1,1], optionally resizing/cropping to `size` (H,W).
    If size is None, keep original resolution.
    """
    tfm = [
        T.ToTensor(),              # [0,1]
        T.Lambda(lambda t: t*2-1)  # [-1,1]
    ]
    if size is not None:
        H, W = size
        # Preserve AR: resize so the *smaller* side >= target, then center-crop
        tfm = [
            T.Resize(min(H, W), interpolation=Image.BICUBIC),
            T.CenterCrop((H, W)),
            *tfm
        ]
    pipe = T.Compose(tfm)
    return pipe(img.convert("RGB"))

def _common_size(
    imgA: Image.Image, imgB: Image.Image,
    mode: str = "fixed",        # "fixed" | "min" | "max"
    fixed_size: Tuple[int,int] = (256,256)
) -> Tuple[int,int]:
    """
    Decide a common (H,W) for an LPIPS pair.
    - fixed: use fixed_size
    - min:   use (min(Ha,Hb), min(Wa,Wb))
    - max:   use (max(Ha,Hb), max(Wa,Wb))
    """
    if mode == "fixed":
        return fixed_size
    Ha, Wa = imgA.size[1], imgA.size[0]
    Hb, Wb = imgB.size[1], imgB.size[0]
    if mode == "min":
        return (min(Ha, Hb), min(Wa, Wb))
    if mode == "max":
        return (max(Ha, Hb), max(Wa, Wb))
    raise ValueError("mode must be one of {'fixed','min','max'}")

@torch.no_grad()
def lpips_distance(
    images_a,
    images_b,
    net: str = "vgg",
    device: Optional[str] = None,
    size_mode: str = "fixed",                  # 'fixed' | 'min' | 'max'
    fixed_size: Tuple[int,int] = (256,256)     # used when size_mode='fixed'
):
    """
    Perceptual distance with LPIPS (lower = more similar) for arbitrary-size inputs.
    We resize each pair to a *common size* (configurable) before evaluation.

    Returns:
      {'distance': Tensor[N]}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_lpips(net, device)

    A, B = _pair_lists(images_a, images_b)
    A = [_to_pil(x) for x in A]; B = [_to_pil(x) for x in B]

    tens_a, tens_b = [], []
    sizes_used: List[Tuple[int,int]] = []
    for a, b in zip(A, B):
        H, W = _common_size(a, b, mode=size_mode, fixed_size=fixed_size)
        ta = _prep_lpips_t(a, (H, W))
        tb = _prep_lpips_t(b, (H, W))
        tens_a.append(ta); tens_b.append(tb)
        sizes_used.append((H, W))

    a_t = torch.stack(tens_a, dim=0).to(device)
    b_t = torch.stack(tens_b, dim=0).to(device)

    out = model(a_t, b_t, normalize=False)

    dist = out.view(out.shape[0], -1).mean(dim=1).detach().cpu()
    dist = dist.cpu().numpy().tolist()[0] if len(dist) == 1 else dist.cpu().numpy().tolist()
    return {"distance": dist, "sizes": sizes_used}

def eval_occupancy_maps(gt_map: OccupancyMap,
                        pred_map: OccupancyMap):
    """
    Evaluate a predicted occupancy map against ground truth, aligning by
    world coordinates (X/Z) using origin and resolution.

    Both maps must have:
        - occupancy: np.ndarray of shape (nz, nx) with values in {-1, 0, 1}
        - x_min, z_min: world coordinates of the (0,0) cell corner
        - nx, nz: grid dimensions along X and Z
        - resolution: cell size in meters

    Values:
        -1 = unknown
         0 = free
         1 = occupied

    Returns a dict with:
        - accuracy_known : accuracy on cells where both gt and pred are known (0 or 1)
        - coverage       : fraction of gt-known cells where pred is also known (0 or 1)
        - iou_free       : IoU for free cells (class 0)
        - iou_occupied   : IoU for occupied cells (class 1)
        - miou           : mean IoU over free+occupied
    """
    # Basic checks
    if gt_map.occupancy is None or pred_map.occupancy is None:
        raise ValueError("Both maps must have non-None occupancy arrays.")

    if gt_map.occupancy.ndim != 2 or pred_map.occupancy.ndim != 2:
        raise ValueError("Expected 2D occupancy arrays (nz, nx).")

    # Require same resolution (otherwise you’d need resampling logic)
    if not np.isclose(gt_map.resolution, pred_map.resolution, rtol=0, atol=1e-6):
        raise ValueError(
            f"Resolution mismatch: gt={gt_map.resolution}, pred={pred_map.resolution}"
        )

    res = gt_map.resolution

    # World extents for each map (half-open intervals)
    def world_extents(m: OccupancyMap):
        x0 = m.x_min
        z0 = m.z_min
        x1 = x0 + m.nx * res
        z1 = z0 + m.nz * res
        return x0, x1, z0, z1

    gt_x0, gt_x1, gt_z0, gt_z1 = world_extents(gt_map)
    pr_x0, pr_x1, pr_z0, pr_z1 = world_extents(pred_map)

    # Overlapping world interval
    x0_ov = max(gt_x0, pr_x0)
    x1_ov = min(gt_x1, pr_x1)
    z0_ov = max(gt_z0, pr_z0)
    z1_ov = min(gt_z1, pr_z1)

    if x1_ov <= x0_ov or z1_ov <= z0_ov:
        raise ValueError("The two occupancy maps have no overlapping world region.")

    # Number of overlapping cells along X and Z.
    # We assume grids are aligned to the same resolution / lattice.
    nx_ov = int(round((x1_ov - x0_ov) / res))
    nz_ov = int(round((z1_ov - z0_ov) / res))

    if nx_ov <= 0 or nz_ov <= 0:
        raise ValueError("Overlapping region is too small or empty.")

    # Convert overlapping world origin to indices for each map
    def overlap_indices(m: OccupancyMap, x0_ov, z0_ov, nx_ov, nz_ov):
        ix0 = int(round((x0_ov - m.x_min) / res))
        iz0 = int(round((z0_ov - m.z_min) / res))
        ix1 = ix0 + nx_ov
        iz1 = iz0 + nz_ov
        return iz0, iz1, ix0, ix1  # (z_start, z_end, x_start, x_end)

    gt_z0_i, gt_z1_i, gt_x0_i, gt_x1_i = overlap_indices(
        gt_map, x0_ov, z0_ov, nx_ov, nz_ov
    )
    pr_z0_i, pr_z1_i, pr_x0_i, pr_x1_i = overlap_indices(
        pred_map, x0_ov, z0_ov, nx_ov, nz_ov
    )

    # Safety checks against array bounds
    def check_bounds(m: OccupancyMap, z0, z1, x0, x1, name: str):
        if not (0 <= z0 < z1 <= m.nz and 0 <= x0 < x1 <= m.nx):
            raise ValueError(
                f"Overlap indices out of bounds for {name}: "
                f"z [{z0}, {z1}), x [{x0}, {x1}), "
                f"map nz={m.nz}, nx={m.nx}"
            )

    check_bounds(gt_map, gt_z0_i, gt_z1_i, gt_x0_i, gt_x1_i, "gt_map")
    check_bounds(pred_map, pr_z0_i, pr_z1_i, pr_x0_i, pr_x1_i, "pred_map")

    # Crop both occupancy grids to the overlapping region
    gt_occ = gt_map.occupancy[gt_z0_i:gt_z1_i, gt_x0_i:gt_x1_i]
    pr_occ = pred_map.occupancy[pr_z0_i:pr_z1_i, pr_x0_i:pr_x1_i]

    # --- Metric computation (same as before, now on world-aligned crops) ---

    # 1) Masks
    mask_gt_known = (gt_occ != -1)
    mask_pred_known = (pr_occ != -1)
    mask_eval = mask_gt_known & mask_pred_known

    # 2) Accuracy on cells where both are known
    if np.any(mask_eval):
        accuracy_known = (gt_occ[mask_eval] == pr_occ[mask_eval]).mean()
    else:
        accuracy_known = np.nan

    # 3) Coverage: among GT-known cells, how many did the predictor label as 0/1?
    if np.any(mask_gt_known):
        coverage = (mask_gt_known & mask_pred_known).sum() / mask_gt_known.sum()
    else:
        coverage = np.nan

    # 4) IoU for free (0) and occupied (1), ignoring GT unknown
    iou_free = np.nan
    iou_occupied = np.nan

    # class 0: free
    inter_free = ((gt_occ == 0) & (pr_occ == 0) & mask_gt_known).sum()
    union_free = (((gt_occ == 0) | (pr_occ == 0)) & mask_gt_known).sum()
    if union_free > 0:
        iou_free = inter_free / union_free

    # class 1: occupied
    inter_occ = ((gt_occ == 1) & (pr_occ == 1) & mask_gt_known).sum()
    union_occ = (((gt_occ == 1) | (pr_occ == 1)) & mask_gt_known).sum()
    if union_occ > 0:
        iou_occupied = inter_occ / union_occ

    # 5) Mean IoU over free+occupied
    miou = np.nanmean([iou_free, iou_occupied])

    return {
        "accuracy_known": float(accuracy_known),
        "coverage": float(coverage),
        "iou_free": float(iou_free),
        "iou_occupied": float(iou_occupied),
        "miou": float(miou),
    }