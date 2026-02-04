from typing import Tuple, List, Optional, Union
import numpy as np
import math
from scipy.ndimage import distance_transform_edt, binary_dilation, label
from belief_baselines.agent.perception.occupancy import OccupancyMap
from belief_baselines.utils.planning_utils import (
    _vectorized_frontiers,
    _frustum_gain
)

GOAL_SAMPLING_REGISTRY = {}

def register_goal_sampling(name):
    def deco(f):
        GOAL_SAMPLING_REGISTRY[name] = f
        return f
    return deco

@register_goal_sampling("simple_frontier")
def sample_simple_frontier_goals(
    occ: OccupancyMap,
    robot_pose_4x4: np.ndarray,
    *,
    k: int = 3,
    alpha: float = 0.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    preferred_dist: float = 2.0,
    sharpness: float = 1.0,
    backup_k: int = 2,           # number of backup goals
    walk_steps: int = 10,
    max_radius_m: float = 2.0,
    retries_per_goal: int = 30,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Standalone frontier-based sampler.

    Args:
        occ: OccupancyMap instance (must have occupancy, resolution, _world_to_grid/_grid_to_world).
        robot_pose_4x4: 4x4 SE(3) pose matrix (we use translation for X,Z).
        k: number of top frontier goals.
        alpha/beta/gamma: weights for (info gain, bandpass distance, obstacle clearance).
        preferred_dist: bandpass peak distance (meters).
        sharpness: bandpass sharpness.
        backup_k: number of random-walk backup goals to add.
        walk_steps: max steps in random walk per backup goal.
        max_radius_m: limit random walk radius around robot (meters).
        retries_per_goal: attempts per backup goal to find a valid cell.

    Returns:
        (goals, forwards): two lists of np.ndarray shape (3,), world-frame.
    """
    assert occ.occupancy is not None, "call set_point_cloud() first"
    occ_grid = occ.occupancy
    nz, nx = occ_grid.shape

    # --- robot position (X,Z) from 4x4 pose ---
    rx = float(robot_pose_4x4[0, 3])
    rz = float(robot_pose_4x4[2, 3])
    robot_xy = (rx, rz)

    # --- obstacle clearance (Euclidean distance transform) ---
    if np.any(occ_grid == 1):
        clearance_cells = distance_transform_edt((occ_grid != 1).astype(np.uint8))
        clearance_m = clearance_cells * float(occ.resolution)
    else:
        clearance_m = np.full_like(occ_grid, fill_value=max(nz, nx) * occ.resolution, dtype=float)

    # --- find frontier cells (free cell with at least one unknown neighbor) ---
    frontiers: List[Tuple[int, int]] = []
    for r in range(nz):
        for c in range(nx):
            if occ_grid[r, c] != 0:
                continue
            neigh = occ_grid[max(r-1, 0):min(r+2, nz), max(c-1, 0):min(c+2, nx)]
            if np.any(neigh == -1):
                frontiers.append((r, c))

    # --- bandpass weighting for preferred distance ---
    def bandpass_weight(d_m: float, d_star: float, s: float) -> float:
        d_star = max(d_star, 1e-6)
        x = max(d_m, 0.0) / d_star
        return (x ** s) * math.exp(s * (1.0 - x))

    rr, cc = occ._world_to_grid(robot_xy)
    scored = []
    nbr_offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]

    for r, c in frontiers:
        # info gain = number of unknowns in 3x3
        gain = int((occ_grid[max(r-1, 0):min(r+2, nz), max(c-1, 0):min(c+2, nx)] == -1).sum())
        dist_robot_m = np.hypot(r - rr, c - cc) * float(occ.resolution)
        clearance = float(clearance_m[r, c])
        dist_term = bandpass_weight(dist_robot_m, preferred_dist, sharpness)
        score = alpha * gain + beta * dist_term + gamma * clearance

        x, z = occ._grid_to_world((r, c))
        goal = np.array([x, 0.0, z], dtype=np.float32)

        # forward direction points toward nearby unknowns (or robot->cell fallback)
        vx, vz = 0.0, 0.0
        for dr, dc in nbr_offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nz and 0 <= nc < nx and occ_grid[nr, nc] == -1:
                vx += dc
                vz += dr
        vx *= occ.resolution
        vz *= occ.resolution
        norm = math.hypot(vx, vz)
        if norm < 1e-9:
            fx, fz = x - rx, z - rz
            f_norm = math.hypot(fx, fz)
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float32) if f_norm < 1e-9 \
                else np.array([fx / f_norm, 0.0, fz / f_norm], dtype=np.float32)
        else:
            forward = np.array([vx / norm, 0.0, vz / norm], dtype=np.float32)

        scored.append((score, goal, forward))

    scored.sort(key=lambda x: x[0], reverse=True)
    goals = [g for _, g, _ in scored[:k]]
    forwards = [f for _, _, f in scored[:k]]

    # --- backup goals via short random walks in free space ---
    rng = getattr(occ, "rng", np.random.default_rng())
    max_radius_cells = max(1, int(round(max_radius_m / float(occ.resolution))))
    in_bounds = lambda r, c: 0 <= r < nz and 0 <= c < nx
    is_free = lambda r, c: in_bounds(r, c) and occ_grid[r, c] == 0

    taken_cells = {occ._world_to_grid((g[0], g[2])) for g in goals}
    for _ in range(backup_k):
        for _try in range(retries_per_goal):
            r, c = int(rr), int(cc)
            if not is_free(r, c):
                neigh = [(rr+dr, cc+dc) for dr in range(-2, 3) for dc in range(-2, 3)]
                neigh = [(int(rn), int(cn)) for rn, cn in neigh if is_free(int(rn), int(cn))]
                if not neigh:
                    break
                r, c = neigh[int(rng.integers(len(neigh)))]

            last_step = (0, 0)
            steps = int(rng.integers(low=max(5, walk_steps//2), high=walk_steps+1))
            for _s in range(steps):
                step_candidates = nbr_offsets if last_step == (0, 0) or rng.random() >= 0.5 else \
                    [last_step] + [t for t in nbr_offsets if t != last_step]
                dr, dc = step_candidates[int(rng.integers(len(step_candidates)))]
                nr, nc = r + dr, c + dc
                if not is_free(nr, nc):
                    continue
                if abs(nr - rr) > max_radius_cells or abs(nc - cc) > max_radius_cells:
                    continue
                r, c = nr, nc
                last_step = (dr, dc)

            if not is_free(r, c) or (r, c) in taken_cells:
                continue

            x, z = occ._grid_to_world((r, c))
            g = np.array([x, 0.0, z], dtype=np.float32)

            vx, vz = 0.0, 0.0
            for dr, dc in nbr_offsets:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and occ_grid[nr, nc] == -1:
                    vx += dc
                    vz += dr
            vx *= occ.resolution
            vz *= occ.resolution
            norm = math.hypot(vx, vz)
            if norm < 1e-9:
                fx, fz = x - rx, z - rz
                f_norm = math.hypot(fx, fz)
                fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32) if f_norm < 1e-9 \
                    else np.array([fx / f_norm, 0.0, fz / f_norm], dtype=np.float32)
            else:
                fwd = np.array([vx / norm, 0.0, vz / norm], dtype=np.float32)

            goals.append(g)
            forwards.append(fwd)
            taken_cells.add((r, c))
            break

    return goals, forwards

@register_goal_sampling("frontier")
def sample_frontier_goals(
    occ: "OccupancyMap",
    robot_T: np.ndarray,
    *,
    k: int = 1,
    intrinsics: Optional[np.ndarray] = None,
    max_range_m: float = 3.0,
    min_clearance_m: float = 0.25,
    nms_radius_m: float = 0.5,
    alpha_gain: float = 2.0,
    beta_pathcost: float = 1.0,
    gamma_clear: float = 0.5,
    delta_heading: float = 2.0,
    memory_mask: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    weight_jitter: float = 0.25,      # ~25% std multiplicative jitter on weights
    score_noise_std: float = 0.05,    # additive Gaussian noise on normalized score
    stochastic_pick: bool = True,     # use Gumbel-top-k order before NMS
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Improved frontier sampler:
    - frontier detection (vectorized),
    - visibility-aware info gain via frustum ray-casting,
    - proxy path cost via distance transform on inflated obstacles,
    - clearance safety, approach-angle preference, NMS diversity.
    Returns (goals_xyz, forward_dirs).
    """
    rng = rng if rng is not None else np.random.default_rng()
    occ_grid = occ.occupancy
    nz, nx = occ_grid.shape
    res = float(occ.resolution)

    # robot pose
    rx, rz = float(robot_T[0,3]), float(robot_T[2,3])
    rr, cc = occ._world_to_grid((rx, rz))

    # distance transform & inflated obstacles for safety/path proxy
    obstacles = (occ_grid == 1)
    inflated = binary_dilation(obstacles, structure=np.ones((3,3), np.uint8))  # 1-cell inflate; tune
    free_like = ~(inflated)  # traversable proxy
    # cost-to-go proxy: distance from robot through traversable space
    # (dt on True=traversable returns distance to nearest False; invert from robot via wavefront is better;
    # for simplicity, use Euclidean to robot on traversable mask + large penalty when blocked.)
    yy, xx = np.mgrid[0:nz, 0:nx]
    pathcost = np.hypot(yy-rr, xx-cc) * res
    pathcost[~free_like] += 1e6

    clearance = distance_transform_edt(~inflated).astype(float) * res

    # frontiers (band)
    fr_mask = _vectorized_frontiers(occ_grid)
    # restrict to safe band
    fr_mask &= (clearance >= min_clearance_m)

    cand_rc = np.argwhere(fr_mask)
    if cand_rc.size == 0:
        return [], []

    # orientation proxy per frontier: gradient of unknown distance gives outward normal
    unknown_dt = distance_transform_edt(~(occ_grid == -1)).astype(float)
    gy, gx = np.gradient(unknown_dt)
    # small epsilon avoid zero
    eps = 1e-9

    # intrinsics parse
    fx = cx = w = None
    if intrinsics is not None:
        fx = float(intrinsics[0,0]); cx = float(intrinsics[0,2]); w = int(round(2*cx))

    # score candidates
    gains = []
    pcs   = []
    clrs  = []
    head  = []
    goals = []
    fwdv  = []

    for r, c in cand_rc:
        x, z = occ._grid_to_world((int(r), int(c)))
        # heading towards frontier: negative gradient of unknown_dt points into unknown
        vx, vz = -gx[r, c], -gy[r, c]
        nrm = math.hypot(vx, vz)
        if nrm < eps:
            # fallback: point from robot to cell
            vx, vz = (c-cc), (r-rr)
            nrm = math.hypot(vx, vz) + eps
        fwd = np.array([vx/nrm, 0.0, vz/nrm], dtype=np.float32)

        # expected info gain by frustum casting at candidate
        yaw = math.atan2(fwd[2], fwd[0])
        gain = _frustum_gain(
            occ_grid, int(r), int(c), yaw,
            fx=fx, cx=cx, max_range_m=max_range_m, res=res, w=w
        )

        # path proxy cost & clearance
        pc = pathcost[r, c]
        cl = clearance[r, c]

        # heading alignment (prefer facing away from robot-to-cell vector)
        to_cell = np.array([x - rx, 0.0, z - rz], dtype=np.float32)
        tn = np.linalg.norm(to_cell) + eps
        cosang = float(np.dot(fwd, to_cell/tn))  # [−1,1], larger → pointing toward the frontier from robot
        # we want positive cosang
        head.append((cosang + 1.0) * 0.5)

        gains.append(gain)
        pcs.append(pc)
        clrs.append(cl)
        goals.append(np.array([x, 0.0, z], dtype=np.float32))
        fwdv.append(fwd)

    gains = np.asarray(gains, float)
    pcs   = np.asarray(pcs, float)
    clrs  = np.asarray(clrs, float)
    head  = np.asarray(head, float)

    # normalize
    def norm01(x):
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=np.max(x[np.isfinite(x)]), neginf=np.min(x[np.isfinite(x)]))
        rng = x.max() - x.min()
        return (x - x.min()) / (rng + 1e-9)

    G = norm01(gains)
    C = norm01(clrs)
    H = norm01(head)
    # path *cost* should be low → convert to score
    P = 1.0 - norm01(np.clip(pcs, 0, np.percentile(pcs, 95)))  # robustify tail

    # optional memory decay (downweight recently explored)
    if memory_mask is not None and memory_mask.shape == occ_grid.shape:
        M = 1.0 - norm01(memory_mask[cand_rc[:,0], cand_rc[:,1]])
    else:
        M = 1.0

    if weight_jitter > 0:
        def jitter(w):
            # multiplicative: w * (1 + N(0, weight_jitter)), floor at 0
            return max(0.0, float(w * (1.0 + rng.normal(0.0, weight_jitter))))
        a = jitter(alpha_gain)
        b = jitter(beta_pathcost)
        g = jitter(gamma_clear)
        d = jitter(delta_heading)
    else:
        a, b, g, d = alpha_gain, beta_pathcost, gamma_clear, delta_heading

    S = a*G + b*P + g*C + d*H
    S = S * M

    if score_noise_std > 0:
        S = S + rng.normal(0.0, score_noise_std, size=S.shape)

    # Non-maximum suppression / Poisson-disc to diversify
    picked = []
    taken = np.zeros(len(goals), dtype=bool)
    rad_cells = max(1, int(round(nms_radius_m / res)))

    if stochastic_pick:
        # Gumbel-top-k: sample order by argmax(S + Gumbel)
        # Gumbel(0,1) = -log(-log(U)), U~Uniform(0,1)
        U = rng.random(S.shape)
        gumbel = -np.log(-np.log(np.clip(U, 1e-12, 1.0-1e-12)))
        order = np.argsort(-(S + gumbel))
    else:
        order = np.argsort(-S)
    
    for idx in order:
        if taken[idx]:
            continue
        picked.append(idx)
        if len(picked) >= k:
            break
        r, c = cand_rc[idx]
        r0, r1 = max(0, r-rad_cells), min(nz, r+rad_cells+1)
        c0, c1 = max(0, c-rad_cells), min(nx, c+rad_cells+1)
        # mark neighborhood as taken
        box = (cand_rc[:,0] >= r0) & (cand_rc[:,0] < r1) & (cand_rc[:,1] >= c0) & (cand_rc[:,1] < c1)
        taken |= box

    goals_out   = [goals[i] for i in picked]
    forwards_out= [fwdv[i]  for i in picked]
    return goals_out, forwards_out

@register_goal_sampling("frontier_relaxed_random")
def sample_frontier_goals_relaxed_random(
    occ: "OccupancyMap",
    robot_T: np.ndarray,
    *,
    k: int = 3,
    min_clearance_m: float = 0.2,
    min_goal_dist_m: float = 0.25,    # filter corner case: too close to robot
    max_goal_dist_m: float = 5.0,     # filter corner case: unreasonably far
    edge_margin_m: float = 0.2,       # filter corner case: hugging grid borders
    nms_radius_m: float = 0.4,        # optional spatial de-duplication
    min_unknown_component_cells: int = 50,   # tune
    unknown_connectivity: int = 8,
    fallback_max_goal_dist_m: float | None = None,  # relax max distance if strict yields none
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Frontier-based goal selector that *only* filters out corner-case candidates
    and otherwise returns a randomized subset (no scoring / no ranking).

    Behavior change:
      - Try strict distance filter: min_goal_dist_m <= d <= max_goal_dist_m
      - If no candidates survive, relax ONLY the upper bound:
          min_goal_dist_m <= d <= fallback_max_goal_dist_m
        If fallback_max_goal_dist_m is None, then no upper bound in fallback.

    Returns (goals_xyz, forward_dirs).
    """
    rng = rng if rng is not None else np.random.default_rng()
    assert occ.occupancy is not None, "call set_point_cloud() first"

    occ_grid = occ.occupancy
    nz, nx = occ_grid.shape
    res = float(occ.resolution)

    # robot pose and grid coords
    rx, rz = float(robot_T[0, 3]), float(robot_T[2, 3])
    rr0, cc0 = occ._world_to_grid((rx, rz))

    # inflated obstacles to create a safety band
    obstacles = (occ_grid == 1)
    inflated = binary_dilation(obstacles, structure=np.ones((3, 3), np.uint8))
    clearance = distance_transform_edt(~inflated).astype(float) * res

    # frontier mask (vectorized), then basic safety filter
    fr_mask = _vectorized_frontiers(occ_grid)
    fr_mask &= (clearance >= min_clearance_m)

    # edge/border filter (avoid weird map edges)
    edge_cells = max(0, int(round(edge_margin_m / res)))
    if edge_cells > 0:
        core = np.zeros_like(fr_mask, dtype=bool)
        core[edge_cells : nz - edge_cells, edge_cells : nx - edge_cells] = True
        fr_mask &= core

    unknown = (occ_grid == -1)

    # label connected components of unknown
    structure = (
        np.ones((3, 3), dtype=np.uint8)
        if unknown_connectivity == 8
        else np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    )
    cc_labels, n_cc = label(unknown, structure=structure)

    if n_cc > 0:
        sizes = np.bincount(cc_labels.ravel())
        big = sizes >= min_unknown_component_cells
        big[0] = False
        big_unknown = big[cc_labels]
        big_unknown_d = binary_dilation(big_unknown, structure=np.ones((3, 3), np.uint8))
        fr_mask &= big_unknown_d

    cand_rc = np.argwhere(fr_mask)
    if cand_rc.size == 0:
        return [], []

    # orientation proxy per frontier: negative grad of unknown distance → outward normal
    unknown_dt = distance_transform_edt(~(occ_grid == -1)).astype(float)
    gy, gx = np.gradient(unknown_dt)
    eps = 1e-9

    # quick distance-to-robot (grid euclidean)
    yy, xx = np.mgrid[0:nz, 0:nx]
    dist_to_robot = np.hypot(yy - rr0, xx - cc0) * res

    # Build all candidate goals first (only clearance/edge/unknown filters already applied)
    keep_rc: list[tuple[int, int]] = []
    goals: list[np.ndarray] = []
    fwds: list[np.ndarray] = []
    dists: list[float] = []

    for r, c in cand_rc:
        r = int(r); c = int(c)

        # not inside inflated obstacle band (should be true due to clearance check, but double-guard)
        if inflated[r, c]:
            continue

        d = float(dist_to_robot[r, c])

        # drop too-close always
        if d < min_goal_dist_m:
            continue

        # world coords
        x, z = occ._grid_to_world((r, c))

        # forward dir: point into unknown (negative gradient), fallback = robot->cell
        vx = -float(gx[r, c])
        vz = -float(gy[r, c])
        nrm = math.hypot(vx, vz)
        if nrm < eps:
            vx = float(c - int(cc0))
            vz = float(r - int(rr0))
            nrm = math.hypot(vx, vz) + eps
        fwd = np.array([vx / nrm, 0.0, vz / nrm], dtype=np.float32)

        keep_rc.append((r, c))
        goals.append(np.array([x, 0.0, z], dtype=np.float32))
        fwds.append(fwd)
        dists.append(d)

    if not keep_rc:
        return [], []

    dists_arr = np.asarray(dists, dtype=float)

    # PASS 1 (strict): min <= d <= max_goal_dist_m
    strict_idx = np.nonzero(dists_arr <= float(max_goal_dist_m))[0].tolist()

    # PASS 2 (fallback): relax upper bound if strict empty
    if not strict_idx:
        if fallback_max_goal_dist_m is None:
            relaxed_idx = list(range(len(keep_rc)))  # no upper bound
        else:
            relaxed_idx = np.nonzero(dists_arr <= float(fallback_max_goal_dist_m))[0].tolist()

        # If still empty (e.g., fallback_max_goal_dist_m too small), return nothing.
        candidate_idx = relaxed_idx
    else:
        candidate_idx = strict_idx

    if not candidate_idx:
        return [], []

    # Optional Poisson-disc / NMS to remove near-duplicates spatially (applied on candidate set)
    if nms_radius_m > 0:
        rad_cells = max(1, int(round(nms_radius_m / res)))
        taken = np.zeros(len(candidate_idx), dtype=bool)
        kept_local = []

        order = rng.permutation(len(candidate_idx))
        r_arr = np.array([keep_rc[i][0] for i in candidate_idx], dtype=int)
        c_arr = np.array([keep_rc[i][1] for i in candidate_idx], dtype=int)

        for local_i in order:
            if taken[local_i]:
                continue
            kept_local.append(local_i)
            r, c = r_arr[local_i], c_arr[local_i]
            r0, r1 = max(0, r - rad_cells), min(nz, r + rad_cells + 1)
            c0, c1 = max(0, c - rad_cells), min(nx, c + rad_cells + 1)
            neighbor = (r_arr >= r0) & (r_arr < r1) & (c_arr >= c0) & (c_arr < c1)
            taken |= neighbor

        kept_idx = [candidate_idx[li] for li in kept_local]
    else:
        kept_idx = candidate_idx

    if not kept_idx:
        return [], []

    # uniformly sample up to k and shuffle final order
    if len(kept_idx) > k:
        picked = rng.choice(kept_idx, size=k, replace=False)
    else:
        picked = np.array(kept_idx, dtype=int)

    rng.shuffle(picked)

    goals_out    = [goals[i] for i in picked]
    forwards_out = [fwds[i]  for i in picked]
    return goals_out, forwards_out

@register_goal_sampling("frontier_balanced_turns")
def sample_frontier_goals_balanced_turns(
    occ: "OccupancyMap",
    robot_T: np.ndarray,
    *,
    k: int = 3,
    min_clearance_m: float = 0.25,
    min_goal_dist_m: float = 0.25,
    max_goal_dist_m: float = 5.0,
    fallback_max_goal_dist_m: Optional[float] = None,
    edge_margin_m: float = 0.2,
    nms_radius_m: float = 0.4,
    sector_half_span_rad: float = np.pi / 2,
    forward_half_width_rad: float = np.deg2rad(15.0),
    min_unknown_component_cells: int = 50,
    unknown_connectivity: int = 8,
    rng: Optional[np.random.Generator] = None,
    forward_fraction: float = 1.0,      # 0..1: fraction of selected goals that should be in the FWD bin
    front_score_power: float = 2.0,     # >=1: higher => stronger preference for straight-ahead over side/back
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Frontier-based goal selector that balances chosen goals across LEFT / FORWARD / RIGHT
    sectors w.r.t. the robot's heading, while preferring goals in front of the agent.

    Added fallback behavior:
      - Pass 1 (strict): min_goal_dist_m <= d <= max_goal_dist_m
      - If no candidates survive, Pass 2 (relaxed): min_goal_dist_m <= d <= fallback_max_goal_dist_m
        If fallback_max_goal_dist_m is None, there is no upper bound in fallback.
    """
    rng = rng if rng is not None else np.random.default_rng()
    assert occ.occupancy is not None, "call set_point_cloud() first"

    occ_grid = occ.occupancy
    nz, nx = occ_grid.shape
    res = float(occ.resolution)

    # robot world position and heading (assume +X of robot frame is "forward")
    rx, rz = float(robot_T[0, 3]), float(robot_T[2, 3])
    rr0, cc0 = occ._world_to_grid((rx, rz))
    rfx, rfz = float(robot_T[0, 0]), float(robot_T[2, 0])  # forward axis projected to xz
    rfn = math.hypot(rfx, rfz) or 1.0
    rfx /= rfn
    rfz /= rfn
    robot_yaw = math.atan2(rfz, rfx)

    # inflated obstacles & clearance
    obstacles = (occ_grid == 1)
    inflated = binary_dilation(obstacles, structure=np.ones((3, 3), np.uint8))
    clearance = distance_transform_edt(~inflated).astype(float) * res

    # frontier mask + safety filters
    fr_mask = _vectorized_frontiers(occ_grid)
    fr_mask &= (clearance >= min_clearance_m)

    # edge/border filter
    edge_cells = max(0, int(round(edge_margin_m / res)))
    if edge_cells > 0:
        core = np.zeros_like(fr_mask, dtype=bool)
        core[edge_cells : nz - edge_cells, edge_cells : nx - edge_cells] = True
        fr_mask &= core

    if min_unknown_component_cells > 0:
        unknown = (occ_grid == -1)
        structure = (
            np.ones((3, 3), dtype=np.uint8)
            if unknown_connectivity == 8
            else np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)
        )
        cc_labels, n_cc = label(unknown, structure=structure)
        if n_cc > 0:
            sizes = np.bincount(cc_labels.ravel())
            big = sizes >= min_unknown_component_cells
            big[0] = False
            big_unknown = big[cc_labels]
            big_unknown_d = binary_dilation(big_unknown, structure=np.ones((3, 3), np.uint8))
            fr_mask &= big_unknown_d

    cand_rc = np.argwhere(fr_mask)
    if cand_rc.size == 0:
        return [], []

    # outward-normal proxy (negative grad of unknown distance)
    unknown_dt = distance_transform_edt(~(occ_grid == -1)).astype(float)
    gy, gx = np.gradient(unknown_dt)
    eps = 1e-9

    # grid distance to robot for range filtering
    yy, xx = np.mgrid[0:nz, 0:nx]
    dist_to_robot = np.hypot(yy - rr0, xx - cc0) * res

    goals: List[np.ndarray] = []
    fwds: List[np.ndarray] = []
    bearings: List[float] = []
    rc_list: List[Tuple[int, int]] = []
    dists: List[float] = []
    front_scores: List[float] = []   # NEW

    span = float(sector_half_span_rad)

    for r, c in cand_rc:
        r = int(r); c = int(c)
        if inflated[r, c]:
            continue

        d = float(dist_to_robot[r, c])
        if d < min_goal_dist_m:
            continue

        x, z = occ._grid_to_world((r, c))

        # relative bearing theta in [-pi, pi]
        bx, bz = (x - rx), (z - rz)
        bn = math.hypot(bx, bz)
        if bn < 1e-6:
            continue
        goal_yaw = math.atan2(bz, bx)
        theta = goal_yaw - robot_yaw
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi

        # keep only within requested span
        if abs(theta) > span:
            continue

        # outward-normal forward dir at the cell
        vx = -float(gx[r, c])
        vz = -float(gy[r, c])
        nrm = math.hypot(vx, vz)
        if nrm < eps:
            vx = float(c - int(cc0))
            vz = float(r - int(rr0))
            nrm = math.hypot(vx, vz) + eps
        fwd = np.array([vx / nrm, 0.0, vz / nrm], dtype=np.float32)

        # NEW: forward preference score (1 ahead, 0 sideways, -1 behind)
        fs = math.cos(theta)
        # make it more aggressive if desired (keeps sign; squashes near-0)
        if front_score_power is not None and front_score_power >= 1.0:
            fs = math.copysign(abs(fs) ** float(front_score_power), fs)

        rc_list.append((r, c))
        goals.append(np.array([x, 0.0, z], dtype=np.float32))
        fwds.append(fwd)
        bearings.append(theta)
        dists.append(d)
        front_scores.append(fs)

    if not goals:
        return [], []

    dists_arr = np.asarray(dists, dtype=float)

    # Pass 1 (strict)
    strict_idx = np.nonzero(dists_arr <= float(max_goal_dist_m))[0].tolist()

    # Pass 2 (relaxed) if strict yields none
    if not strict_idx:
        if fallback_max_goal_dist_m is None:
            candidate_idx = list(range(len(goals)))
        else:
            candidate_idx = np.nonzero(dists_arr <= float(fallback_max_goal_dist_m))[0].tolist()
    else:
        candidate_idx = strict_idx

    if not candidate_idx:
        return [], []

    kept_idx = candidate_idx

    # Optional NMS / Poisson-disc (bias to keep forward-ish candidates)
    if nms_radius_m > 0:
        rad_cells = max(1, int(round(nms_radius_m / res)))

        r_all = np.array([rc[0] for rc in rc_list], dtype=int)
        c_all = np.array([rc[1] for rc in rc_list], dtype=int)

        r_arr = r_all[kept_idx]
        c_arr = c_all[kept_idx]

        fs_all = np.asarray(front_scores, dtype=float)
        fs_sub = fs_all[kept_idx]

        taken = np.zeros(len(kept_idx), dtype=bool)

        # NEW: instead of random order, pick higher forward-score first, with tiny jitter
        jitter = 1e-3 * rng.standard_normal(len(kept_idx))
        order = np.argsort(-(fs_sub + jitter))  # descending

        new_kept_local = []
        for local_i in order:
            if taken[local_i]:
                continue
            new_kept_local.append(local_i)

            r, c = int(r_arr[local_i]), int(c_arr[local_i])
            r0, r1 = max(0, r - rad_cells), min(nz, r + rad_cells + 1)
            c0, c1 = max(0, c - rad_cells), min(nx, c + rad_cells + 1)

            neighbor = (r_arr >= r0) & (r_arr < r1) & (c_arr >= c0) & (c_arr < c1)
            taken |= neighbor

        kept_idx = [kept_idx[li] for li in new_kept_local]

    if not kept_idx:
        return [], []

    # ---- Prefer FRONT while balancing LEFT / FORWARD / RIGHT ----
    fw = float(forward_half_width_rad)

    left_idxs, fwd_idxs, right_idxs = [], [], []
    for i in kept_idx:
        th = bearings[i]
        if -fw <= th <= fw:
            fwd_idxs.append(i)
        elif th < -fw:
            left_idxs.append(i)
        else:
            right_idxs.append(i)

    # NEW: sort each bucket by forward preference score (desc), with tiny random tie-break
    fs_all = np.asarray(front_scores, dtype=float)

    def sort_by_front(idxs: List[int]) -> List[int]:
        if not idxs:
            return idxs
        jitter = 1e-6 * rng.standard_normal(len(idxs))
        vals = fs_all[np.array(idxs, dtype=int)] + jitter
        order = np.argsort(-vals)
        return [idxs[j] for j in order]

    left_idxs  = sort_by_front(left_idxs)
    fwd_idxs   = sort_by_front(fwd_idxs)
    right_idxs = sort_by_front(right_idxs)

    # NEW: allocate more quota to FWD bucket
    forward_fraction = float(np.clip(forward_fraction, 0.0, 1.0))
    target_fwd = int(round(k * forward_fraction))
    remaining = max(0, k - target_fwd)
    target_left = remaining // 2
    target_right = remaining - target_left

    take_fwd   = min(target_fwd, len(fwd_idxs))
    take_left  = min(target_left, len(left_idxs))
    take_right = min(target_right, len(right_idxs))

    selected = []
    selected.extend(fwd_idxs[:take_fwd])
    selected.extend(left_idxs[:take_left])
    selected.extend(right_idxs[:take_right])

    # Fill the rest preferring higher front score overall (so behind tends to lose)
    if len(selected) < k:
        leftovers = []
        leftovers.extend(fwd_idxs[take_fwd:])
        leftovers.extend(left_idxs[take_left:])
        leftovers.extend(right_idxs[take_right:])
        leftovers = sort_by_front(leftovers)
        need = k - len(selected)
        selected.extend(leftovers[:need])

    if not selected:
        return [], []

    # Optional interleaving (still front-biased because each bucket is sorted by front score)
    buckets = [left_idxs[:take_left], fwd_idxs[:take_fwd], right_idxs[:take_right]]
    rr_order = []
    ptrs = [0, 0, 0]
    while len(rr_order) < min(k, len(selected)):
        progressed = False
        for b in range(3):
            if ptrs[b] < len(buckets[b]):
                rr_order.append(buckets[b][ptrs[b]])
                ptrs[b] += 1
                progressed = True
            if len(rr_order) >= k:
                break
        if not progressed:
            break

    if len(rr_order) < len(selected):
        seen = set(rr_order)
        rest = [i for i in selected if i not in seen]
        rr_order.extend(rest[: (k - len(rr_order))])

    chosen = rr_order if rr_order else selected[:k]
    goals_out    = [goals[i] for i in chosen]
    forwards_out = [fwds[i]  for i in chosen]
    return goals_out, forwards_out

@register_goal_sampling("random_free")
def sample_random_free_goals(
    occ: OccupancyMap,
    robot_pose_4x4: np.ndarray,
    *,
    k: int = 3,
    min_clearance_m: float = 0.25,     # hard floor to be considered
    softmax_temp: float = 0.25,        # lower = sharper preference for high clearance
    bias_exponent: float = 1.5,        # amplify clearance before softmax
    avoid_radius_m: float = 0.4,       # NMS radius between sampled goals
    treat_unknown_as_obstacle: bool = True,
    obstacle_inflate_cells: int = 0,   # pre-inflate obstacles before clearance
    max_tries: int = 200,              # cap on re-sampling attempts
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Random goal sampler that prefers cells far from obstacles (high clearance).

    Strategy
    --------
    1) Build a candidate mask of FREE cells (occ == 0) with clearance >= min_clearance_m.
    2) Turn clearance into sampling weights using (clearance ** bias_exponent), then softmax with
       temperature `softmax_temp` to stochastically prefer wide-open areas.
    3) Sample without replacement, applying a simple NMS in grid space so goals don't cluster.
    4) Forward vector = normalized gradient of clearance at the picked cell (points "away from walls");
       if near-flat gradient, use robot->goal as fallback.

    Returns
    -------
    goals:    list[np.ndarray(3,)] world-frame [x, y=0, z]
    forwards: list[np.ndarray(3,)] world-frame unit vectors
    """
    assert occ.occupancy is not None, "call set_point_cloud() first"
    occ_grid = occ.occupancy
    nz, nx = occ_grid.shape
    res = float(occ.resolution)

    # Robot position (world X,Z)
    rx = float(robot_pose_4x4[0, 3])
    rz = float(robot_pose_4x4[2, 3])

    # Clearance map (meters)
    clr_m = occ.compute_clearance_map(
        treat_unknown_as_obstacle=treat_unknown_as_obstacle,
        obstacle_inflate_cells=obstacle_inflate_cells,
    )

    # Candidate mask: free & above clearance floor
    free_mask = (occ_grid == 0)
    cand_mask = free_mask & (clr_m >= float(min_clearance_m))
    if not cand_mask.any():
        # Relax gracefully: fall back to "best we can find" among free cells
        cand_mask = free_mask
        if not cand_mask.any():
            return [], []  # no free cells at all

    # Weights from clearance → softmax
    eps = 1e-12
    base = np.zeros_like(clr_m, dtype=np.float64)
    base[cand_mask] = np.maximum(clr_m[cand_mask], 0.0) ** float(bias_exponent)

    if base[cand_mask].sum() <= 0.0:
        # Degenerate: uniform over candidates
        weights = np.zeros_like(base)
        weights[cand_mask] = 1.0 / cand_mask.sum()
    else:
        # Temperature-scaled softmax on candidate cells only
        t = max(float(softmax_temp), 1e-3)
        x = np.zeros_like(base)
        val = base[cand_mask] / t
        val -= val.max()  # numeric stability
        expv = np.exp(val)
        x[cand_mask] = expv / (expv.sum() + eps)
        weights = x

    # Precompute clearance gradient for forward direction (grid-space gradients)
    # np.gradient returns (d/drow, d/dcol); convert to meters by *res if needed
    dclr_dr, dclr_dc = np.gradient(clr_m)  # shape (nz, nx) each

    # NMS radius in cells
    avoid_cells = max(0, int(round(avoid_radius_m / res)))

    # Linear indices of all candidates with non-zero weight
    cand_idx = np.flatnonzero(weights > 0)
    if cand_idx.size == 0:
        # Nothing weighted? fall back to uniform over cand_mask
        cand_idx = np.flatnonzero(cand_mask)
        weights = np.zeros_like(base)
        weights.ravel()[cand_idx] = 1.0 / max(1, cand_idx.size)

    rng = getattr(occ, "rng", np.random.default_rng())

    def pick_one(current_weights_flat: np.ndarray) -> Optional[int]:
        """Sample one index using the provided flat weights (sum>0)."""
        w = current_weights_flat[current_weights_flat > 0]
        if w.size == 0:
            return None
        # draw over all cells using np.random.choice on flat vector
        idx_all = np.arange(current_weights_flat.size)
        try:
            choice = rng.choice(idx_all, p=current_weights_flat / (current_weights_flat.sum() + eps))
        except ValueError:
            return None
        return int(choice)

    def suppress_nms(wflat: np.ndarray, picked_rc: Tuple[int, int]) -> None:
        """Zero weights in a (2*avoid_cells+1)^2 window around picked cell."""
        if avoid_cells <= 0:
            return
        pr, pc = picked_rc
        r0, r1 = max(0, pr - avoid_cells), min(nz, pr + avoid_cells + 1)
        c0, c1 = max(0, pc - avoid_cells), min(nx, pc + avoid_cells + 1)
        window = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        window[...] = True
        wview = wflat.reshape(nz, nx)
        wview[r0:r1, c0:c1] = np.where(window, 0.0, wview[r0:r1, c0:c1])

    goals: List[np.ndarray] = []
    forwards: List[np.ndarray] = []

    wflat = weights.ravel().copy()
    tries = 0
    while len(goals) < k and tries < max_tries:
        tries += 1
        pick = pick_one(wflat)
        if pick is None:
            break
        r, c = divmod(pick, nx)

        # Must be free (paranoia check) and not already chosen
        if occ_grid[r, c] != 0:
            wflat[pick] = 0.0
            continue

        # World coordinates (cell center)
        x, z = occ._grid_to_world((r, c))
        goal = np.array([x, 0.0, z], dtype=np.float32)

        # Forward = gradient of clearance at (r,c) in world XZ
        gx = float(dclr_dc[r, c])   # d/dcol ~ +X
        gz = float(dclr_dr[r, c])   # d/drow ~ +Z
        norm = math.hypot(gx, gz)
        if norm < 1e-9:
            # Fallback: point from robot -> goal
            fx, fz = (x - rx), (z - rz)
            f_norm = math.hypot(fx, fz)
            fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32) if f_norm < 1e-9 \
                else np.array([fx / f_norm, 0.0, fz / f_norm], dtype=np.float32)
        else:
            fwd = np.array([gx / norm, 0.0, gz / norm], dtype=np.float32)

        goals.append(goal)
        forwards.append(fwd)

        # NMS suppression around picked cell
        suppress_nms(wflat, (r, c))
        # Also zero out the exact picked cell
        wflat[pick] = 0.0

    return goals, forwards

@register_goal_sampling("random_free_uniform")
def sample_random_free_uniform_goals(
    occ: OccupancyMap,
    robot_pose_4x4: np.ndarray,
    *,
    k: int = 3,
    avoid_radius_m: float = 0.0,        # set >0 to discourage clustering; 0 to disable
    treat_unknown_as_obstacle: bool = True,  # ignored (we only allow FREE==0 anyway)
    max_tries: int = 200,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Randomly sample k FREE cells (occ == 0) uniformly at random. No clearance bias.

    Returns
    -------
    goals:    list[np.ndarray(3,)] world-frame [x, y=0, z]
    forwards: list[np.ndarray(3,)] world-frame unit vectors (robot->goal direction)
    """
    assert occ.occupancy is not None, "call set_point_cloud() first"
    occ_grid = occ.occupancy
    nz, nx = occ_grid.shape
    res = float(occ.resolution)

    # Candidate mask: strictly FREE cells (== 0)
    free_mask = (occ_grid == 0)
    if not free_mask.any():
        return [], []

    # Flatten candidate indices
    cand_idx = np.flatnonzero(free_mask.ravel())
    rng = getattr(occ, "rng", np.random.default_rng())

    # NMS radius (in cells)
    avoid_cells = max(0, int(round(avoid_radius_m / res)))

    # Robot position for forward vector
    rx = float(robot_pose_4x4[0, 3])
    rz = float(robot_pose_4x4[2, 3])

    goals: List[np.ndarray] = []
    forwards: List[np.ndarray] = []

    # We keep a working mask for NMS suppression if requested
    work_mask = free_mask.copy()
    tries = 0

    def suppress_nms(mask: np.ndarray, r: int, c: int) -> None:
        if avoid_cells <= 0:
            return
        r0, r1 = max(0, r - avoid_cells), min(nz, r + avoid_cells + 1)
        c0, c1 = max(0, c - avoid_cells), min(nx, c + avoid_cells + 1)
        mask[r0:r1, c0:c1] = False

    while len(goals) < k and tries < max_tries:
        tries += 1

        # Refresh candidate list from current work_mask
        flat_candidates = np.flatnonzero(work_mask.ravel())
        if flat_candidates.size == 0:
            break

        pick_flat = int(rng.choice(flat_candidates))
        r, c = divmod(pick_flat, nx)

        # World coordinates (cell center)
        x, z = occ._grid_to_world((r, c))
        goal = np.array([x, 0.0, z], dtype=np.float32)

        # Forward vector: point from robot to goal (fallback to +Z if degenerate)
        fx, fz = (x - rx), (z - rz)
        norm = math.hypot(fx, fz)
        fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32) if norm < 1e-9 \
            else np.array([fx / norm, 0.0, fz / norm], dtype=np.float32)

        goals.append(goal)
        forwards.append(fwd)

        # Suppress local neighborhood to avoid clustering
        suppress_nms(work_mask, r, c)
        work_mask[r, c] = False  # also suppress the picked cell

    return goals, forwards