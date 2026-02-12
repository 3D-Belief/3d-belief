from typing import Optional, List, Tuple, Callable
import numpy as np
import heapq
import math
import skimage.draw
from collections import deque
from wm_baselines.agent.perception.occupancy import OccupancyMap
from wm_baselines.utils.planning_utils import (
    _bresenham_segment_is_free,
    _catmull_rom_spline,
    _equal_arclength_resample,
    _poses_from_xyz,
    _shortcut_path,
    _segment_is_free_grid,
    _segment_is_free_world,
    _waypoint_y,
    _rc_to_xyz_center,
)

PATH_PLANNING_REGISTRY = {}

def register_path_planning(name):
    def deco(f):
        PATH_PLANNING_REGISTRY[name] = f
        return f
    return deco

@register_path_planning("astar")
def plan_astar(
    occ: "OccupancyMap",
    start: tuple,
    goal:  tuple,
    *,
    use_height_map: bool = True,
    min_clearance_m: float = 0.25,     # hard constraint: forbid cells closer than this to any obstacle
    w_clearance: float = 3.0,         # soft cost weight: larger -> prefers wider halls
    soft_radius_m: float = 0.5,       # soft influence radius; >0 to shape how far the penalty reaches
    unknown_penalty: float = 0.25,    # small penalty to step into unknown (-1) cells (0 to disable)
    forbid_corner_cut: bool = True,   # prevent diagonals that "squeeze" between two blocked cells
) -> "Optional[List[np.ndarray]]":
    """
    A* on X-Z grid with clearance-aware costs:
      - Hard: min_clearance_m forbids cells within that distance of obstacles.
      - Soft: additional cost encourages staying farther from obstacles.

    Returns waypoints [(x,y,z), ...] at cell centers; ensures >=2 points.
    """
    assert occ.occupancy is not None, "call set_point_cloud() first"
    grid = occ.occupancy  # shape (nz, nx), 1=occupied, 0=free, -1=unknown
    nz, nx = grid.shape
    res = float(occ.resolution)

    def in_bounds(rc):
        r, c = rc
        return (0 <= r < nz) and (0 <= c < nx)

    # obstacles are True at occupied cells
    obstacles = (grid == 1)

    def edt_distance_m(obstacle_mask: np.ndarray) -> np.ndarray:
        # prefer scipy's EDT (euclidean, fast). Fallback to BFS (Manhattan approx).
        try:
            from scipy import ndimage as ndi
            # distance in pixels (cells); convert to meters
            dist_cells = ndi.distance_transform_edt(~obstacle_mask)
            return dist_cells * res
        except Exception:
            # BFS from all obstacles simultaneously to get grid (Manhattan) distance in cells
            INF = 10**9
            dist = np.full(obstacle_mask.shape, INF, dtype=np.int32)
            q = deque()
            # seed all obstacle cells with distance 0
            occ_idx = np.argwhere(obstacle_mask)
            for r, c in occ_idx:
                dist[r, c] = 0
                q.append((r, c))
            nbrs4 = [(1,0),(-1,0),(0,1),(0,-1)]
            while q:
                r, c = q.popleft()
                d = dist[r, c]
                nd = d + 1
                for dr, dc in nbrs4:
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < obstacle_mask.shape[0] and 0 <= cc < obstacle_mask.shape[1]:
                        if dist[rr, cc] > nd:
                            dist[rr, cc] = nd
                            q.append((rr, cc))
            # Convert cells->meters; obstacles themselves are 0m
            dist[dist == INF] = max(nz, nx) * 2  # far away -> big number
            return dist.astype(np.float32) * res

    dist_to_obs_m = edt_distance_m(obstacles)

    # Apply hard clearance mask if requested
    if min_clearance_m > 0:
        unsafe = dist_to_obs_m < min_clearance_m
    else:
        unsafe = np.zeros_like(obstacles, dtype=bool)

    # indexing helpers
    s_idx = occ._world_to_grid((start[0], start[2]))
    g_idx = occ._world_to_grid((goal [0], goal [2]))

    if not in_bounds(s_idx) or not in_bounds(g_idx):
        print("start outside grid bounds" if not in_bounds(s_idx) else "goal outside grid bounds")
        return None

    # forbid start/goal if occupied or fails hard-clearance
    def cell_forbidden(rc):
        r, c = rc
        if grid[r, c] == 1:
            return True
        if unsafe[r, c]:
            return True
        return False

    # if cell_forbidden(s_idx) or cell_forbidden(g_idx):
    #     print("start or goal is blocked by occupancy or hard-clearance")
    #     return None

    if cell_forbidden(g_idx):
        print("goal is blocked by occupancy or hard-clearance")
        return None

    # admissible heuristic (Euclidean on grid indices)
    def heuristic(a, b):
        return np.hypot(b[0]-a[0], b[1]-a[1])

    # 8-connected neighbors, with optional corner-cut check
    def neighbors(rc):
        r, c = rc
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r+dr, c+dc
                if not in_bounds((rr, cc)):
                    continue
                if cell_forbidden((rr, cc)):
                    continue
                if forbid_corner_cut and dr != 0 and dc != 0:
                    # don't allow squeezing between two blocked orthogonals
                    if cell_forbidden((r, c+dc)) or cell_forbidden((r+dr, c)):
                        continue
                yield (rr, cc)

    # soft clearance penalty function (bigger near obstacles; ~0 when far)
    # smooth inverse-square inside a soft radius; 0 outside
    def clearance_penalty_m(dist_m: float) -> float:
        if soft_radius_m <= 0:
            return 0.0
        if dist_m >= soft_radius_m:
            return 0.0
        # map [0, soft_radius] -> [big, 0], smoothly; add small eps to avoid div by 0
        eps = 1e-3
        x = max(dist_m, eps) / soft_radius_m  # (0,1]
        # 1/x^2 - 1 -> [inf, 0]; clamp for numerical safety
        val = (1.0 / (x * x)) - 1.0
        # optional mild cap to avoid huge spikes (still prefers wider space)
        return float(min(val, 50.0))

    # movement cost (base euclidean in cells) + clearance & unknown penalties
    def step_cost(a, b):
        # base metric: diagonal ~1.414, straight ~1.0 (in cell units)
        base = np.hypot(b[0]-a[0], b[1]-a[1])
        r, c = b
        pen = 0.0
        # unknown cells get a small additive penalty (encourages known free space)
        if grid[r, c] == -1 and unknown_penalty > 0:
            pen += unknown_penalty
        # soft clearance
        if w_clearance > 0:
            pen += w_clearance * clearance_penalty_m(dist_to_obs_m[r, c])
        return base + pen

    came_from = {}
    g_score = {s_idx: 0.0}
    f_score = {s_idx: heuristic(s_idx, g_idx)}
    open_heap = [(f_score[s_idx], s_idx)]
    visited = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == g_idx:
            # reconstruct path (grid)
            path_rc = [current]
            while current in came_from:
                current = came_from[current]
                path_rc.append(current)
            path_rc.reverse()

            # guarantee >= 2 cells
            if len(path_rc) == 1:
                start_cell = path_rc[0]
                cand = [n for n in neighbors(start_cell)]
                if cand:
                    best = min(cand, key=lambda n: heuristic(n, g_idx))
                    path_rc.append(best)
                else:
                    path_rc.append(start_cell)

            # grid -> world waypoints at cell centers
            waypoints = []
            for row, col in path_rc:
                x, z = occ._grid_to_world((row, col))
                # if use_height_map and (occ.height_map is not None):
                #     y = float(occ.height_map[row, col])
                # else:
                #     y = 0.0
                y = 0.0
                waypoints.append(np.array([x, y, z], dtype=np.float32))
            return waypoints

        for nbr in neighbors(current):
            tentative_g = g_score[current] + step_cost(current, nbr)
            if tentative_g < g_score.get(nbr, np.inf):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                f_score[nbr] = tentative_g + heuristic(nbr, g_idx)
                heapq.heappush(open_heap, (f_score[nbr], nbr))

    return None

@register_path_planning("rrt_star")
def plan_rrt_star(
    occ: OccupancyMap,
    start: Tuple[float, float, float],
    goal:  Tuple[float, float, float],
    *,
    max_iterations: int = 1000,
    step_size: float = 0.5,
    goal_sample_rate: float = 0.1,
    search_radius: float = 1.0,
    use_height_map: bool = False,
) -> Optional[List[np.ndarray]]:
    """
    RRT* in continuous X-Z space, using occ.occupancy for collision checks.
    Returns waypoints [(x,y,z), ...].
    """
    assert occ.occupancy is not None, "call set_point_cloud() first"

    # Extract 2D positions (x,z)
    start_2d = (start[0], start[2])
    goal_2d  = (goal [0], goal [2])

    # Check bounds & occupancy
    s_idx = occ._world_to_grid(start_2d)
    g_idx = occ._world_to_grid(goal_2d)
    if not occ._in_bounds(s_idx) or not occ._in_bounds(g_idx):
        if not occ._in_bounds(s_idx):
            print("start outside grid bounds")
        else:
            print("goal outside grid bounds")
        return None
    if occ.occupancy[s_idx] == 1 or occ.occupancy[g_idx] == 1:
        print("start or goal is in occupied cell")
        return None

    class Node:
        __slots__ = ("x", "z", "parent", "cost")
        def __init__(self, x: float, z: float):
            self.x = x
            self.z = z
            self.parent: Optional["Node"] = None
            self.cost: float = 0.0

    def calc_dist(n1: Node, n2: Node) -> float:
        return math.hypot(n2.x - n1.x, n2.z - n1.z)

    def steer(from_node: Node, to_node: Node, step: float) -> Node:
        d = calc_dist(from_node, to_node)
        if d <= step:
            return Node(to_node.x, to_node.z)
        theta = math.atan2(to_node.z - from_node.z, to_node.x - from_node.x)
        return Node(from_node.x + step * math.cos(theta),
                    from_node.z + step * math.sin(theta))

    def world_segment_is_free(p1: Tuple[float,float], p2: Tuple[float,float]) -> bool:
        (r1, c1) = occ._world_to_grid(p1)
        (r2, c2) = occ._world_to_grid(p2)
        for r, c in zip(*skimage.draw.line(r1, c1, r2, c2)):
            if not occ._in_bounds((r, c)) or occ.occupancy[r, c] == 1:
                return False
        return True

    def nearest_index(nodes: List[Node], q: Node) -> int:
        d2 = [(n.x - q.x) ** 2 + (n.z - q.z) ** 2 for n in nodes]
        return int(np.argmin(d2))

    def near_indices(nodes: List[Node], q: Node, radius: float) -> List[int]:
        out = []
        for i, n in enumerate(nodes):
            if calc_dist(n, q) <= radius:
                out.append(i)
        return out

    # bounds for sampling
    x_min, x_max = occ.x_min, occ.x_min + occ.nx * occ.resolution
    z_min, z_max = occ.z_min, occ.z_min + occ.nz * occ.resolution

    start_node = Node(*start_2d)
    nodes: List[Node] = [start_node]
    goal_node: Optional[Node] = None

    for i in range(max_iterations):
        # sample
        if np.random.rand() < goal_sample_rate:
            rnd = Node(*goal_2d)
        else:
            rnd = Node(np.random.uniform(x_min, x_max),
                       np.random.uniform(z_min, z_max))

        # nearest & steer
        ni = nearest_index(nodes, rnd)
        new_node = steer(nodes[ni], rnd, step_size)

        # collision check to new_node
        if not world_segment_is_free((nodes[ni].x, nodes[ni].z), (new_node.x, new_node.z)):
            continue

        # choose parent among near nodes
        near = near_indices(nodes, new_node, search_radius)
        parent = nodes[ni]
        min_cost = parent.cost + calc_dist(parent, new_node)
        for j in near:
            n = nodes[j]
            if world_segment_is_free((n.x, n.z), (new_node.x, new_node.z)):
                c = n.cost + calc_dist(n, new_node)
                if c < min_cost:
                    min_cost, parent = c, n
        new_node.parent = parent
        new_node.cost = min_cost
        nodes.append(new_node)

        # rewire
        for j in near:
            n = nodes[j]
            new_cost = new_node.cost + calc_dist(new_node, n)
            if new_cost < n.cost and world_segment_is_free((new_node.x, new_node.z), (n.x, n.z)):
                n.parent = new_node
                n.cost = new_cost

        # try connect to goal
        tmp_goal = Node(*goal_2d)
        if calc_dist(new_node, tmp_goal) <= step_size:
            if world_segment_is_free((new_node.x, new_node.z), (tmp_goal.x, tmp_goal.z)):
                tmp_goal.parent = new_node
                tmp_goal.cost = new_node.cost + calc_dist(new_node, tmp_goal)
                goal_node = tmp_goal
                print(f"Found path to goal after {i+1} iterations")
                break

    if goal_node is None:
        print("Could not find path to goal")
        return None

    # extract path (x,z)
    path_xz: List[Tuple[float,float]] = []
    cur = goal_node
    while cur is not None:
        path_xz.append((cur.x, cur.z))
        cur = cur.parent
    path_xz.reverse()

    # to 3D
    waypoints: List[np.ndarray] = []
    for x, z in path_xz:
        if use_height_map and (occ.height_map is not None):
            r, c = occ._world_to_grid((x, z))
            y = float(occ.height_map[r, c])
        else:
            y = 0.0
        waypoints.append(np.array([x, y, z], dtype=np.float32))
    return waypoints

@register_path_planning("random_walk")
def plan_random_walk(
    occ: OccupancyMap,
    start: Tuple[float, float, float],
    goal:  Tuple[float, float, float],
    *,
    steps: int = 20,
    step_m: float = 0.4,
    goal_bias: float = 0.75,          # 0..1, bias direction toward goal
    goal_tol_m: float = 0.5,
    use_height_map: bool = False,
    tries: int = 60,
    rng: Optional[np.random.Generator] = None,
    allow_unknown_as_free: bool = True,   # treat unknown as traversable
    snap_to_free: bool = True,            # snap occupied/OOB start/goal to nearest traversable
    max_snap_m: float = 2.0,              # maximum snap distance (meters)
) -> Optional[List[np.ndarray]]:
    """
    Random-walk fallback planner that *always tries* to produce a path:
    - Snaps invalid (occupied/OOB) start/goal to nearest traversable cell (<= max_snap_m).
    - Walks with goal-biased random steps and short sidestep recovery.
    - If the walk fails, returns a minimal two-point fallback (snapped start -> snapped goal).
    """
    assert occ.occupancy is not None, "call set_point_cloud() first"
    rng = rng or np.random.default_rng()

    def _is_traversable_mask():
        occv = occ.occupancy
        if allow_unknown_as_free:
            return (occv != 1)            # free (0) or unknown (-1)
        else:
            return (occv == 0)            # only free

    def _waypoint_y_from_rc(r: int, c: int) -> float:
        if use_height_map and (occ.height_map is not None):
            return float(occ.height_map[r, c])
        return 0.0

    def _nearest_traversable_rc(target_xz: Tuple[float,float]) -> Optional[Tuple[int,int,float]]:
        """Return (r,c,dist_m) of nearest traversable cell to target_xz, or None."""
        tr, tc = occ._world_to_grid(target_xz)
        trav = _is_traversable_mask()
        # If the grid is entirely blocked, bail.
        if not trav.any():
            return None
        # Distance in *cells* to the nearest traversable from every cell:
        # EDT computes distance to nearest False; so invert:
        # We want distance to nearest True -> run EDT on ~trav
        import numpy as np
        from scipy.ndimage import distance_transform_edt
        dist_cells = distance_transform_edt(~trav)
        # Clamp (tr,tc) to bounds in case target_xz was OOB
        tr = int(np.clip(tr, 0, occ.nz - 1))
        tc = int(np.clip(tc, 0, occ.nx - 1))
        # nearest traversable index around (tr,tc):
        # Walk a small window around to find the actual nearest True (optional but cheap)
        # Or project by gradient—simpler: argmin of (dist + manhattan around).
        # We'll just take the closest by distance transform then refine in a 3x3 ring.
        best_r, best_c = tr, tc
        best_d = float(dist_cells[tr, tc])
        if not trav[tr, tc]:
            # find nearest True by looking for minimal distance over the whole map (rarely needed)
            idx = np.argmin(dist_cells + (~trav) * 1e6)  # force pick only where trav=True
            r0, c0 = divmod(int(idx), occ.nx)
            best_r, best_c = r0, c0
            best_d = float(dist_cells[r0, c0])
        return best_r, best_c, best_d * float(occ.resolution)

    def _xz_from_rc(rc: Tuple[int,int]) -> np.ndarray:
        x, z = occ._grid_to_world(rc)
        return np.array([x, z], dtype=np.float32)

    def _segment_is_free_world(occ: OccupancyMap, p1: Tuple[float,float], p2: Tuple[float,float]) -> bool:
        # Bresenham through occupancy (treat unknown per setting)
        (r1, c1) = occ._world_to_grid(p1)
        (r2, c2) = occ._world_to_grid(p2)
        rr, cc = skimage.draw.line(r1, c1, r2, c2)
        trav = _is_traversable_mask()
        nz, nx = occ.nz, occ.nx
        for r, c in zip(rr, cc):
            if not (0 <= r < nz and 0 <= c < nx):
                return False
            if not trav[r, c]:
                return False
        return True

    # Original inputs → XZ (float32)
    start_xz = np.array([start[0], start[2]], dtype=np.float32)
    goal_xz  = np.array([goal [0], goal [2]], dtype=np.float32)

    # Snap start/goal to traversable if needed
    s_rc = occ._world_to_grid(tuple(start_xz))
    g_rc = occ._world_to_grid(tuple(goal_xz))
    trav = _is_traversable_mask()

    # If out-of-bounds or not traversable, snap
    def _snap_if_needed(xz: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int], bool]:
        rc = occ._world_to_grid(tuple(xz))
        inb = occ._in_bounds(rc)
        ok  = inb and trav[rc]
        if ok or not snap_to_free:
            return xz, rc, ok
        near = _nearest_traversable_rc(tuple(xz))
        if near is None:
            return xz, rc, False
        nr, nc, d_m = near
        if d_m <= max_snap_m:
            snapped = _xz_from_rc((nr, nc))
            return snapped, (nr, nc), True
        return xz, rc, False

    start_xz, s_rc, s_ok = _snap_if_needed(start_xz)
    goal_xz,  g_rc, g_ok = _snap_if_needed(goal_xz)

    # If *everything* is blocked (no traversable cells), return a trivial two-point fallback in place.
    if not trav.any():
        p0 = np.array([start[0], 0.0, start[2]], dtype=np.float32)
        p1 = np.array([goal [0], 0.0, goal [2]], dtype=np.float32)
        return [p0, p1]

    # If either start/goal still invalid after snap and we're strict, allow walking anyway:
    # We’ll still try random-walk; segments will be checked with _segment_is_free_world.

    # Bounds in world
    x_min, x_max = occ.x_min, occ.x_min + occ.nx * occ.resolution
    z_min, z_max = occ.z_min, occ.z_min + occ.nz * occ.resolution

    for _ in range(tries):
        path_xz = [start_xz.copy()]
        for _step in range(steps):
            cur = path_xz[-1]
            to_goal = goal_xz - cur
            d = np.linalg.norm(to_goal) + 1e-9
            dir_goal = to_goal / d

            # biased random direction
            theta = rng.uniform(-math.pi, math.pi)
            dir_rand = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
            direction = goal_bias * dir_goal + (1.0 - goal_bias) * dir_rand
            direction /= (np.linalg.norm(direction) + 1e-9)

            nxt = cur + step_m * direction
            nxt[0] = np.clip(nxt[0], x_min, x_max)
            nxt[1] = np.clip(nxt[1], z_min, z_max)

            if _segment_is_free_world(occ, tuple(cur), tuple(nxt)):
                path_xz.append(nxt)
                if np.linalg.norm(nxt - goal_xz) <= goal_tol_m:
                    path_xz.append(goal_xz.copy())
                    break
            else:
                # small sidestep recovery
                for delta in (+math.pi/6, -math.pi/6, +math.pi/3, -math.pi/3):
                    c,s = math.cos(delta), math.sin(delta)
                    rot = np.array([[c,-s],[s,c]], dtype=np.float32)
                    alt_dir = rot @ direction
                    alt_dir /= (np.linalg.norm(alt_dir) + 1e-9)
                    alt_nxt = cur + step_m * alt_dir
                    alt_nxt[0] = np.clip(alt_nxt[0], x_min, x_max)
                    alt_nxt[1] = np.clip(alt_nxt[1], z_min, z_max)
                    if _segment_is_free_world(occ, tuple(cur), tuple(alt_nxt)):
                        path_xz.append(alt_nxt)
                        break
                else:
                    # stuck this attempt
                    path_xz = []
                    break

        if len(path_xz) >= 2:
            # produce 3D waypoints
            waypoints: List[np.ndarray] = []
            for x, z in path_xz:
                r, c = occ._world_to_grid((float(x), float(z)))
                if not occ._in_bounds((r, c)):
                    continue
                y = _waypoint_y_from_rc(r, c)
                waypoints.append(np.array([float(x), y, float(z)], dtype=np.float32))
            if len(waypoints) >= 2:
                return waypoints

    # ------------ last-resort fallback ------------
    # If the random walk failed, still output a simple two-point path
    # (snapped start -> snapped goal). This guarantees *some* path.
    s_r, s_c = occ._world_to_grid(tuple(start_xz))
    g_r, g_c = occ._world_to_grid(tuple(goal_xz))
    y0 = _waypoint_y_from_rc(int(s_r), int(s_c)) if occ._in_bounds((s_r, s_c)) else 0.0
    y1 = _waypoint_y_from_rc(int(g_r), int(g_c)) if occ._in_bounds((g_r, g_c)) else 0.0
    p0 = np.array([float(start_xz[0]), y0, float(start_xz[1])], dtype=np.float32)
    p1 = np.array([float(goal_xz [0]), y1, float(goal_xz [1])], dtype=np.float32)
    return [p0, p1]

def path_to_trajectory(
    waypoints: List[np.ndarray],
    *,
    ds: float = 0.05,                 # meters between output poses
    spline_alpha: float = 0.5,        # 0.5 = centripetal CR spline
    samples_per_seg: int = 20,
    occ: Optional[OccupancyMap] = None,
    shortcut_passes: int = 2,
    ref_pose: Optional[np.ndarray] = None,
    interpolate: bool = False,
    goal_forward: Optional[np.ndarray] = None,
    tail_blend: int = 3                          # blend last N-1 yaws toward goal yaw
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Convert discrete (x,y,z) waypoints into a trajectory.
    - interpolate=False: use raw waypoints only.
    - interpolate=True : optional shortcut -> Catmull-Rom -> arc-length resample.
    If goal_forward is given, the last pose keeps the goal position AND its yaw
    is set to align with goal_forward (in XZ); the previous few yaws are blended.
    Returns: (poses_4x4_list, xyz_samples_list)
    """
    if waypoints is None or len(waypoints) < 2:
        return None

    P = np.asarray(waypoints, dtype=float)  # (N,3)

    if not interpolate:
        X = P  # no shortcut, no spline, no resample
    else:
        # Optional shortcut (keeps endpoints)
        is_free = None
        if occ is not None:
            is_free = lambda a, b: _bresenham_segment_is_free(occ, a, b)
        P2 = _shortcut_path(P, is_free=is_free, max_passes=shortcut_passes)

        # Smooth with Catmull–Rom
        C = _catmull_rom_spline(P2, samples_per_seg=samples_per_seg, alpha=spline_alpha)

        # Re-sample by arc length so pose spacing is ~ds
        X = _equal_arclength_resample(C, ds=ds)

        # Ensure the last position is the exact goal position
        X[-1] = P[-1]

    # Build tangents (for yaw) in XZ
    tangents = np.zeros_like(X)
    tangents[1:-1] = X[2:] - X[:-2]
    tangents[0]    = X[1] - X[0]
    tangents[-1]   = X[-1] - X[-2]

    # If goal_forward is provided, override the last yaw (and gently blend previous yaws)
    if goal_forward is not None:
        gf = np.asarray(goal_forward, dtype=float)
        # project to XZ plane
        gf_xz = np.array([gf[0], 0.0, gf[2]], dtype=float)
        n = np.hypot(gf_xz[0], gf_xz[2])
        if n > 1e-12:
            gf_xz /= n
            goal_yaw = math.atan2(gf_xz[2], gf_xz[0])

            # replace last tangent direction to match goal_forward
            tangents[-1] = np.array([gf_xz[0], 0.0, gf_xz[2]], dtype=float)

            # optional: blend previous few yaws toward goal yaw to reduce snap
            # compute current yaws for last tail_blend elements
            K = min(tail_blend, len(X)-1)
            for i in range(1, K+1):  # i=1 affects last-1, i=K affects farther back
                j = -1 - i
                t = tangents[j]
                if np.linalg.norm(t[: :2]) < 1e-12:  # guard; use forward diff if degenerate
                    t = X[min(j+1, len(X)-1)] - X[max(j-1, 0)]
                yaw_j = math.atan2(t[2], t[0])

                # linear blend in angle space (shortest angular distance)
                w = i / (K + 1.0)  # small weight near the end, larger nearer the last
                # unwrap to avoid jumps
                dyaw = (goal_yaw - yaw_j + math.pi) % (2*math.pi) - math.pi
                yaw_blend = yaw_j + w * dyaw

                # convert back to a unit XZ direction
                tangents[j] = np.array([math.cos(yaw_blend), 0.0, math.sin(yaw_blend)], dtype=float)

    # Now convert to poses; last orientation is already enforced via tangents
    poses = []
    for p, t in zip(X, tangents):
        if np.linalg.norm(t) < 1e-9:
            yaw = 0.0
        else:
            yaw = math.atan2(t[2], t[0])
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
            T = ref_pose @ T
        poses.append(T)

    return poses, [p for p in X]