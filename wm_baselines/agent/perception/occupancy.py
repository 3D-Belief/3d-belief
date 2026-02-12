import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import distance_transform_edt, binary_dilation
from pathlib import Path
import math
from typing import Tuple, List, Optional, Union
from wm_baselines.utils.common_utils import with_timing

class OccupancyMap:
    def __init__(
        self,
        resolution: float = 0.1,
        obstacle_height_thresh: float = 0.2,
        ceiling_height: float = 1.0,
        max_range: float = 2.0,
        free_overrides_occupied: bool = True,
    ):
        """
        Y-up, Z-forward convention:
          X: right
          Y: up (height)
          Z: forward
        Projects the 3D point cloud onto the X-Z plane,
        building a height-map of Y values and an occupancy grid.
        """
        self.resolution = resolution
        self.obstacle_height_thresh = obstacle_height_thresh
        self.ceiling_height = ceiling_height
        self.max_range = max_range
        self.free_overrides_occupied = free_overrides_occupied

        # set by set_point_cloud()
        self.height_map: Optional[np.ndarray] = None     # shape: (nz, nx) = (rows, cols)
        self.occupancy: Optional[np.ndarray] = None     # same shape, 1=occupied, 0=free, -1 unknown
        self.x_min: float = 0.0
        self.z_min: float = 0.0
        self.nx: int = 0
        self.nz: int = 0

    @with_timing
    def integrate(
        self,
        pcd: np.ndarray,
        position: Tuple[float, float, float],
        rotation: Optional[Union[np.ndarray, float]] = None,
        mark_free: bool = True,
        intrinsics: Optional[np.ndarray] = None,
        prev_free_map: Optional["OccupancyMap"] = None,
    ) -> None:
        """
        Integrate a new point cloud observation into the occupancy map.

        If prev_free_map is provided, we will import only its FREE space (0) into the
        current observation's map, without importing obstacles (1) or unknowns (-1).
        Obstacles in the current map are never overwritten by prev_free_map.
        """
        # Filter out invalid points
        pcd = pcd[pcd[:, 1] <= self.ceiling_height]
        # Extract yaw from rotation parameter
        if rotation is None:
            yaw = 0.0
        elif isinstance(rotation, (int, float)):
            yaw = float(rotation)
        elif isinstance(rotation, np.ndarray):
            if rotation.shape == (3, 3):
                forward = rotation @ np.array([0.0, 0.0, 1.0])
                yaw = -math.atan2(forward[0], forward[2])
                yaw = yaw % (2*math.pi)
            else:
                raise ValueError("Rotation matrix must be 3x3")
        else:
            raise TypeError("Rotation must be a float (yaw) or 3x3 numpy array")

        is_first_frame = self.occupancy is None

        if is_first_frame:
            # First frame: initialize directly
            pcd = np.vstack([pcd, [0, 0, 0]])  # include origin
            self.set_point_cloud(
                pcd=pcd,
                mark_free=mark_free,
                sensor_origin=position,
                yaw=yaw,
                intrinsics=intrinsics,
                free_radius=0.5,
            )
            # Import FREE space from previous map (if any)
            if prev_free_map is not None:
                self.merge_free_from(prev_free_map)
        else:
            # Build a temp map from this observation
            temp_map = OccupancyMap(
                resolution=self.resolution,
                obstacle_height_thresh=self.obstacle_height_thresh,
                ceiling_height=self.ceiling_height,
                max_range=self.max_range,
                free_overrides_occupied=self.free_overrides_occupied,
            )
            temp_map.set_point_cloud(
                pcd=pcd,
                mark_free=mark_free,
                sensor_origin=position,
                yaw=yaw,
                intrinsics=intrinsics
            )
            # Import FREE space from previous map (if any) into the temp map
            if prev_free_map is not None:
                temp_map.merge_free_from(prev_free_map)

            # Now merge temp_map into the global map with your normal policy
            self.merge(temp_map)

    def set_point_cloud(
        self, 
        pcd: np.ndarray, 
        mark_free: bool=True, 
        sensor_origin: Optional[Tuple[float, float, float]]=(0.0, 0.0, 0.0),
        yaw: Optional[float]=0.0,
        free_radius: Optional[float]=None,
        intrinsics: Optional[np.ndarray]=None
    ):
        """
        Build:
          - self.cell_counts: (#points in each X-Z cell)
          - self.height_map:   (max Y per cell)
          - self.occupancy:    (binary)
        using a 2D histogram for counts.
        Optionally clear occupancy within free_radius of sensor_origin.
        """
        # pcd = pcd.copy()
        # pcd[:, 1] = -pcd[:, 1]  # flip Y to match Y-up convention
        assert pcd.ndim == 2 and pcd.shape[1] == 3
        xs, ys, zs = pcd[:,0], pcd[:,1], pcd[:,2]

        # grid bounds
        self.x_min, x_max = float(xs.min()), float(xs.max())
        self.z_min, z_max = float(zs.min()), float(zs.max())
        self.nx = int(np.ceil((x_max - self.x_min) / self.resolution)) + 1
        self.nz = int(np.ceil((z_max - self.z_min) / self.resolution)) + 1

        # get 2D counts via histogram2d
        counts, z_edges, x_edges = np.histogram2d(
            zs, xs,
            bins=[self.nz, self.nx],
            range=[
                [self.z_min, self.z_min + self.nz*self.resolution],
                [self.x_min, self.x_min + self.nx*self.resolution],
            ],
        )
        self.cell_counts = counts.astype(np.int32)

        # build height_map (max Y per cell)
        hmap = np.full((self.nz, self.nx), -np.inf, dtype=np.float32)
        col_idxs = np.clip(((xs - self.x_min) / self.resolution).astype(int), 0, self.nx - 1)
        row_idxs = np.clip(((zs - self.z_min) / self.resolution).astype(int), 0, self.nz - 1)
        for (r, c, y) in zip(row_idxs, col_idxs, ys):
            if y > hmap[r, c]:
                hmap[r, c] = y
        y_min = float(ys.min())
        hmap[hmap == -np.inf] = y_min

        self.height_map = hmap
        # initial occupancy: obstacle heights
        occ = (hmap > self.obstacle_height_thresh).astype(np.uint8)

        if free_radius is not None:
            ox, oy, oz = sensor_origin
            # compute cell-center coordinates
            xs_centers = self.x_min + (np.arange(self.nx) + 0.5) * self.resolution
            zs_centers = self.z_min + (np.arange(self.nz) + 0.5) * self.resolution
            # meshgrid
            Zc, Xc = np.meshgrid(zs_centers, xs_centers, indexing='xy')
            # distance map in X-Z plane
            dist = np.sqrt((Xc - ox)**2 + (Zc - oz)**2)
            occ[dist.T <= free_radius] = 0

        self.occupancy = occ

        if mark_free:
            self.mark_free_space_with_rays(sensor_origin, yaw=yaw, intrinsics=intrinsics, max_range=self.max_range)
        
        self.apply_clearance(
            min_clearance_m=0.05,
            treat_unknown_as_obstacle=False,
            overwrite_unknown=False,
        )

    def mark_free_space_with_rays(
        self,
        sensor_origin: Tuple[float, float, float],
        max_range: float = 5.0,
        yaw: float = 0.0,
        intrinsics: Optional[np.ndarray] = None,
        *,
        include_diagonals: bool = True,      # 8-neighborhood if True, else 4-neighborhood
        max_shifts: int = 64,                # cap on #shifted ray casts to avoid explosion
        angles_per_ring: int = 5,            # for sub-cell ring shifts
        skip_blocked_origins: bool = True,   # skip shifts whose origin is not free
        nudge_blocked_origins: bool = False, # try to nudge blocked origins to nearest free cell
        max_nudge_m: float = 0.5,            # max nudge radius (meters)
        treat_unknown_origin_as_blocked: bool = True,  # treat -1 as blocked for origin check
    ):
        """
        Multi-cast ray marking:
        - Cast from the given origin and from small spatial shifts (one-cell and sub-cell).
        - Mark a cell as FREE (0) only if *all* casts mark it free.
        - Obstacles (1) preserved; others remain unknown (-1).

        Args:
            sensor_origin: (x, y, z) world coordinates of the sensor
            max_range: maximum distance (m) to mark free
            yaw: camera yaw (rad) in world frame as view center
            intrinsics: 3x3 intrinsics K (to derive horizontal FOV if given)
            include_diagonals: include diagonal one-cell shifts if True
            max_shifts: upper bound on total number of shifted casts
            angles_per_ring: #sub-cell samples on the small ring
            skip_blocked_origins: if True, skip casts whose origin is occupied/unknown
            nudge_blocked_origins: if True, try nudging blocked origins to nearest free cell
            max_nudge_m: maximum nudge distance (meters)
            treat_unknown_origin_as_blocked: consider unknown (-1) as blocked for origin check
        """
        assert self.occupancy is not None, "Call set_point_cloud() first"

        # Base occupancy clone: keep obstacles; everything else unknown to start
        new_occ = np.full_like(self.occupancy, -1, dtype=np.int8)
        new_occ[self.occupancy == 1] = 1

        # --- helpers -------------------------------------------------------------
        def is_origin_traversable(ox: float, oz: float) -> bool:
            r, c = self._world_to_grid((ox, oz))
            v = self.occupancy[r, c]
            if v == 1:
                return False
            if treat_unknown_origin_as_blocked and v == -1:
                return False
            return True

        def nudge_to_nearest_free(ox: float, oz: float) -> Tuple[float, float]:
            """
            Move (ox, oz) to the nearest FREE cell center within max_nudge_m; return original if none.
            """
            # quick accept if already free
            r, c = self._world_to_grid((ox, oz))
            if self.occupancy[r, c] == 0:
                return ox, oz

            # clearance map treats unknown as obstacles to be conservative
            clr = self.compute_clearance_map(treat_unknown_as_obstacle=True)
            res = float(self.resolution)
            radius_cells = max(1, int(max_nudge_m / res))

            r0, r1 = max(0, r - radius_cells), min(self.nz, r + radius_cells + 1)
            c0, c1 = max(0, c - radius_cells), min(self.nx, c + radius_cells + 1)

            occ_sub = self.occupancy[r0:r1, c0:c1]
            clr_sub = clr[r0:r1, c0:c1]

            # mask of truly free cells
            free_mask = (occ_sub == 0)
            if not free_mask.any():
                return ox, oz  # give up

            # choose the free cell with largest clearance (ties arbitrary)
            rr, cc = np.unravel_index(np.argmax(clr_sub * free_mask), clr_sub.shape)
            rf, cf = r0 + rr, c0 + cc
            xw, zw = self._grid_to_world((rf, cf))
            return xw, zw

        # horizontal FOV from intrinsics (if provided)
        if intrinsics is not None:
            fx = float(intrinsics[0, 0])
            cx = float(intrinsics[0, 2])
            w = int(round(2 * cx))
            h_fov_rad = 2.0 * math.atan(w / (2.0 * fx))
            fov_rad = h_fov_rad
        else:
            fov_rad = None

        r_max = int(math.ceil(max_range / float(self.resolution)))

        # angle samples
        if fov_rad is None:
            N = max(int(math.ceil(2.0 * math.pi * max_range / self.resolution)), 360)
            angles = np.linspace(0.0, 2.0 * math.pi, N, endpoint=False)
        else:
            N = max(int(math.ceil(fov_rad * max_range / self.resolution)), 30)
            angles = np.linspace(yaw - fov_rad / 2.0, yaw + fov_rad / 2.0, N, endpoint=False)

        # rotate so 0 rad points +Z (match X-right, Z-forward)
        angles = angles + math.pi / 2.0
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        ds = np.arange(1, r_max + 1) * float(self.resolution)

        # build one-pixel (one-cell) shift set
        cell = float(self.resolution)
        base_shifts = [
            (0.0, 0.0),
            ( cell, 0.0), (-cell, 0.0),
            (0.0,  cell), ( 0.0, -cell),
        ]
        if include_diagonals:
            base_shifts += [
                ( cell,  cell), ( cell, -cell),
                (-cell,  cell), (-cell, -cell),
            ]

        # small ring of sub-cell shifts (in *cells*)
        ring_radii_cells = [0.1]
        ring_shifts = []
        for r_cells in ring_radii_cells:
            r = r_cells * cell
            thetas = np.linspace(0.0, 2.0 * math.pi, angles_per_ring, endpoint=False)
            ring_shifts.extend([(float(r * math.cos(t)), float(r * math.sin(t))) for t in thetas])

        # optional square grid (disabled by default)
        use_square_grid = False
        grid_extent_cells = 1.0
        grid_step_cells = 0.5
        grid_shifts = []
        if use_square_grid:
            coords = np.arange(-grid_extent_cells, grid_extent_cells + 1e-6, grid_step_cells)
            for gx in coords:
                for gz in coords:
                    grid_shifts.append((float(gx * cell), float(gz * cell)))

        # merge, deduplicate, cap
        all_shifts_raw = base_shifts + ring_shifts + grid_shifts
        dedup = {}
        for dx, dz in all_shifts_raw:
            dedup[(round(dx, 6), round(dz, 6))] = (dx, dz)
        shifts = list(dedup.values())
        if len(shifts) > max_shifts:
            shifts = shifts[:max_shifts]

        # one cast -> boolean free mask
        def raycast_free_mask(origin_x: float, origin_z: float) -> np.ndarray:
            # world points along rays (shape: [r_max, N])
            Xw = origin_x + np.outer(ds, cos_a)
            Zw = origin_z + np.outer(ds, sin_a)

            # to grid
            cols = np.clip(((Xw - self.x_min) / self.resolution).astype(int), 0, self.nx - 1)
            rows = np.clip(((Zw - self.z_min) / self.resolution).astype(int), 0, self.nz - 1)

            # occupancy lookups along each ray
            occ_ray = self.occupancy[rows, cols]   # shape: [r_max, N]
            blocked = (occ_ray == 1)
            any_blocked = blocked.any(axis=0)
            first_obs = np.where(any_blocked, blocked.argmax(axis=0), r_max)

            # collect free indices up to first obstacle (exclusive)
            free_mask = np.zeros_like(self.occupancy, dtype=bool)
            for i in range(angles.shape[0]):
                end = int(first_obs[i])
                if end > 0:
                    rr = rows[:end, i]
                    cc = cols[:end, i]
                    free_mask[rr, cc] = True
            return free_mask

        # run multiple casts & intersect
        all_free_mask: Optional[np.ndarray] = None
        base_x, _, base_z = float(sensor_origin[0]), float(sensor_origin[1]), float(sensor_origin[2])

        valid_casts = 0
        for dx, dz in shifts:
            ox, oz = base_x + dx, base_z + dz

            # Apply origin policy: try nudge, else skip, else proceed
            if not is_origin_traversable(ox, oz):
                if nudge_blocked_origins:
                    ox, oz = nudge_to_nearest_free(ox, oz)
                # after possible nudge, check again (or honor skip policy)
                if not is_origin_traversable(ox, oz) and skip_blocked_origins:
                    continue  # skip this shift entirely

            fm = raycast_free_mask(ox, oz)
            valid_casts += 1
            if all_free_mask is None:
                all_free_mask = fm
            else:
                all_free_mask &= fm

        if all_free_mask is None or valid_casts == 0:
            # No valid casts; keep obstacles, leave others unknown
            self.occupancy = new_occ
            return

        # do not overwrite obstacles; mark as free only where all casts agree AND cell isn't an obstacle
        free_rows, free_cols = np.where(all_free_mask & (self.occupancy != 1))
        if free_rows.size:
            new_occ[free_rows, free_cols] = 0

        self.occupancy = new_occ

    def merge(self, other: "OccupancyMap", free_overrides_occupied: Optional[bool] = None) -> None:
        """
        In-place merge of another OccupancyMap into self.
        Requirements:
          - same resolution
          - both maps have height_map & occupancy
        Result:
          - self becomes the union of both maps over world coords
          - background fill uses this map's original floor, preserving its min
        """
        if not isinstance(other, OccupancyMap):
            raise TypeError("can only merge with another OccupancyMap")
        if self.resolution != other.resolution:
            raise ValueError("resolutions must match to merge")

        policy_free_wins = self.free_overrides_occupied if free_overrides_occupied is None else free_overrides_occupied

        res = self.resolution

        # compute world‐frame union bounds
        x0 = min(self.x_min, other.x_min)
        z0 = min(self.z_min, other.z_min)
        x1 = max(self.x_min + self.nx * res, other.x_min + other.nx * res)
        z1 = max(self.z_min + self.nz * res, other.z_min + other.nz * res)

        new_x_min, new_z_min = x0, z0
        new_nx = int(math.ceil((x1 - x0) / res))
        new_nz = int(math.ceil((z1 - z0) / res))

        # preserve this map's background floor
        y_floor = float(self.height_map.min())
        new_hmap = np.full((new_nz, new_nx), y_floor, dtype=self.height_map.dtype)

        new_counts = None
        if hasattr(self, "cell_counts"):
            new_counts = np.zeros((new_nz, new_nx), dtype=self.cell_counts.dtype)

        # -1 unknown, 0 free, 1 occupied
        new_occup = -np.ones((new_nz, new_nx), dtype=np.int8)

        def blit(src):
            # compute grid‐offset slices for this source map
            dr = int(round((src.z_min - new_z_min) / res))
            dc = int(round((src.x_min - new_x_min) / res))
            rsl = slice(dr, dr + src.nz)
            csl = slice(dc, dc + src.nx)

            np.maximum(
                new_hmap[rsl, csl],
                src.height_map,
                out=new_hmap[rsl, csl]
            )

            if new_counts is not None:
                new_counts[rsl, csl] += src.cell_counts

            old_occ = new_occup[rsl, csl]
            src_occ = src.occupancy.astype(np.int8)

            if policy_free_wins:
                merged = np.where(
                    (old_occ == 0) | (src_occ == 0), 0,
                    np.where(
                        (old_occ == 1) | (src_occ == 1), 1,
                        -1
                    )
                )
            else:
                merged = np.where(
                    (old_occ == 1) | (src_occ == 1), 1,
                    np.where(
                        (old_occ == 0) | (src_occ == 0), 0,
                        -1
                    )
                )
            new_occup[rsl, csl] = merged

        blit(self)
        blit(other)

        # write back
        self.x_min, self.z_min = new_x_min, new_z_min
        self.nx, self.nz = new_nx, new_nz
        self.height_map = new_hmap
        if new_counts is not None:
            self.cell_counts = new_counts
        self.occupancy = new_occup

    def merge_free_from(self, other: "OccupancyMap") -> None:
        """
        In-place import of FREE space (0) from another map into self.
        - Requires same resolution.
        - Only FREE (0) cells from `other` are copied.
        - Obstacles (1) or unknown (-1) from `other` are ignored.
        - Obstacles in `self` are NEVER overwritten.
        - If `other` extends outside `self` bounds, we expand to the union bounds.

        Height-map is combined via max() where both maps have data, preserving self's
        existing floor for background.
        """
        if not isinstance(other, OccupancyMap):
            raise TypeError("can only merge_free_from another OccupancyMap")
        if self.resolution != other.resolution:
            raise ValueError("resolutions must match for merge_free_from")

        res = self.resolution

        # Union bounds in world frame
        x0 = min(self.x_min, other.x_min)
        z0 = min(self.z_min, other.z_min)
        x1 = max(self.x_min + self.nx * res, other.x_min + other.nx * res)
        z1 = max(self.z_min + self.nz * res, other.z_min + other.nz * res)

        new_x_min, new_z_min = x0, z0
        new_nx = int(math.ceil((x1 - x0) / res))
        new_nz = int(math.ceil((z1 - z0) / res))

        # Prepare new arrays
        y_floor = float(self.height_map.min()) if self.height_map is not None else 0.0
        new_hmap = np.full((new_nz, new_nx), y_floor, dtype=np.float32)

        new_counts = None
        if hasattr(self, "cell_counts"):
            new_counts = np.zeros((new_nz, new_nx), dtype=self.cell_counts.dtype)

        # start unknown everywhere
        new_occup = -np.ones((new_nz, new_nx), dtype=np.int8)

        def blit_src_into(dest_hmap, dest_counts, dest_occ, src: "OccupancyMap"):
            dr = int(round((src.z_min - new_z_min) / res))
            dc = int(round((src.x_min - new_x_min) / res))
            rsl = slice(dr, dr + src.nz)
            csl = slice(dc, dc + src.nx)

            if src.height_map is not None:
                np.maximum(dest_hmap[rsl, csl], src.height_map, out=dest_hmap[rsl, csl])

            if (dest_counts is not None) and hasattr(src, "cell_counts"):
                dest_counts[rsl, csl] += src.cell_counts

            # copy src occupancy where dest is unknown; do not resolve conflicts here
            # we will do explicit free-only import from `other` below.
            src_occ = src.occupancy
            if src_occ is not None:
                # Where dest is unknown, set to src (so self region gets copied)
                mask = (dest_occ[rsl, csl] == -1)
                dest_occ[rsl, csl][mask] = src_occ[mask]

        # First, blit self fully
        blit_src_into(new_hmap, new_counts, new_occup, self)

        # Then, import FREE from `other` only (never obstacles/unknown)
        dr_o = int(round((other.z_min - new_z_min) / res))
        dc_o = int(round((other.x_min - new_x_min) / res))
        rsl_o = slice(dr_o, dr_o + other.nz)
        csl_o = slice(dc_o, dc_o + other.nx)

        if other.height_map is not None:
            np.maximum(new_hmap[rsl_o, csl_o], other.height_map, out=new_hmap[rsl_o, csl_o])
        if (new_counts is not None) and hasattr(other, "cell_counts"):
            new_counts[rsl_o, csl_o] += other.cell_counts

        # Only set FREE (0) from other where current is NOT an obstacle
        o_occ = other.occupancy.astype(np.int8)
        dest_view = new_occup[rsl_o, csl_o]
        free_from_other = (o_occ == 0)
        not_obstacle_here = (dest_view != 1)  # don't overwrite current obstacles
        write_mask = free_from_other & not_obstacle_here
        dest_view[write_mask] = 0  # import free cells

        # Write back
        self.x_min, self.z_min = new_x_min, new_z_min
        self.nx, self.nz = new_nx, new_nz
        self.height_map = new_hmap
        if new_counts is not None:
            self.cell_counts = new_counts
        self.occupancy = new_occup

    def compute_clearance_map(
        self,
        *,
        treat_unknown_as_obstacle: bool = True,
        obstacle_inflate_cells: int = 0,
    ) -> np.ndarray:
        """
        Returns a (nz, nx) float array of clearance in meters to the nearest obstacle.
        Args:
            treat_unknown_as_obstacle: if True, unknown (-1) are treated as blocking for clearance.
            obstacle_inflate_cells: optional integer pre-inflation (structuring element size in cells)
                                    before measuring clearance (useful for sensor/footprint padding).
        """
        assert self.occupancy is not None, "call set_point_cloud() first"

        occ = self.occupancy
        """
        Returns a (nz, nx) float array of clearance in meters to the nearest obstacle.
        Args:
            treat_unknown_as_obstacle: if True, unknown (-1) are treated as blocking for clearance.
            obstacle_inflate_cells: optional integer pre-inflation (structuring element size in cells)
                                    before measuring clearance (useful for sensor/footprint padding).
        """
        assert self.occupancy is not None, "call set_point_cloud() first"

        occ = self.occupancy
        # obstacles base mask
        obstacles = (occ == 1) | ((occ == -1) if treat_unknown_as_obstacle else False)

        if obstacle_inflate_cells > 0:
            se = np.ones((2 * obstacle_inflate_cells + 1, 2 * obstacle_inflate_cells + 1), dtype=np.uint8)
            obstacles = binary_dilation(obstacles, structure=se)

        # distance_transform_edt computes distance to the nearest zero; give it ~obstacles
        # so that distances are computed in the traversable region to the nearest obstacle cell.
        free_like = ~obstacles
        clearance_cells = distance_transform_edt(free_like.astype(np.uint8))
        clearance_m = clearance_cells * float(self.resolution)
        return clearance_m

    def apply_clearance(
        self,
        min_clearance_m: float,
        *,
        treat_unknown_as_obstacle: bool = True,
        overwrite_unknown: bool = False,
        obstacle_inflate_cells: int = 0,
    ) -> None:
        """
        In-place **inflation** of obstacles to enforce a minimum clearance around them.

        Behavior:
            - Computes clearance (meters) to nearest obstacle (unknown optionally treated as obstacles).
            - Any FREE cell (0) with clearance < min_clearance_m becomes OCCUPIED (1).
            - Unknown cells (-1) are left as-is unless `overwrite_unknown=True`, in which case unknowns
              that are within the clearance band are also marked OCCUPIED.

        Args:
            min_clearance_m: required clearance radius in meters (robot radius + margin).
            treat_unknown_as_obstacle: if True, unknowns repel (more conservative).
            overwrite_unknown: if True, unknown cells inside the inflated band are set to 1.
            obstacle_inflate_cells: optional pre-inflation (cells) before computing clearance.
        """
        assert self.occupancy is not None, "call set_point_cloud() first"
        occ = self.occupancy.copy()

        clearance_m = self.compute_clearance_map(
            treat_unknown_as_obstacle=treat_unknown_as_obstacle,
            obstacle_inflate_cells=obstacle_inflate_cells,
        )

        # Mask of cells too close to obstacles
        too_close = clearance_m < float(min_clearance_m)

        # Inflate: free cells within band become occupied
        free_mask = (occ == 0)
        occ[np.where(too_close & free_mask)] = 1

        if overwrite_unknown:
            unk_mask = (occ == -1)
            occ[np.where(too_close & unk_mask)] = 1

        self.occupancy = occ

    def save_height_map(
        self,
        file_path: Union[str, Path],
        cmap: str = "viridis",
        dpi: int = 150,
        units: str = "meters",
        path: Optional[List[np.ndarray]] = None,
        min_size: Tuple[int,int] = (512, 512)
    ):
        """
        Save the current height_map as a PNG image, optionally overlaying a path.
        Ensures the saved image is at least min_size in pixels, scaling up
        if necessary without changing aspect ratio.

        :param file_path: where to write the .png
        :param cmap: any matplotlib colormap name
        :param dpi: resolution of the saved figure
        :param units: "grid" for cell-indices, "meters" for world coords
        :param path: list of [x,y,z] waypoints from plan(); plotted in red
        :param min_size: (min_width_px, min_height_px), defaults to (512,512)
        """
        assert self.height_map is not None, "no height map - call set_point_cloud() first"

        # ensure output folder exists
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # base figure size in inches
        base_w_in = self.nx / 20
        base_h_in = self.nz / 20

        # what the pixel dims would be
        pix_w = base_w_in * dpi
        pix_h = base_h_in * dpi

        # compute scale factor ≥1 so both dims ≥ min_size
        min_w_px, min_h_px = min_size
        scale = max(
            1.0,
            (min_w_px / pix_w) if min_w_px > 0 else 1.0,
            (min_h_px / pix_h) if min_h_px > 0 else 1.0,
        )

        fig_w = base_w_in * scale
        fig_h = base_h_in * scale

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        # display height map
        if units == "meters":
            xmin = self.x_min
            xmax = self.x_min + self.nx * self.resolution
            zmin = self.z_min
            zmax = self.z_min + self.nz * self.resolution
            im = ax.imshow(
                self.height_map,
                origin="lower",
                cmap=cmap,
                extent=[xmin, xmax, zmin, zmax],
                aspect='equal'
            )
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
        else:
            im = ax.imshow(
                self.height_map,
                origin="lower",
                cmap=cmap
            )
            ax.set_xlabel("Grid X (cols)")
            ax.set_ylabel("Grid Z (rows)")

        # overlay path if provided
        if path:
            xs = [wp[0] for wp in path]
            zs = [wp[2] for wp in path]

            if units == "grid":
                grid = [self._world_to_grid((x, z)) for x, z in zip(xs, zs)]
                rows, cols = zip(*grid)
                ax.plot(cols, rows, color='red', linewidth=2)
                ax.plot(cols[0], rows[0], marker='o', color='blue', markersize=8, label='start')
                ax.plot(cols[-1], rows[-1], marker='*', color='red', markersize=12, label='goal')
            else:
                ax.plot(xs, zs, color='red', linewidth=2)
                ax.plot(xs[0], zs[0], marker='o', color='blue', markersize=8, label='start')
                ax.plot(xs[-1], zs[-1], marker='*', color='red', markersize=12, label='goal')

            ax.legend(
                loc='upper right',
                fontsize='small',
                markerscale=0.7,
                frameon=True,
                facecolor='white',
                edgecolor='black',
                framealpha=1.0
            )

        cbar = fig.colorbar(im, ax=ax, label="Height Y (m)")
        ax.set_title("Height Map")
        fig.tight_layout()
        fig.savefig(str(file_path))
        plt.close(fig)

    def save_occupancy_map(
        self,
        file_path: Union[str, Path],
        dpi: int = 150,
        units: str = "meters",
        goals: Optional[List[Union[np.ndarray, Tuple[float, float, float]]]] = None,
        path: Optional[List[Union[np.ndarray, Tuple[float, float, float]]]] = None,
        min_size: Tuple[int,int] = (512, 512),
        save: bool = True,               # write to disk?
        return_image: bool = True,       # return rendered RGBA image (H,W,4), uint8
        include_axes: bool = True,       # set False to get a clean image (no ticks/labels)
    ) -> Optional[np.ndarray]:
        """
        Render the occupancy grid, optionally overlaying goals/path.
        Returns an RGBA image array if return_image=True; never returns a Figure.
        """
        assert self.occupancy is not None, "Occupancy grid is not initialized"

        # discrete colormap for {-1: unknown, 0: free, 1: occupied}
        cmap = ListedColormap(['gray', 'white', 'black'])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

        file_path = Path(file_path)
        if save:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Base figure size scaled to ensure min pixel dims
        base_w_in = self.nx / 20
        base_h_in = self.nz / 20
        pix_w = base_w_in * dpi
        pix_h = base_h_in * dpi
        min_w_px, min_h_px = min_size
        scale = max(
            1.0,
            (min_w_px / pix_w) if min_w_px > 0 else 1.0,
            (min_h_px / pix_h) if min_h_px > 0 else 1.0,
        )
        fig_w = base_w_in * scale
        fig_h = base_h_in * scale

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        # display occupancy
        if units == "meters":
            xmin = self.x_min
            xmax = self.x_min + self.nx * self.resolution
            zmin = self.z_min
            zmax = self.z_min + self.nz * self.resolution
            ax.imshow(
                self.occupancy,
                origin="lower",
                cmap=cmap,
                norm=norm,
                extent=[xmin, xmax, zmin, zmax],
                aspect="equal"
            )
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
        else:
            ax.imshow(self.occupancy, origin="lower", cmap=cmap, norm=norm)
            ax.set_xlabel("Grid X (cols)")
            ax.set_ylabel("Grid Z (rows)")

        # overlay goals if provided
        if goals is not None:
            if units == "meters":
                xs = [g[0] for g in goals]
                zs = [g[2] for g in goals]
                ax.scatter(xs, zs, marker='x', s=50, linewidths=2, edgecolors='red', label='Goals')
            else:
                pts = [self._world_to_grid((g[0], g[2])) for g in goals]
                rows, cols = zip(*pts)
                ax.scatter(cols, rows, marker='x', s=50, linewidths=2, edgecolors='red', label='Goals')

        # overlay path if provided
        if path is not None:
            xs = [wp[0] for wp in path]
            zs = [wp[2] for wp in path]
            if units == "grid":
                grid_pts = [self._world_to_grid((x, z)) for x, z in zip(xs, zs)]
                rows, cols = zip(*grid_pts)
                ax.plot(cols, rows, color='blue', linewidth=2, label='Path')
                ax.plot(cols[0], rows[0], marker='o', color='green', markersize=8, label='Start')
                ax.plot(cols[-1], rows[-1], marker='*', color='orange', markersize=12, label='Goal')
            else:
                ax.plot(xs, zs, color='blue', linewidth=2, label='Path')
                ax.plot(xs[0], zs[0], marker='o', color='green', markersize=8, label='Start')
                ax.plot(xs[-1], zs[-1], marker='*', color='orange', markersize=12, label='Goal')

        # legend if overlays exist
        if goals is not None or path is not None:
            ax.legend(
                loc='upper right',
                fontsize='small',
                markerscale=0.7,
                frameon=True,
                facecolor='white',
                edgecolor='black',
                framealpha=1.0
            )

        # optionally hide axes to get a clean image
        if not include_axes:
            ax.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        fig.tight_layout()
        fig.canvas.draw()

        # save to disk if requested
        if save:
            # when axes hidden, also remove margins in saved file
            save_kwargs = {}
            if not include_axes:
                save_kwargs.update(dict(bbox_inches='tight', pad_inches=0))
            fig.savefig(str(file_path), **save_kwargs)

        img = None
        if return_image:
            img = np.asarray(fig.canvas.buffer_rgba()).copy()  # (H, W, 4), uint8

        plt.close(fig)
        return img

    def _world_to_grid(self, xz: Tuple[float,float]) -> Tuple[int,int]:
        x, z = xz
        col = int((x - self.x_min) // self.resolution)
        row = int((z - self.z_min) // self.resolution)
        col = np.clip(col, 0, self.nx - 1)
        row = np.clip(row, 0, self.nz - 1)
        return (row, col)

    def _grid_to_world(self, rc: Tuple[int,int]) -> Tuple[float,float]:
        row, col = rc
        x = self.x_min + col * self.resolution + self.resolution/2
        z = self.z_min + row * self.resolution + self.resolution/2
        return (x, z)

    def _in_bounds(self, rc: Tuple[int,int]) -> bool:
        r, c = rc
        return 0 <= r < self.nz and 0 <= c < self.nx