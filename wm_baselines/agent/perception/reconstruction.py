import numpy as np
import open3d as o3d
from wm_baselines.utils.data_classes import Frame
from wm_baselines.utils.common_utils import with_timing


class TSDFFusion:
    """
    TSDF fusion using legacy integration (CPU) + Tensor raycasting (CPU/GPU).

    - Integration: open3d.pipelines.integration.ScalableTSDFVolume (CPU)
    - Rendering : open3d.t.geometry.RaycastingScene (Tensor; CPU or CUDA)

    Args:
        K: 3x3 intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]] in pixels
        width,height: image size in pixels
        voxel_length: TSDF voxel size (m)
        sdf_trunc: truncation distance (m)
        depth_scale: raw_depth / depth_scale -> meters (1000 for mm, 1.0 for meters)
        depth_trunc: max valid depth (meters)
        integrate_color: fuse RGB into legacy volume (raycast returns depth/normals; no color)
        device: "CPU:0" or "CUDA:0" — used for raycasting (not for integration)
    """
    def __init__(
        self,
        K,
        width, height,
        voxel_length=0.01,
        sdf_trunc=0.04,
        depth_scale=1.0,
        depth_trunc=10.0,
        integrate_color=True,
        device="CPU:0",
    ):
        self.width = int(width)
        self.height = int(height)
        self.depth_scale = float(depth_scale)
        self.depth_max = float(depth_trunc)
        self.integrate_color = bool(integrate_color)
        self._mesh_legacy_cache = None

        K = np.asarray(K, dtype=np.float32)
        assert K.shape == (3, 3), "K must be 3x3"
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.K = K.copy()

        self._vol_cfg = dict(
            voxel_length=float(voxel_length),
            sdf_trunc=float(sdf_trunc),
            color_type=(
                o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                if integrate_color else
                o3d.pipelines.integration.TSDFVolumeColorType.NoneColor
            ),
        )

        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(**self._vol_cfg)

        self._intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height,
            fx=float(self.fx), fy=float(self.fy),
            cx=float(self.cx), cy=float(self.cy),
        )

        # Tensor device for raycasting (CPU or CUDA)
        self._device = o3d.core.Device(device)
        self._scene = None                # o3d.t.geometry.RaycastingScene
        self._mesh_t = None               # o3d.t.geometry.TriangleMesh
        self._dirty_scene = True          # mark scene invalid after integration

        # Precompute unit ray directions in camera frame (H, W, 3) → reuse across renders
        us, vs = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32),
        )
        xs = (us - self.cx) / self.fx
        ys = (vs - self.cy) / self.fy
        dirs = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)  # unit rays
        self._unit_dirs_cam = dirs  # (H, W, 3)

    def reset(self, *, integrate_color: bool = None) -> None:
        """
        Clear the TSDF and raycasting scene.

        Args:
            integrate_color: optionally change whether color is fused on reset.
                             If None, keep previous behavior.
        """
        # update color setting if requested
        if integrate_color is not None:
            self.integrate_color = bool(integrate_color)
            self._vol_cfg["color_type"] = (
                o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                if self.integrate_color else
                o3d.pipelines.integration.TSDFVolumeColorType.NoneColor
            )

        # rebuild empty volume
        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(**self._vol_cfg)

        # drop scene + caches so next render rebuilds from scratch
        self._scene = None
        self._mesh_t = None
        self._mesh_legacy_cache = None
        self._dirty_scene = True

    def reset_scene_only(self) -> None:
        """
        Throw away only the raycasting scene/mesh caches.
        Useful if changed device or want to release GPU memory,
        without wiping the fused TSDF volume.
        """
        self._scene = None
        self._mesh_t = None
        self._mesh_legacy_cache = None
        self._dirty_scene = True

    # -------------------------- Integration --------------------------
    @with_timing
    def integrate_frame(self, depth, color=None, T_cam2world=None):
        """
        Integrate one RGB-D (or depth-only) frame.

        depth: (H,W) uint16 (e.g., mm) or float32 (m). Use depth_scale accordingly.
        color: (H,W,3) uint8 RGB or None
        T_cam2world: (4x4) camera pose in world frame
        """
        if T_cam2world is None:
            T_cam2world = np.eye(4, dtype=np.float64)

        assert depth.shape[:2] == (self.height, self.width), "depth size mismatch"
        if self.integrate_color:
            if color is None:
                color = np.zeros((self.height, self.width, 3), np.uint8)
            else:
                assert color.shape[:2] == (self.height, self.width), "color size mismatch"

        # --- Make buffers C-contiguous with the correct dtype ---
        if depth.dtype == np.uint16:
            depth_np = np.ascontiguousarray(depth)  # already uint16
        else:
            depth_np = np.ascontiguousarray(depth.astype(np.float32, copy=False))

        if self.integrate_color:
            color_np = np.ascontiguousarray(color.astype(np.uint8, copy=False))
        else:
            color_np = np.zeros((self.height, self.width, 3), np.uint8)  # already contiguous

        # --- Wrap as Open3D Images ---
        depth_img = o3d.geometry.Image(depth_np)
        color_img = o3d.geometry.Image(color_np)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img,
            depth_img,
            depth_scale=self.depth_scale,   # meters = raw / depth_scale
            depth_trunc=self.depth_max,
            convert_rgb_to_intensity=False,
        )

        # Legacy integrate() expects world->camera extrinsic
        T = np.asarray(T_cam2world, dtype=np.float64)
        extrinsic_w2c = np.linalg.inv(T)

        self._volume.integrate(rgbd, self._intrinsic, extrinsic_w2c)
        self._dirty_scene = True
    
    # -------------------------- Rendering ---------------------------

    def _ensure_scene(self):
        """(Re)build RaycastingScene from the current TSDF mesh when dirty."""
        if not self._dirty_scene and self._scene is not None:
            return

        mesh_legacy = self._volume.extract_triangle_mesh()
        if len(mesh_legacy.vertices) == 0:
            self._scene = None
            self._mesh_t = None
            self._mesh_legacy_cache = None   # <--- add cache
            self._dirty_scene = False
            return

        mesh_legacy.compute_triangle_normals()
        self._mesh_legacy_cache = mesh_legacy          # <--- add cache
        self._mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy).to(self._device)

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(self._mesh_t)
        self._scene = scene
        self._dirty_scene = False
    
    def _get_cached_mesh_legacy(self):
        """Return cached mesh if scene is up-to-date; else extract once."""
        if not self._dirty_scene and self._mesh_legacy_cache is not None:
            return self._mesh_legacy_cache
        mesh = self._volume.extract_triangle_mesh()
        if len(mesh.vertices):
            mesh.compute_triangle_normals()
        return mesh

    def render_view(
        self,
        T_cam2world,
        width=None, height=None,
        min_depth=0.1, max_depth=10.0,
        depth_scale_out=1.0,
        return_normals=True,
    ):
        """
        Raycast a novel view (depth + normals + rgb if mesh has vertex colors).
        Returns Frame(rgb, depth, normals) where rgb is (H,W,3) float32 in [0,1] or None.
        """

        # --- coalesce inputs ---
        md = float(min_depth if min_depth is not None else getattr(self, "min_depth", 0.1))
        MX = float(max_depth if max_depth is not None else getattr(self, "max_depth", 10.0))
        ds = float(depth_scale_out if depth_scale_out is not None else 1.0)
        w = int(width if width is not None else self.width)
        h = int(height if height is not None else self.height)

        self._ensure_scene()
        if self._scene is None:
            depth = np.zeros((h, w), np.float32)
            return Frame(rgb=None, depth=depth, normals=None)

        # --- ray directions in camera frame ---
        if (h, w) != (self.height, self.width) or getattr(self, "_unit_dirs_cam", None) is None:
            us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            xs = (us - float(self.cx)) / float(self.fx)
            ys = (vs - float(self.cy)) / float(self.fy)
            dirs_cam = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
            dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True) + 1e-8
            if (h, w) == (self.height, self.width):
                self._unit_dirs_cam = dirs_cam
        else:
            dirs_cam = self._unit_dirs_cam

        T = np.asarray(T_cam2world, dtype=np.float32)
        R = T[:3, :3]
        t = T[:3, 3]

        rays_o = np.broadcast_to(t, dirs_cam.shape)      # (H,W,3)
        rays_d = dirs_cam @ R.T                          # (H,W,3)
        rays = np.concatenate([rays_o, rays_d], axis=-1).reshape(-1, 6).astype(np.float32)
        rays_t = o3d.core.Tensor(rays, device=self._device)

        ans = self._scene.cast_rays(rays_t)

        # --- depth ---
        t_hit = ans["t_hit"].cpu().numpy().reshape(h, w)  # NaN where no hit
        depth = np.where(np.isfinite(t_hit), t_hit, 0.0).astype(np.float32)
        depth = np.where((depth >= md) & (depth <= MX), depth, 0.0)
        depth = depth * ds

        # --- normals (triangle normals via primitive_ids) ---
        normals = None
        if return_normals and "primitive_ids" in ans:
            prim_ids = ans["primitive_ids"].cpu().numpy().astype(np.int64).reshape(h, w)
            mesh_legacy = getattr(self, "_mesh_legacy_cache", None)
            if mesh_legacy is not None and len(mesh_legacy.triangle_normals):
                tri_normals = np.asarray(mesh_legacy.triangle_normals)
                valid = (prim_ids >= 0) & np.isfinite(t_hit)
                normals = np.zeros((h, w, 3), dtype=np.float32)
                normals[valid] = tri_normals[prim_ids[valid]]
                n = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
                normals = normals / n

        # --- RGB from vertex colors via barycentric interpolation ---
        rgb = None
        if "primitive_ids" in ans:
            prim_ids = ans["primitive_ids"].cpu().numpy().astype(np.int64).reshape(h, w)
            # Some Open3D builds provide "primitive_uvs" (barycentric u,v on the hit triangle)
            has_uv = "primitive_uvs" in ans
            mesh_legacy = getattr(self, "_mesh_legacy_cache", None)

            if mesh_legacy is not None and mesh_legacy.has_vertex_colors() and len(mesh_legacy.triangles) > 0:
                triangles = np.asarray(mesh_legacy.triangles, dtype=np.int64)      # (F,3)
                vcolors  = np.asarray(mesh_legacy.vertex_colors, dtype=np.float32) # (V,3) in [0,1] typically

                rgb = np.zeros((h, w, 3), dtype=np.float32)
                valid = (prim_ids >= 0) & np.isfinite(t_hit)

                if has_uv:
                    # barycentric weights w0=1-u-v, w1=u, w2=v
                    uvs = ans["primitive_uvs"].cpu().numpy().reshape(h, w, 2).astype(np.float32)
                    u = uvs[..., 0]
                    v = uvs[..., 1]
                    w0 = 1.0 - u - v
                    w1 = u
                    w2 = v

                    # Gather vertex indices per pixel
                    tri_idx = prim_ids[valid]
                    i0 = triangles[tri_idx, 0]
                    i1 = triangles[tri_idx, 1]
                    i2 = triangles[tri_idx, 2]

                    c0 = vcolors[i0]   # (N,3)
                    c1 = vcolors[i1]
                    c2 = vcolors[i2]

                    rgb_valid = (w0[valid, None] * c0 +
                                w1[valid, None] * c1 +
                                w2[valid, None] * c2)
                    rgb[valid] = np.clip(rgb_valid, 0.0, 1.0)
                else:
                    # Fallback: flat color per face (average of vertex colors)
                    tri_mean = (vcolors[triangles[:, 0]] +
                                vcolors[triangles[:, 1]] +
                                vcolors[triangles[:, 2]]) / 3.0
                    rgb[valid] = tri_mean[prim_ids[valid]]

                # Optional: zero out background explicitly
                rgb[~valid] = 0.0

        return Frame(rgb=rgb, depth=depth, normals=normals)

    # ----------------------- Utilities / Extraction ------------------

    def reset(self):
        """Clear the TSDF volume and invalidate the raycasting scene."""
        # Recreate volume to reset efficiently
        params = self._volume  # grab params before replacement
        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=params.voxel_length,
            sdf_trunc=params.sdf_trunc,
            color_type=params.color_type,
        )
        self._scene = None
        self._mesh_t = None
        self._dirty_scene = True
        self._mesh_legacy_cache = None

    def extract_mesh(self, to_legacy=True):
        mesh = self._volume.extract_triangle_mesh()
        return mesh if to_legacy else o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    def extract_point_cloud(
        self,
        to_legacy: bool = True,
        approx_points: int = 300_000,
        voxel_size: float = None,
        mode: str = "auto",
    ):
        """
        Fast point cloud extraction.

        mode:
        - "auto": try direct TSDF PCD -> voxel surface PCD -> cached mesh uniform
        - "direct": force ScalableTSDFVolume.extract_point_cloud()
        - "voxel":  force ScalableTSDFVolume.extract_voxel_point_cloud()
        - "uniform": force cached mesh uniform sampling
        """
        # DIRECT TSDF -> PCD (fastest, includes colors if integrated)
        if mode in ("auto", "direct") and hasattr(self._volume, "extract_point_cloud"):
            pcd = self._volume.extract_point_cloud()
            if voxel_size:
                pcd = pcd.voxel_down_sample(voxel_size)
            return pcd if to_legacy else o3d.t.geometry.PointCloud.from_legacy(pcd)

        # VOXEL SURFACE PCD (very fast, coarser sampling of surface)
        if mode in ("auto", "voxel") and hasattr(self._volume, "extract_voxel_point_cloud"):
            pcd = self._volume.extract_voxel_point_cloud()
            if voxel_size:
                pcd = pcd.voxel_down_sample(voxel_size)
            return pcd if to_legacy else o3d.t.geometry.PointCloud.from_legacy(pcd)

        # CACHED MESH -> UNIFORM SAMPLING (much faster than Poisson disk)
        mesh = self._get_cached_mesh_legacy()
        if len(mesh.vertices) == 0:
            return o3d.geometry.PointCloud() if to_legacy else o3d.t.geometry.PointCloud()

        approx_points = int(max(10_000, approx_points))
        pcd = mesh.sample_points_uniformly(number_of_points=approx_points)

        if voxel_size:
            pcd = pcd.voxel_down_sample(voxel_size)

        return pcd if to_legacy else o3d.t.geometry.PointCloud.from_legacy(pcd)