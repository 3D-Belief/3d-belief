from typing import Any, Dict, Union, List, Tuple, Optional
from collections import deque
import math
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from omegaconf import DictConfig
from copy import deepcopy
from wm_baselines.utils.vision_utils import visualize_semantic_query_intensity_map
from wm_baselines.agent.perception.occupancy import OccupancyMap
from wm_baselines.planner.base_planner import BasePlanner
from wm_baselines.utils.planning_utils import rotation_angle, goals_and_forwards_to_poses
from wm_baselines.utils.common_utils import with_timing
from wm_baselines.world_model.base_world_model import BaseWorldModel
from wm_baselines.planner.planning.path_planning import PATH_PLANNING_REGISTRY, path_to_trajectory
from wm_baselines.planner.planning.goal_sampling import GOAL_SAMPLING_REGISTRY
from wm_baselines.planner.exploration_planner import ExplorationPlanner

class Belief3DModelGoalSelector(ExplorationPlanner):
    """A planner that uses 3d belief model built-in semantics for goal selection."""

    def __init__(
        self,
        goal_selection_strategy: str = "max_camera_score",
        anti_goal_radius: float = 1.5,
        no_progress_window: int = 30,
        no_progress_min_displacement_m: float = 0.5,
        frontier_fallback_after_stuck: int = 3,
        frontier_fallback_strategy: str = "random_free",
        semantic_tie_margin: float = 0.003,
        hotspot_tie_radius_m: float = 0.75,
        forward_tiebreak_weight: float = 0.35,
        goal_stick_radius_m: float = 0.75,
        goal_stick_bonus: float = 0.35,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.world_model: BaseWorldModel = None  # to be set externally
        assert goal_selection_strategy in (
            "max_camera_score", "max_pixel_3d", "closest_camera_to_max_pixel"
        ), f"Unknown goal_selection_strategy: {goal_selection_strategy}"
        self.goal_selection_strategy = goal_selection_strategy

        # Recently failed goals to avoid.
        self.anti_goal_radius: float = float(anti_goal_radius)
        self.anti_goals: List[Tuple[float, float, int]] = []

        # Stall detection.
        self.no_progress_window: int = int(no_progress_window)
        self.no_progress_min_displacement_m: float = float(no_progress_min_displacement_m)
        self._pose_window: deque = deque(maxlen=self.no_progress_window)

        # Long-range fallback after repeated stalls.
        self.frontier_fallback_after_stuck: int = int(frontier_fallback_after_stuck)
        self.frontier_fallback_strategy: str = str(frontier_fallback_strategy)
        assert self.frontier_fallback_strategy in GOAL_SAMPLING_REGISTRY, \
            f"Unknown frontier_fallback_strategy: {self.frontier_fallback_strategy}"
        self._stuck_commit_count: int = 0
        self._last_progress_xz: Optional[Tuple[float, float]] = None

        # Tie-break near-equal semantic scores.
        self.semantic_tie_margin: float = float(semantic_tie_margin)
        self.hotspot_tie_radius_m: float = float(hotspot_tie_radius_m)
        self.forward_tiebreak_weight: float = float(forward_tiebreak_weight)
        self.goal_stick_radius_m: float = float(goal_stick_radius_m)
        self.goal_stick_bonus: float = float(goal_stick_bonus)
        self._last_goal_ranking: List[int] = []

    def reset(self):
        super().reset()
        self.anti_goals = []
        self._pose_window.clear()
        self._stuck_commit_count = 0
        self._last_progress_xz = None
        self._last_goal_ranking = []

    def get_next_action(self, current_asset: Dict[str, Any], current_step: int, force_update: bool = False) -> Dict[str, Any]:
        """Get the next action."""
        if current_asset is not None and current_step > self.step:
            self.set_state(current_asset)
            self.step += 1
            assert self.step == current_step, f"step mismatch: {self.step} vs {current_step}"
        self.current_goal_steps += 1
        self._update_replan(force=force_update)
        self._detect_agent_stuck()
        self._detect_no_progress_and_flag()
        self._check_progress_and_reset_counter()
        if 'pose' in current_asset:
            self.current_asset["pose"] = current_asset["pose"]

        if self.random_free:
            self.current_trajectory = []
            self.path_keypoints = []
            trace_asset = {
                # "pcd": current_asset["pcd"],
                "rgb": current_asset["rgb"],
                "occupancy": deepcopy(current_asset["occupancy"]),
                "path": None,
                "goal": self.current_goal if self.current_goal is not None else None,
                "all_goals": self.all_goals if len(self.all_goals) > 0 else None,
                "goal_images": self.current_asset.get("goal_images", None),
                "goal_semantics": self.current_asset.get("goal_semantics", None),
                "goal_scores": self.current_asset.get("goal_scores", None),
                "goal_poses": self.current_asset.get("goal_poses", None),
            }

            # Try primitive actions until one produces motion.
            if not hasattr(self, "_unstick_sequence") or self._unstick_sequence is None:
                self._unstick_sequence = [
                    "move_back",
                    "turn_left",
                    "move_forward",
                    "turn_right", "turn_right",   # 180 turn
                    "move_forward",
                    "move_back",
                    "turn_left", "turn_left",     # original heading
                    "move_forward",
                    "turn_right",
                ]
            if self.rotate_count == 0:
                # New stuck episode.
                self._unstick_idx = 0
                self._unstick_last_pose = None
                self._unstick_last_action = None

            cur_pose = self.current_asset["pose"]
            cur_pos = cur_pose[:3, 3]
            cur_pos = cur_pos.cpu().numpy() if isinstance(cur_pos, Tensor) else np.asarray(cur_pos)
            last_pose = getattr(self, "_unstick_last_pose", None)
            moved = False
            if last_pose is not None:
                last_pos = last_pose[:3, 3]
                last_pos = last_pos.cpu().numpy() if isinstance(last_pos, Tensor) else np.asarray(last_pos)
                dpos = float(np.linalg.norm(cur_pos - last_pos))
                dyaw = float(rotation_angle(cur_pose[:3, 2], last_pose[:3, 2]))
                moved = (dpos > 0.05) or (dyaw > 0.05)
                if moved:
                    print(
                        f"[unstick] action '{self._unstick_last_action}' produced motion "
                        f"(Δpos={dpos:.3f}m, Δyaw={dyaw:.3f}rad); exiting stuck mode and replanning"
                    )

            if moved:
                self.random_free = False
                self.rotate_count = 0
                self._unstick_idx = 0
                self._unstick_last_pose = None
                self._unstick_last_action = None
                self.replan = True
                # Advance one step before replanning.
                return ({"action_name": "turn_right", "args": {}}, trace_asset)

            if self._unstick_idx >= len(self._unstick_sequence) or \
                    self.rotate_count >= self.stuck_rotate_times:
                # Replan after unstick fails.
                print(
                    f"[unstick] sequence exhausted "
                    f"(idx={self._unstick_idx}, rotate_count={self.rotate_count}); "
                    f"resetting and forcing replan"
                )
                self.random_free = False
                self.rotate_count = 0
                self._unstick_idx = 0
                self._unstick_last_pose = None
                self._unstick_last_action = None
                self.replan = True
                return ({"action_name": "turn_right", "args": {}}, trace_asset)

            action_name = self._unstick_sequence[self._unstick_idx]
            print(
                f"[unstick] step {self.rotate_count}: trying '{action_name}' "
                f"(idx={self._unstick_idx}/{len(self._unstick_sequence)})"
            )
            self._unstick_idx += 1
            self.rotate_count += 1
            self._unstick_last_pose = cur_pose
            self._unstick_last_action = action_name
            return ({"action_name": action_name, "args": {}}, trace_asset)

        if self.replan or self.random_free or len(self.current_trajectory)==0:
            print("Replanning...")
            self.current_path, exe_time = self.plan()
            self._metrics["planning_time"] += exe_time
            if self.current_path is None or len(self.current_path) == 0:
                self.current_trajectory = []
                self.path_keypoints = []
                print("No more valid path found")
                trace_asset = {
                    "rgb": current_asset["rgb"],
                    "occupancy": deepcopy(current_asset["occupancy"]),
                    "path": None,
                    "goal": self.current_goal if self.current_goal is not None else None,
                    "all_goals": self.all_goals if len(self.all_goals) > 0 else None,
                    "goal_images": self.current_asset.get("goal_images", None),
                    "goal_semantics": self.current_asset.get("goal_semantics", None),
                    "goal_scores": self.current_asset.get("goal_scores", None),
                    "goal_poses": self.current_asset.get("goal_poses", None),
                }
                return {"action_name": "turn_right", "args": {}}, trace_asset
            self.current_trajectory, self.path_keypoints = path_to_trajectory(self.current_path, occ=self.current_asset["occupancy"], ref_pose=self.current_asset["initial_pose"])
            self.replan = False
            self.current_goal_steps = 0
            self.current_path_len = len(self.current_path)

        next_pose = self.current_trajectory.pop(0)  # (x, y, z)
        path_keypoints = self.path_keypoints.copy()      # or lst[:] for a shallow copy
        self.path_keypoints.pop(0)
        action = {"action_name": self.action_name, "args": {"target_pose": next_pose}}

        trace_asset = {
            "rgb": current_asset["rgb"],
            "occupancy": deepcopy(current_asset["occupancy"]),
            "path": path_keypoints if len(path_keypoints)>0 else None,
            "goal": self.current_goal if self.current_goal is not None else None,
            "all_goals": self.all_goals if len(self.all_goals) > 0 else None,
            "goal_images": self.current_asset.get("goal_images", None),
            "goal_semantics": self.current_asset.get("goal_semantics", None),
            "goal_scores": self.current_asset.get("goal_scores", None),
            "goal_poses": self.current_asset.get("goal_poses", None),
        }

        return action, trace_asset

    @with_timing
    def plan(self) -> Dict[str, Any]:
        """Given the current observation and assets, return the next action dict."""
        occ: OccupancyMap = self.current_asset["occupancy"]
        current_pose: Union[Tensor, np.ndarray] = self.current_asset["pose"]  # (4, 4)
        initial_pose: Union[Tensor, np.ndarray] = self.current_asset["initial_pose"]  # (4, 4)
        object_name: str = self.current_asset["object_name"]
        start = current_pose[:3, 3].cpu().numpy() if isinstance(current_pose, Tensor) else current_pose[:3, 3]  # (3,)
        # Use long-range fallback after repeated stalls.
        use_fallback = self._stuck_commit_count >= self.frontier_fallback_after_stuck
        if use_fallback:
            print(
                f"[fallback_frontier_stuck] stuck_count={self._stuck_commit_count} "
                f">= {self.frontier_fallback_after_stuck}; switching sampler to "
                f"'{self.frontier_fallback_strategy}' for this plan"
            )
            goals, forwards = GOAL_SAMPLING_REGISTRY[self.frontier_fallback_strategy](occ, current_pose)
            self._stuck_commit_count = 0
        else:
            # sample goal
            goals, forwards = GOAL_SAMPLING_REGISTRY[self.goal_sampling_strategy](occ, current_pose, **self.goal_sampling_kwargs)
        # Drop candidates near recent failed goals.
        if goals and len(goals) > 0 and len(self.anti_goals) > 0:
            kept = [(g, f) for g, f in zip(goals, forwards)
                    if not self._is_anti_goal(float(g[0]), float(g[2]))]
            if len(kept) > 0:
                goals = [g for g, _ in kept]
                forwards = [f for _, f in kept]
            else:
                print(
                    f"[anti_goal] all {len(self.anti_goals)} anti-goals filter out "
                    f"every candidate; using unfiltered set this step"
                )
        if goals is None or len(goals) == 0:
            self.all_goals = []
            self.current_goal = None
            return []  # no valid goal found

        self.all_goals = goals
        if use_fallback:
            self._clear_goal_render_assets()
            self._last_goal_ranking = list(range(len(goals)))
        else:
            self.select_goal_semantic(goals, forwards, initial_pose, object_name)

        ranked_indices = self._last_goal_ranking or list(range(len(goals)))
        best_idx, current_path = self._plan_first_reachable(
            occ=occ,
            start=start,
            goals=goals,
            forwards=forwards,
            ranked_indices=ranked_indices,
            source="semantic",
        )

        # Try long-range fallback before random walk.
        if current_path is None and not use_fallback:
            fallback_goals, fallback_forwards = GOAL_SAMPLING_REGISTRY[self.frontier_fallback_strategy](occ, current_pose)
            if fallback_goals is not None and len(fallback_goals) > 0:
                if len(self.anti_goals) > 0:
                    kept = [(g, f) for g, f in zip(fallback_goals, fallback_forwards)
                            if not self._is_anti_goal(float(g[0]), float(g[2]))]
                    if len(kept) > 0:
                        fallback_goals = [g for g, _ in kept]
                        fallback_forwards = [f for _, f in kept]
                f_idx, f_path = self._plan_first_reachable(
                    occ=occ,
                    start=start,
                    goals=fallback_goals,
                    forwards=fallback_forwards,
                    ranked_indices=list(range(len(fallback_goals))),
                    source=self.frontier_fallback_strategy,
                )
                if f_path is not None:
                    goals, forwards = fallback_goals, fallback_forwards
                    best_idx, current_path = f_idx, f_path
                    self.all_goals = goals
                    self._clear_goal_render_assets()

        if current_path is None:
            fallback_idx = ranked_indices[0]
            self.current_goal = goals[fallback_idx]  # (x, y, z)
            self.current_forward = forwards[fallback_idx]  # (3,)
            print("All candidate paths failed, using random walk fallback")
            current_path = PATH_PLANNING_REGISTRY["random_walk"](occ, start, self.current_goal)
            self.replan = True  # force replan next time
        else:
            self.current_goal = goals[best_idx]  # (x, y, z)
            self.current_forward = forwards[best_idx]  # (3,)

        self.current_path = current_path
        return self.current_path

    def _clear_goal_render_assets(self) -> None:
        """Clear rendered goal assets when a non-rendered fallback sampler wins."""
        if self.current_asset is None:
            return
        for key in ("goal_images", "goal_semantics", "goal_scores", "goal_poses"):
            self.current_asset.pop(key, None)

    def _plan_first_reachable(
        self,
        occ: OccupancyMap,
        start: np.ndarray,
        goals: List[np.ndarray],
        forwards: List[np.ndarray],
        ranked_indices: List[int],
        source: str,
    ) -> Tuple[Optional[int], Optional[List[np.ndarray]]]:
        """Try ranked candidates in order and return the first A*-reachable one."""
        tried = set()
        for idx in ranked_indices:
            idx = int(idx)
            if idx in tried or idx < 0 or idx >= len(goals):
                continue
            tried.add(idx)
            goal = goals[idx]
            current_path = PATH_PLANNING_REGISTRY[self.path_planning_algorithm](occ, start, goal)
            if current_path is not None and len(current_path) > 0:
                if len(tried) > 1 or source != "semantic":
                    print(f"[path_retry] using {source} candidate {idx} after {len(tried)} path attempts")
                return idx, current_path
            print(
                f"[path_retry] {source} candidate {idx} "
                f"({float(goal[0]):.2f}, {float(goal[2]):.2f}) is not A*-reachable"
            )
        return None, None

    def _goal_tiebreak_values(self, goals: List[np.ndarray]) -> np.ndarray:
        """Forward/sticky preference used only inside near-tied candidate groups."""
        if self.current_asset is None or len(goals) == 0:
            return np.zeros(len(goals), dtype=np.float64)
        cur_pose = self.current_asset["pose"]
        cur_t = cur_pose[:3, 3]
        cur_t = cur_t.cpu().numpy() if isinstance(cur_t, Tensor) else np.asarray(cur_t)
        cur_fwd = cur_pose[:3, 2]
        cur_fwd = cur_fwd.cpu().numpy() if isinstance(cur_fwd, Tensor) else np.asarray(cur_fwd)
        fwd_xz = np.array([float(cur_fwd[0]), float(cur_fwd[2])], dtype=np.float64)
        fwd_norm = float(np.linalg.norm(fwd_xz))
        if fwd_norm < 1e-9:
            fwd_xz = np.array([1.0, 0.0], dtype=np.float64)
        else:
            fwd_xz /= fwd_norm

        prev_goal = self.current_goal
        prev_xz = None
        if prev_goal is not None:
            prev_goal = prev_goal.cpu().numpy() if isinstance(prev_goal, Tensor) else np.asarray(prev_goal)
            prev_xz = np.array([float(prev_goal[0]), float(prev_goal[2])], dtype=np.float64)

        values = []
        for goal in goals:
            goal = goal.cpu().numpy() if isinstance(goal, Tensor) else np.asarray(goal)
            to_goal = np.array([float(goal[0]) - float(cur_t[0]), float(goal[2]) - float(cur_t[2])], dtype=np.float64)
            norm = float(np.linalg.norm(to_goal))
            front = 0.0 if norm < 1e-9 else float(np.clip(np.dot(fwd_xz, to_goal / norm), -1.0, 1.0))
            stick = 0.0
            if prev_xz is not None and self.goal_stick_radius_m > 0:
                goal_xz = np.array([float(goal[0]), float(goal[2])], dtype=np.float64)
                d = float(np.linalg.norm(goal_xz - prev_xz))
                stick = max(0.0, 1.0 - d / self.goal_stick_radius_m)
            values.append(self.forward_tiebreak_weight * front + self.goal_stick_bonus * stick)
        return np.asarray(values, dtype=np.float64)

    def _rank_by_semantic_ties(self, goals: List[np.ndarray], scores: np.ndarray) -> List[int]:
        """Rank by semantic score, resolving near ties with forward/sticky preference."""
        n = min(len(goals), int(scores.shape[0]))
        if n <= 0:
            return []
        scores = scores[:n]
        order = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)
        best_score = float(scores[order[0]])
        tied = [i for i in order if best_score - float(scores[i]) <= self.semantic_tie_margin]
        tie_values = self._goal_tiebreak_values(goals[:n])
        tied = sorted(tied, key=lambda i: (float(tie_values[i]), float(scores[i])), reverse=True)
        rest = [i for i in order if i not in set(tied)]
        return tied + rest

    def _rank_by_hotspot_ties(
        self,
        goals: List[np.ndarray],
        hotspot_dists: List[float],
        semantic_scores: np.ndarray,
    ) -> List[int]:
        """Rank by hotspot distance, resolving local ties with forward/sticky preference."""
        n = min(len(goals), len(hotspot_dists))
        if n <= 0:
            return []
        dists = np.asarray(hotspot_dists[:n], dtype=np.float64)
        scores = np.zeros(n, dtype=np.float64)
        if semantic_scores.size > 0:
            m = min(n, int(semantic_scores.shape[0]))
            scores[:m] = np.asarray(semantic_scores[:m], dtype=np.float64)
        order = sorted(range(n), key=lambda i: float(dists[i]))
        best_dist = float(dists[order[0]])
        tied = [i for i in order if float(dists[i]) - best_dist <= self.hotspot_tie_radius_m]
        tie_values = self._goal_tiebreak_values(goals[:n])
        tied = sorted(
            tied,
            key=lambda i: (float(tie_values[i]), float(scores[i]), -float(dists[i])),
            reverse=True,
        )
        tied_set = set(tied)
        rest = [i for i in order if i not in tied_set]
        return tied + rest

    def select_goal_semantic(self, goals: List[np.ndarray], forwards: List[np.ndarray], initial_pose: Union[Tensor, np.ndarray], object_name: str) -> int:
        """Use 3d belief built-in semantic to select goal based on images and object name."""
        assert self.world_model is not None, "World model must be set for Belief3DModelGoalSelector"
        object_name: str = self.current_asset["object_name"]
        render_output = self.world_model.render_goal_images(goals, forwards, initial_pose, query_label=object_name)  # list of np arrays
        images = render_output.rgb  # list of np arrays
        semantics = render_output.semantic  # list of np arrays
        depths = render_output.depth  # list of np arrays / tensors (may be empty)
        render_poses = render_output.pose  # list of (4, 4) c2w (may be empty)
        semantics_viz = [np.ascontiguousarray(visualize_semantic_query_intensity_map(semantic[0])) for semantic in semantics]  # list of np arrays
        goal_poses = goals_and_forwards_to_poses(goals, forwards)
        self.current_asset["goal_images"] = images
        self.current_asset["goal_semantics"] = semantics_viz
        self.current_asset["goal_poses"] = goal_poses
        self.current_asset["goal_scores"] = [float(np.max(semantics[i])) for i in range(len(semantics))]
        semantic_scores = np.asarray(self.current_asset["goal_scores"], dtype=np.float64)
        semantic_ranking = self._rank_by_semantic_ties(goals, semantic_scores)
        if len(semantic_ranking) == 0:
            semantic_ranking = [0]
        best_idx = min(int(semantic_ranking[0]), len(goals)-1)  # ensure within bounds
        raw_semantic_idx = int(np.argmax(semantic_scores)) if semantic_scores.size > 0 else best_idx
        raw_semantic_idx = min(raw_semantic_idx, len(goals)-1)
        self._last_goal_ranking = semantic_ranking

        if self.goal_selection_strategy == "closest_camera_to_max_pixel" and len(depths) > raw_semantic_idx and len(render_poses) > raw_semantic_idx:
            raw_goal = self._max_pixel_to_world(
                semantic=semantics[raw_semantic_idx],
                depth=depths[raw_semantic_idx],
                pose_c2w=render_poses[raw_semantic_idx],
            )
            if raw_goal is not None:
                # Choose the camera nearest the hotspot.
                dists = [
                    float(np.linalg.norm(np.asarray(g, dtype=np.float64) - raw_goal))
                    for g in goals
                ]
                hotspot_ranking = self._rank_by_hotspot_ties(goals, dists, semantic_scores)
                closest_idx = int(hotspot_ranking[0]) if len(hotspot_ranking) > 0 else int(np.argmin(dists))
                print(
                    f"[closest_camera_to_max_pixel] hotspot=({raw_goal[0]:.3f}, {raw_goal[2]:.3f}); "
                    f"best_score_idx={raw_semantic_idx} closest_idx={closest_idx} "
                    f"dist={dists[closest_idx]:.3f}m"
                )
                best_idx = min(closest_idx, len(goals) - 1)
                self._last_goal_ranking = hotspot_ranking
        elif self.goal_selection_strategy == "max_pixel_3d" and len(depths) > best_idx and len(render_poses) > best_idx:
            raw_goal = self._max_pixel_to_world(
                semantic=semantics[best_idx],
                depth=depths[best_idx],
                pose_c2w=render_poses[best_idx],
            )
            if raw_goal is not None:
                # Snap object pixels to a navigable cell.
                cur_t = self.current_asset["pose"][:3, 3]
                cur_t = cur_t.cpu().numpy() if isinstance(cur_t, Tensor) else np.asarray(cur_t)
                snapped = self._snap_to_reachable(
                    raw_goal,
                    occ=self.current_asset["occupancy"],
                    cur_xz=(float(cur_t[0]), float(cur_t[2])),
                )
                if snapped is not None:
                    new_goal = snapped
                    goals[best_idx] = new_goal.astype(goals[best_idx].dtype)
                    fwd = new_goal - cur_t.astype(new_goal.dtype)
                    fwd[1] = 0.0
                    n = float(np.linalg.norm(fwd))
                    if n > 1e-6:
                        forwards[best_idx] = (fwd / n).astype(forwards[best_idx].dtype)
                    print(
                        f"[max_pixel_3d] back-projected goal -> "
                        f"raw=({raw_goal[0]:.3f}, {raw_goal[2]:.3f}) "
                        f"snapped=({new_goal[0]:.3f}, {new_goal[2]:.3f})"
                    )
                else:
                    # Fall back to the candidate camera.
                    print(
                        f"[max_pixel_3d] no reachable snap for "
                        f"({raw_goal[0]:.3f}, {raw_goal[2]:.3f}); "
                        f"keeping candidate camera position"
                    )
        return best_idx

    def _max_pixel_to_world(self, semantic, depth, pose_c2w):
        """Back-project the semantic peak to a floor-plane world point."""
        sem = semantic.cpu().numpy() if isinstance(semantic, Tensor) else np.asarray(semantic)
        dep = depth.cpu().numpy() if isinstance(depth, Tensor) else np.asarray(depth)
        # Collapse leading dims to (H, W).
        sem2d = sem
        while sem2d.ndim > 2:
            sem2d = sem2d[0]
        dep2d = dep
        while dep2d.ndim > 2:
            dep2d = dep2d[0]
        if sem2d.ndim != 2 or dep2d.ndim != 2 or sem2d.shape != dep2d.shape:
            return None
        H, W = sem2d.shape
        # Ignore invalid depth.
        valid = np.isfinite(dep2d) & (dep2d > 0)
        masked = np.where(valid, sem2d, -np.inf)
        if not np.isfinite(masked).any():
            return None
        v_star, u_star = np.unravel_index(int(np.argmax(masked)), sem2d.shape)
        Z = float(dep2d[v_star, u_star])
        if not np.isfinite(Z) or Z <= 0:
            return None
        # Intrinsics are normalized.
        cam = self.world_model.camera
        fx_pix = float(cam.fx) * W
        fy_pix = float(cam.fy) * H
        cx_pix = float(cam.cx) * W
        cy_pix = float(cam.cy) * H
        X_cam = (float(u_star) - cx_pix) * Z / fx_pix
        Y_cam = (float(v_star) - cy_pix) * Z / fy_pix
        p_cam = np.array([X_cam, Y_cam, Z, 1.0], dtype=np.float64)
        T = pose_c2w.cpu().numpy() if isinstance(pose_c2w, Tensor) else np.asarray(pose_c2w)
        T = T.astype(np.float64)
        p_world = T @ p_cam
        # Path planner expects floor-plane goals.
        return np.array([p_world[0], 0.0, p_world[2]], dtype=np.float64)
    
    def _snap_to_reachable(
        self,
        point_xyz: np.ndarray,
        occ: OccupancyMap,
        cur_xz,
        min_clearance_m: float = 0.25,
        max_search_radius_m: float = 3.0,
    ):
        """Snap a back-projected goal to the nearest reachable free cell."""
        grid = occ.occupancy
        if grid is None:
            return None
        nz, nx = grid.shape
        res = float(occ.resolution)
        max_search_cells = max(1, int(round(max_search_radius_m / res)))

        obstacles = (grid == 1)
        try:
            from scipy.ndimage import distance_transform_edt
            dist_to_obs_m = distance_transform_edt(~obstacles).astype(np.float32) * res
        except Exception:
            dist_to_obs_m = None

        def cell_ok(rr: int, cc: int) -> bool:
            if rr < 0 or rr >= nz or cc < 0 or cc >= nx:
                return False
            if grid[rr, cc] == 1:
                return False
            if dist_to_obs_m is not None and dist_to_obs_m[rr, cc] < min_clearance_m:
                return False
            return True

        gx, gz = float(point_xyz[0]), float(point_xyz[2])
        cx, cz = float(cur_xz[0]), float(cur_xz[1])
        gr, gc = occ._world_to_grid((gx, gz))
        cr, cc = occ._world_to_grid((cx, cz))

        # Already navigable.
        if cell_ok(gr, gc):
            return np.array([gx, 0.0, gz], dtype=np.float64)

        # Walk back from goal toward current cell.
        dr_total = cr - gr
        dc_total = cc - gc
        n_steps = max(abs(dr_total), abs(dc_total))
        if n_steps > 0:
            for s in range(1, n_steps + 1):
                rr = int(round(gr + dr_total * s / n_steps))
                ccol = int(round(gc + dc_total * s / n_steps))
                if cell_ok(rr, ccol):
                    x_snap, z_snap = occ._grid_to_world((rr, ccol))
                    return np.array([x_snap, 0.0, z_snap], dtype=np.float64)

        # Last-resort: square-ring search outward from the goal cell.
        best = None
        best_d2 = None
        for d in range(1, max_search_cells + 1):
            for dr in range(-d, d + 1):
                if abs(dr) == d:
                    dc_iter = range(-d, d + 1)
                else:
                    dc_iter = (-d, d)
                for dc in dc_iter:
                    rr, ccol = gr + dr, gc + dc
                    if not cell_ok(rr, ccol):
                        continue
                    d2 = dr * dr + dc * dc
                    if best is None or d2 < best_d2:
                        best, best_d2 = (rr, ccol), d2
            if best is not None:
                rr, ccol = best
                x_snap, z_snap = occ._grid_to_world((rr, ccol))
                return np.array([x_snap, 0.0, z_snap], dtype=np.float64)
        return None

    def _is_anti_goal(self, x: float, z: float) -> bool:
        """True if (x, z) lies within `anti_goal_radius` of any flagged anti-goal."""
        if not self.anti_goals:
            return False
        r2 = self.anti_goal_radius * self.anti_goal_radius
        for ax, az, _ in self.anti_goals:
            if (x - ax) ** 2 + (z - az) ** 2 <= r2:
                return True
        return False

    def _detect_no_progress_and_flag(self) -> None:
        """Flag the current goal if recent poses show no progress."""
        if self.current_asset is None or self.current_goal is None:
            return
        cur_t = self.current_asset["pose"][:3, 3]
        cur_t = cur_t.cpu().numpy() if isinstance(cur_t, Tensor) else np.asarray(cur_t)
        self._pose_window.append((float(cur_t[0]), float(cur_t[2])))
        if len(self._pose_window) < self.no_progress_window:
            return
        xs = np.fromiter((p[0] for p in self._pose_window), dtype=np.float64)
        zs = np.fromiter((p[1] for p in self._pose_window), dtype=np.float64)
        span = float(max(xs.max() - xs.min(), zs.max() - zs.min()))
        if span >= self.no_progress_min_displacement_m:
            return
        gx, gz = float(self.current_goal[0]), float(self.current_goal[2])
        # Avoid flagging the same goal repeatedly.
        if not self._is_anti_goal(gx, gz):
            self.anti_goals.append((gx, gz, int(self.step)))
            print(
                f"[no_progress] flagging goal ({gx:.2f}, {gz:.2f}) as anti-goal; "
                f"span={span:.2f}m over {self.no_progress_window} steps "
                f"(total anti_goals={len(self.anti_goals)})"
            )
        self._stuck_commit_count += 1
        self.replan = True
        self._pose_window.clear()

    def _check_progress_and_reset_counter(self) -> None:
        """Reset the stuck counter after sufficient movement."""
        if self.current_asset is None:
            return
        cur_t = self.current_asset["pose"][:3, 3]
        cur_t = cur_t.cpu().numpy() if isinstance(cur_t, Tensor) else np.asarray(cur_t)
        cur_xz = (float(cur_t[0]), float(cur_t[2]))
        if self._last_progress_xz is None:
            self._last_progress_xz = cur_xz
            return
        dx = cur_xz[0] - self._last_progress_xz[0]
        dz = cur_xz[1] - self._last_progress_xz[1]
        if (dx * dx + dz * dz) >= 4.0:  # moved >= 2m from last progress anchor
            if self._stuck_commit_count > 0:
                print(
                    f"[progress] moved >=2m from last anchor; "
                    f"resetting stuck_commit_count from {self._stuck_commit_count} to 0"
                )
            self._stuck_commit_count = 0
            self._last_progress_xz = cur_xz

    def set_world_model(self, world_model: BaseWorldModel) -> None:
        """Set the world model for the planner."""
        self.world_model = world_model
