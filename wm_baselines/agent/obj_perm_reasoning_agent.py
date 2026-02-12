from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import json
from torch import Tensor
from pathlib import Path
from PIL import Image
import open3d as o3d
from rollout_utils import visualize_semantic_query_intensity_map
from wm_baselines.agent.perception.occupancy import OccupancyMap
from wm_baselines.env_interface.base_env_interface import BaseEnvInterface
from wm_baselines.world_model.base_world_model import BaseWorldModel
from wm_baselines.planner.base_planner import BasePlanner
from wm_baselines.agent.base_agent import BaseAgent
from wm_baselines.agent.perception.camera import Camera
from wm_baselines.agent.perception.metrics import (
    eval_occupancy_maps, clip_similarity, siglip_similarity, lpips_distance
)
from wm_baselines.utils.vision_utils import to_pil

class ObjPermanenceReasoningAgent(BaseAgent):
    """Agent that uses belief reasoning for decision making."""

    def __init__(self,
                 env_interface: BaseEnvInterface, 
                 world_model: BaseWorldModel, 
                 planner: BasePlanner,
                 camera: Camera,
                 assets_saver: Dict[str, Any],
                 **kwargs
        ):
        super().__init__(env_interface, world_model, planner, camera=camera, assets_saver=assets_saver)
        self.current_ep_name = None
        self.current_destination_room = None

    def observe(self) -> Dict[str, Any]: #  complete
        """Get the current observation and return the observation dict."""
        self.world_model.reset()
        obs = self.env_interface.task_manager.get_observation()
        assets = self.world_model.update_observation(obs, self.force_update)
        self.current_observation = obs
        self.current_assets = assets
        self.step += 1
        return obs

    def imagine(self) -> Any:
        """Imagine along the remaining path poses in the episode."""
        imagine_poses = self.env_interface.task_manager.imagination_poses
        (key_output, full_output, colored_pcd), exe_time = self.world_model._imagine_along_path(
            imagine_poses=imagine_poses,
            return_full_video=True,
            return_colored_pcd=True,
            query_label=""
        )
        rgb = key_output.rgb
        depth = key_output.depth
        semantics = key_output.semantic
        if semantics is not None:
            semantics_viz = [np.ascontiguousarray(visualize_semantic_query_intensity_map(semantic[0])) for semantic in semantics]  # list of np arrays
        else:
            semantics_viz = None
        poses = key_output.pose
        scores = [np.max(semantics[i]) for i in range(len(semantics))] if semantics is not None else []
        self.current_assets.update({
            "imagine_rgb": rgb,
            "imagine_depth": depth,
            "imagine_semantics": semantics_viz,
            "imagine_semantics_raw": semantics,
            "imagine_poses": poses,
            "imagine_scores": scores,
            "imagine_video": full_output.rgb if full_output is not None else None,
            "imagine_depth_video": full_output.depth if full_output is not None else None,
            "imagine_semantics_video": full_output.semantic if full_output is not None else None,
            "imagine_pcd": colored_pcd,
        })
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for the current episode."""
        gt_ret = self.env_interface.task_manager._calculate_gt_metrics()
        rendered_ret = self.env_interface.task_manager._calculate_rendered_metrics(self.current_assets)
        belief_occupancy = rendered_ret["belief_occupancy"]
        gt_occupancy = gt_ret["gt_occupancy"]
        occupancy_metrics = eval_occupancy_maps(
            gt_map=gt_occupancy,
            pred_map=belief_occupancy
        )
        imagined_rgb_before = self.current_assets["imagine_video"][0]
        imagined_rgb_after = self.current_assets["imagine_video"][-1]
        lpips_dist = lpips_distance(
            imagined_rgb_before,
            imagined_rgb_after,
        )["distance"]
        metric_siglip_similarity = siglip_similarity(
            imagined_rgb_before,
            imagined_rgb_after,
        )["similarity"]

        metrics = {
            "lpips_distance": lpips_dist,
            "siglip_similarity": metric_siglip_similarity,
        }
        self.current_assets["metrics"] = metrics
        self.current_assets["belief_occupancy"] = belief_occupancy
        self.current_assets["gt_occupancy"] = gt_occupancy
        return metrics

    def save_step(self) -> None:
        self._step_assets_save(self.current_assets)

    def reset(self):
        """Reset the agent state if needed. Subclasses can override."""
        self.world_model.reset()
        self.current_observation = None
        self.current_assets = None
        self.step = -1
        self.current_ep_name = self.env_interface.task_manager.current_ep_name
        self._setup_assets_save()
        self.force_update = False
    
    def _step_assets_save(self, trace_asset: Dict[str, Any]) -> None:
        """Return the current step's assets for saving/logging."""
        if self.save_assets and self.assets_save_path and self.assets_save_path_ep:
            # Save the specified assets to the designated path
            for key in self.assets_save_keys:
                if key=='belief_occupancy':
                    occupancy = trace_asset.get('belief_occupancy', None)
                    if occupancy is None: continue
                    occupancy.save_height_map(self.assets_save_path_ep / key / f"height_map_{self.step}.png")
                    occupancy.save_occupancy_map(self.assets_save_path_ep / key / f"occupancy_map_{self.step}.png")
                elif key=='gt_occupancy':
                    occupancy = trace_asset.get('gt_occupancy', None)
                    if occupancy is None: continue
                    occupancy.save_height_map(self.assets_save_path_ep / key / f"height_map_{self.step}.png")
                    occupancy.save_occupancy_map(self.assets_save_path_ep / key / f"occupancy_map_{self.step}.png")
                elif key=='pcd':
                    if trace_asset.get('pcd', None) is None: continue
                    o3d.io.write_point_cloud(str(self.assets_save_path_ep / key / f"pcd_{self.step}.ply"), trace_asset['pcd'])
                elif key=='rgb':
                    if trace_asset.get('rgb', None) is None: continue
                    rgb = trace_asset['rgb']
                    rgb = Image.fromarray(rgb)
                    rgb.save(self.assets_save_path_ep / key / f"rgb_{self.step}.png")
                elif key=='imagine_rgb':
                    imagine_rgb = trace_asset.get('imagine_rgb', None)
                    if imagine_rgb is None: continue
                    for i, img in enumerate(imagine_rgb):
                        img = to_pil(img)
                        img.save(self.assets_save_path_ep / key / f"imagine_rgb_{self.step}_{i}.png")
                elif key=='imagine_video':
                    imagine_video = trace_asset.get('imagine_video', None)
                    if imagine_video is None: continue
                    for i, img in enumerate(imagine_video):
                        img = to_pil(img)
                        img.save(self.assets_save_path_ep / key / f"imagine_video_{self.step}_{i}.png")
                elif key=='imagine_semantics':
                    imagine_semantics = trace_asset.get('imagine_semantics', None)
                    if imagine_semantics is None: continue
                    for i, img in enumerate(imagine_semantics):
                        img = to_pil(img)
                        img.save(self.assets_save_path_ep / key / f"imagine_semantic_{self.step}_{i}.png")
                elif key=='imagine_scores':
                    imagine_scores = trace_asset.get('imagine_scores', None)
                    if imagine_scores is None: continue
                    with open(self.assets_save_path_ep / key / f"imagine_scores_{self.step}.txt", 'w') as f:
                        for score in imagine_scores:
                            f.write(f"{score}\n")
                elif key=='imagine_poses':
                    imagine_poses = trace_asset.get('imagine_poses', None)
                    if imagine_poses is None: continue
                    with open(self.assets_save_path_ep / key / f"imagine_poses_{self.step}.json", 'w') as f:
                        json.dump([pose.tolist() for pose in imagine_poses], f, indent=4)
                elif key=='metrics':
                    metrics = trace_asset.get('metrics', None)
                    if metrics is None: continue
                    with open(self.assets_save_path_ep / key / f"metrics_{self.step}.json", 'w') as f:
                        json.dump(metrics, f, indent=4)
                elif key=='gt_bbox_image':
                    gt_bbox_image = trace_asset.get('gt_bbox_image', None)
                    if gt_bbox_image is None: continue
                    img = to_pil(gt_bbox_image)
                    img.save(self.assets_save_path_ep / key / f"gt_bbox_image_{self.step}.png")
                elif key=='rendered_bbox_image':
                    rendered_bbox_image = trace_asset.get('rendered_bbox_image', None)
                    if rendered_bbox_image is None: continue
                    img = to_pil(rendered_bbox_image)
                    img.save(self.assets_save_path_ep / key / f"rendered_bbox_image_{self.step}.png")