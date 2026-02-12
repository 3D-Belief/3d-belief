from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import json
from torch import Tensor
from pathlib import Path
from PIL import Image
import open3d as o3d
from rollout_utils import visualize_semantic_query_intensity_map
from wm_baselines.env_interface.base_env_interface import BaseEnvInterface
from wm_baselines.world_model.base_world_model import BaseWorldModel
from wm_baselines.planner.base_planner import BasePlanner
from wm_baselines.agent.base_agent import BaseAgent
from wm_baselines.agent.perception.camera import Camera
from wm_baselines.agent.perception.metrics import (
    bev_iou, iou_3d, box_errors, chamfer_np, clip_similarity, siglip_similarity, lpips_distance, default_chamfer_from_gt
)
from wm_baselines.utils.vision_utils import to_pil

class ObjCompletionReasoningAgent(BaseAgent):
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
        self.current_object_name = None

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
            query_label=self.current_object_name
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
            "imagine_colored_pcd": colored_pcd,
        })
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for the current episode."""
        gt_ret = self.env_interface.task_manager._calculate_gt_metrics()
        self.current_assets.update(**gt_ret)
        rendered_ret = self.env_interface.task_manager._calculate_rendered_metrics(self.current_assets, seed=gt_ret.get("target_seed", None))
        gt_bbox_3d = gt_ret.get("gt_bbox_3d", None)
        rendered_bbox_3d = rendered_ret.get("rendered_bbox_3d", None)

        metric_bev_iou = bev_iou(rendered_bbox_3d, gt_bbox_3d) if (rendered_bbox_3d is not None and gt_bbox_3d is not None) else 0.0
        metric_iou_3d = iou_3d(rendered_bbox_3d, gt_bbox_3d) if (rendered_bbox_3d is not None and gt_bbox_3d is not None) else 0.0

        gt_pcd = gt_ret["gt_target_pcd"].points
        rendered_pcd = rendered_ret["rendered_target_pcd"].points
        if rendered_pcd is not None and len(rendered_pcd) > 0:
            metric_chamfer = chamfer_np(np.array(rendered_pcd), np.array(gt_pcd))
        else:
            metric_chamfer = float(default_chamfer_from_gt(gt_pcd, chamfer_fn=chamfer_np))

        img_rendered_crop = rendered_ret["rendered_bbox_image"]
        img_gt_crop = gt_ret["gt_bbox_image"]
        if img_rendered_crop is None or img_gt_crop is None:
            metric_clip_similarity = 0.0
            metric_siglip_similarity = 0.0
            lpips_dist = 1.0
        else:
            metric_clip_similarity = clip_similarity(
                img_rendered_crop,
                img_gt_crop,
            )["similarity"]
            metric_siglip_similarity = siglip_similarity(
                img_rendered_crop,
                img_gt_crop,
            )["similarity"]
            lpips_dist = lpips_distance(
                img_rendered_crop,
                img_gt_crop,
            )["distance"]

        task_manager = self.env_interface.task_manager
        metrics = {
            "visibility": task_manager.visibility_percents[task_manager._current_step-1],
            "metric_bev_iou": metric_bev_iou,
            "metric_iou_3d": metric_iou_3d,
            "metric_chamfer": metric_chamfer,
            "clip_similarity": metric_clip_similarity,
            "siglip_similarity": metric_siglip_similarity,
            "lpips_distance": lpips_dist,
        }
        vlm_obj_recognition = rendered_ret.get("vlm_obj_recognition", None)
        if vlm_obj_recognition is not None:
            metrics.update({
                "vlm_obj_recognition": vlm_obj_recognition,
            })
        self.current_assets["metrics"] = metrics
        self.current_assets["gt_bbox_image"] = gt_ret["gt_bbox_image"]
        self.current_assets["rendered_bbox_image"] = rendered_ret["rendered_bbox_image"]
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
        self.current_object_name = self.env_interface.task_manager.fix_object_name(self.env_interface.task_manager.current_target_obj)
        self._setup_assets_save()
        self.force_update = False
    
    def _step_assets_save(self, trace_asset: Dict[str, Any]) -> None:
        """Return the current step's assets for saving/logging."""
        if self.save_assets and self.assets_save_path and self.assets_save_path_ep:
            # Save the specified assets to the designated path
            for key in self.assets_save_keys:
                if key=='pcd':
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