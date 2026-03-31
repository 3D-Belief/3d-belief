from typing import Any

from .backbone import Backbone
from .dit3d_pose import DiT3DPose, BackboneDiT3DPoseCfg
from .u_vit3d_pose import UViT3DPose, BackboneUViT3DPoseCfg
from .u_vit3d_pose_sg import UViT3DPoseSG, BackboneUViT3DPoseSGCfg

BACKBONES: dict[str, Backbone[Any]] = {
    "dit3d_pose": DiT3DPose,
    "u_vit3d_pose": UViT3DPose,
    "u_vit3d_pose_sg": UViT3DPoseSG,
}

BackboneCfg = BackboneDiT3DPoseCfg | BackboneUViT3DPoseCfg | BackboneUViT3DPoseSGCfg


def get_backbone(cfg: BackboneCfg, d_in: int) -> Backbone[Any]:
    return BACKBONES[cfg.name](cfg, d_in)
