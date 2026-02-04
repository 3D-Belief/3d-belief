from typing import Any

from .backbone import Backbone
from .DiT import DiT, BackboneDiTCfg
from .dit3d_pose import DiT3DPose, BackboneDiT3DPoseCfg
from .u_vit3d_pose import UViT3DPose, BackboneUViT3DPoseCfg

BACKBONES: dict[str, Backbone[Any]] = {
    "dit": DiT,
    "dit3d_pose": DiT3DPose,
    "u_vit3d_pose": UViT3DPose,
}

BackboneCfg = BackboneDiTCfg | BackboneDiT3DPoseCfg | BackboneUViT3DPoseCfg


def get_backbone(cfg: BackboneCfg, d_in: int) -> Backbone[Any]:
    return BACKBONES[cfg.name](cfg, d_in)
