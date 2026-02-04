from typing import Optional

from .encoder import Encoder
from .encoder_epidiff import EncoderEpiDiffusion, EncoderEpiDiffusionCfg
from .encoder_costvolume import EncoderCostVolume, EncoderCostVolumeCfg
from .encoder_videodiff import EncoderVideoDiffusion, EncoderVideoDiffusionCfg
from .encoder_uvitdiff import EncoderUViTDiffusion, EncoderUViTDiffusionCfg
from .encoder_ditmvsplat import EncoderDiTMVSplat, EncoderDiTMVSplatCfg
from .encoder_uvitmvsplat import EncoderUViTMVSplat, EncoderUViTMVSplatCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_costvolume import EncoderVisualizerCostVolume

ENCODERS = {
    "epidiff": (EncoderEpiDiffusion, None),
    "costvolume": (EncoderCostVolume, EncoderVisualizerCostVolume),
    "videodiff": (EncoderVideoDiffusion, None),
    "uvitdiff": (EncoderUViTDiffusion, None),
    "ditmvsplat": (EncoderDiTMVSplat, None),
    "uvitmvsplat": (EncoderUViTMVSplat, None),
}

EncoderCfg = EncoderEpiDiffusionCfg | EncoderCostVolumeCfg | EncoderVideoDiffusionCfg | EncoderUViTDiffusionCfg | EncoderDiTMVSplatCfg | EncoderUViTMVSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
