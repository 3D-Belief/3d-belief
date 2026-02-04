from omegaconf import DictConfig
from torch.utils.data import Dataset

from data_io.realestate import RealEstate10kDatasetOM
from data_io.realestate_seq import RealEstate10kDatasetSeq
from data_io.hm3d import HM3DDataset
from data_io.hm3d_seq import HM3DDatasetSeq
from data_io.spoc import SPOCDataset
from data_io.spoc_seq import SPOCDatasetSeq
from splat.layers import T5Encoder

def get_path(dataset_name: str) -> str:
    if dataset_name == "realestate":
        return "/scratch/tshu2/yyin34/projects/3d_belief/DFM/data/realestate_full"
    if dataset_name == "realestate_seq":
        return "/scratch/tshu2/yyin34/projects/3d_belief/DFM/data/realestate_full"
    elif dataset_name == "hm3d":
        return "/home/ubuntu/VLMP/tianmin-project/yyin34/dataset/HM3D-dataset-non-sem"
    elif dataset_name == "hm3d_seq":
        return "/home/ubuntu/VLMP/tianmin-project/yyin34/dataset/eval_dataset"
    elif dataset_name == "spoc":
        return "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories"
    elif dataset_name == "spoc_seq":
        return "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories"
    raise NotImplementedError(f'Dataset "{dataset_name}" not supported.')


def get_dataset(config: DictConfig) -> Dataset:
    name = config.dataset.name
    del config.dataset.name

    if name == "realestate":
        paths = get_path(name)
        language_encoder = T5Encoder()
        return RealEstate10kDatasetOM(
            root=paths,
            num_context=config.num_context,
            num_target=config.num_target,
            context_min_distance=config.ctxt_min,
            context_max_distance=config.ctxt_max,
            max_scenes=config.max_scenes,
            stage=config.stage,
            image_size=config.image_size,
            intermediate=config.intermediate,
            num_intermediate=config.num_intermediate,
            language_encoder=language_encoder,
            adjacent_angle=config.adjacent_angle,
            overfit_to_index=config.overfit_to_index,
            use_depth_supervision=config.use_depth_supervision,
        )
    elif name == "realestate_seq":
        paths = get_path(name)
        language_encoder = T5Encoder()
        return RealEstate10kDatasetSeq(
            root=paths,
            num_context=config.num_context,
            num_target=config.num_target,
            context_min_distance=config.ctxt_min,
            context_max_distance=config.ctxt_max,
            max_scenes=config.max_scenes,
            stage=config.stage,
            image_size=config.image_size,
            language_encoder=language_encoder,
            overfit_to_index=config.overfit_to_index,
            adjacent_angle=config.adjacent_angle,
            adjacent_distance=config.adjacent_distance,
            num_intermediate=config.num_intermediate,
            use_depth_supervision=config.use_depth_supervision,
        )
    elif name == "hm3d":
        paths = get_path(name)
        language_encoder = T5Encoder()
        return HM3DDataset(
            root=paths,
            num_context=config.num_context,
            num_target=config.num_target,
            context_min_distance=config.ctxt_min,
            context_max_distance=config.ctxt_max,
            max_scenes=config.max_scenes,
            stage=config.stage,
            image_size=config.image_size,
            language_encoder=language_encoder,
            use_first_frame_prob=config.clevr_first_frame_prob,
            start_frame_id=config.clevr_start_frame_id,
            raw_dataset_dir=config.dataset.raw_dataset_dir,
            semantic_config=config.semantic_config,
            adjacent_angle=config.adjacent_angle,
            intermediate=config.intermediate,
            num_intermediate=config.num_intermediate,
        )
    elif name == "hm3d_seq":
        paths = get_path(name)
        language_encoder = T5Encoder()
        return HM3DDatasetSeq(
            root=paths,
            num_context=config.num_context,
            num_target=config.num_target,
            context_min_distance=config.ctxt_min,
            context_max_distance=config.ctxt_max,
            max_scenes=config.max_scenes,
            stage=config.stage,
            image_size=config.image_size,
            language_encoder=language_encoder,
            use_first_frame_prob=config.clevr_first_frame_prob,
            start_frame_id=config.clevr_start_frame_id,
            raw_dataset_dir=config.dataset.raw_dataset_dir,
            semantic_config=config.semantic_config,
            adjacent_angle=config.adjacent_angle,
            adjacent_distance=config.adjacent_distance,
            num_intermediate=config.num_intermediate,
        )
    elif name == "spoc":
        paths = get_path(name)
        language_encoder = T5Encoder()
        return SPOCDataset(
            root=paths,
            num_context=config.num_context,
            num_target=config.num_target,
            context_min_distance=config.ctxt_min,
            context_max_distance=config.ctxt_max,
            max_scenes=config.max_scenes,
            stage=config.stage,
            image_size=config.image_size,
            intermediate=config.intermediate,
            num_intermediate=config.num_intermediate,
            language_encoder=language_encoder,
            adjacent_angle=config.adjacent_angle,
            overfit_to_index=config.overfit_to_index,
        )
    elif name == "spoc_seq":
        paths = get_path(name)
        language_encoder = T5Encoder()
        return SPOCDatasetSeq(
            root=paths,
            num_context=config.num_context,
            num_target=config.num_target,
            context_min_distance=config.ctxt_min,
            context_max_distance=config.ctxt_max,
            max_scenes=config.max_scenes,
            stage=config.stage,
            image_size=config.image_size,
            language_encoder=language_encoder,
            overfit_to_index=config.overfit_to_index,
            adjacent_angle=config.adjacent_angle,
            adjacent_distance=config.adjacent_distance,
            num_intermediate=config.num_intermediate,
        )
    raise NotImplementedError(f'Dataset "{name}" not supported.')
