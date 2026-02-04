import copy

from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Dataset
from layers import T5Encoder

from data_io.composite import CompositeDataset

from data_io.realestate import RealEstate10kDatasetOM
from data_io.realestate_seq import RealEstate10kDatasetSeq
from data_io.hm3d import HM3DDataset
from data_io.hm3d_seq import HM3DDatasetSeq
from data_io.spoc import SPOCDataset
from data_io.spoc_seq import SPOCDatasetSeq
from data_io.dl3dv import DL3DVDataset

def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))

def get_path(dataset_name: str) -> str:
    if dataset_name == "realestate":
        return "/projects/u5aa/public/re10k"
    if dataset_name == "realestate_seq":
        return "/projects/u5aa/public/re10k"
    elif dataset_name == "hm3d":
        return "/home/ubuntu/VLMP/tianmin-project/yyin34/dataset/HM3D-dataset-non-sem"
    elif dataset_name == "hm3d_seq":
        return "/home/ubuntu/VLMP/tianmin-project/yyin34/dataset/eval_dataset"
    elif dataset_name == "spoc":
        return "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories"
    elif dataset_name == "spoc_seq":
        return "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories"
    elif dataset_name == "dl3dv":
        return "/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/dataset/dl3dv"
    raise NotImplementedError(f'Dataset "{dataset_name}" not supported.')

def get_dataset(config: DictConfig) -> Dataset:
    name = config.dataset.name  # <-- do NOT delete it anymore

    # -----------------------------
    # composite dataset branch
    # -----------------------------
    if name == "composite":
        # Expect:
        # config.dataset.datasets = [
        #   {name: "spoc", weight: 0.7, root_dir: "...", <optional overrides>...},
        #   {name: "dl3dv", weight: 0.3, root_dir: "...", <optional overrides>...},
        # ]
        specs = list(config.dataset.datasets)
        assert len(specs) > 0, "dataset.datasets must be a non-empty list for composite."

        # Share one language encoder instance across children to save memory/time
        language_encoder = T5Encoder()

        children = []
        weights = []
        for spec in specs:
            spec = OmegaConf.to_container(spec, resolve=False)
            child_name = spec["name"]
            weights.append(float(spec.get("weight", 1.0)))

            # Create a child config = base config + overrides from spec
            child_cfg = _clone_cfg(config)
            child_cfg.dataset.name = child_name

            # Apply overrides:
            # - If spec has root_dir/raw_dataset_dir/etc, put them under child_cfg.dataset.*
            # - If spec has other keys, treat them as top-level overrides (num_context, image_size, etc.)
            for k, v in spec.items():
                if k in ("name", "weight"):
                    continue
                if k in ("root_dir", "raw_dataset_dir"):
                    child_cfg.dataset[k] = v
                else:
                    # allow overriding common training params, e.g. image_size, num_context...
                    child_cfg[k] = v

            # Build the child dataset by calling the same factory,
            # but making sure it uses the shared language encoder.
            # Easiest: temporarily inject it by setting a flag and passing into constructors below.
            child_ds = _get_single_dataset(child_cfg, language_encoder=language_encoder)
            children.append((child_name, child_ds))

        return CompositeDataset(
            children,
            weights=weights,
            length=int(getattr(config.dataset, "length", None) or 0) or None,
            sample_index_within_dataset=str(getattr(config.dataset, "sample_index", "random")),
            return_dataset_name=bool(getattr(config.dataset, "return_dataset_name", False)),
            base_seed=int(getattr(config.dataset, "seed", 0)),
            image_size=config.image_size,
        )

    # -----------------------------
    # Existing single-dataset logic
    # -----------------------------
    return _get_single_dataset(config, language_encoder=None)

def _get_single_dataset(config: DictConfig, language_encoder=None) -> Dataset:
    
    name = config.dataset.name

    def _get_lang():
        return language_encoder if language_encoder is not None else T5Encoder()

    if name == "realestate":
        paths = config.dataset.root_dir if config.dataset.root_dir!="" else get_path(name)
        ctxt_min = config.dataset.ctxt_min if hasattr(config.dataset, "ctxt_min") else config.ctxt_min
        ctxt_max = config.dataset.ctxt_max if hasattr(config.dataset, "ctxt_max") else config.ctxt_max
        le = _get_lang()
        return RealEstate10kDatasetOM(
            root=paths,
            num_context=config.num_context,
            num_target=config.num_target,
            context_min_distance=ctxt_min,
            context_max_distance=ctxt_max,
            max_scenes=config.max_scenes,
            stage=config.stage,
            image_size=config.image_size,
            intermediate=config.intermediate,
            num_intermediate=config.num_intermediate,
            language_encoder=le,
            adjacent_angle=config.adjacent_angle,
            overfit_to_index=config.overfit_to_index,
            use_depth_supervision=config.use_depth_supervision,
        )
    elif name == "realestate_seq":
        paths = config.dataset.root_dir if config.dataset.root_dir!="" else get_path(name)
        ctxt_min = config.dataset.ctxt_min if hasattr(config.dataset, "ctxt_min") else config.ctxt_min
        ctxt_max = config.dataset.ctxt_max if hasattr(config.dataset, "ctxt_max") else config.ctxt_max
        le = _get_lang()
        return RealEstate10kDatasetSeq(
            root=paths,
            num_context=config.num_context,
            num_target=config.num_target,
            context_min_distance=config.ctxt_min,
            context_max_distance=config.ctxt_max,
            max_scenes=config.max_scenes,
            stage=config.stage,
            image_size=config.image_size,
            language_encoder=le,
            overfit_to_index=config.overfit_to_index,
            adjacent_angle=config.adjacent_angle,
            adjacent_distance=config.adjacent_distance,
            num_intermediate=config.num_intermediate,
            use_depth_supervision=config.use_depth_supervision,
        )
    elif name == "hm3d":
        paths = config.dataset.root_dir if config.dataset.root_dir!="" else get_path(name)
        ctxt_min = config.dataset.ctxt_min if hasattr(config.dataset, "ctxt_min") else config.ctxt_min
        ctxt_max = config.dataset.ctxt_max if hasattr(config.dataset, "ctxt_max") else config.ctxt_max
        le = _get_lang()
        return HM3DDataset(
            root=paths,
            num_context=config.num_context,
            num_target=config.num_target,
            context_min_distance=config.ctxt_min,
            context_max_distance=config.ctxt_max,
            max_scenes=config.max_scenes,
            stage=config.stage,
            image_size=config.image_size,
            language_encoder=le,
            use_first_frame_prob=config.clevr_first_frame_prob,
            start_frame_id=config.clevr_start_frame_id,
            raw_dataset_dir=config.dataset.raw_dataset_dir,
            semantic_config=config.semantic_config,
            adjacent_angle=config.adjacent_angle,
            intermediate=config.intermediate,
            num_intermediate=config.num_intermediate,
            use_depth_supervision=config.use_depth_supervision,
        )
    elif name == "hm3d_seq":
        paths = config.dataset.root_dir if config.dataset.root_dir!="" else get_path(name)
        ctxt_min = config.dataset.ctxt_min if hasattr(config.dataset, "ctxt_min") else config.ctxt_min
        ctxt_max = config.dataset.ctxt_max if hasattr(config.dataset, "ctxt_max") else config.ctxt_max
        le = _get_lang()
        return HM3DDatasetSeq(
            root=paths,
            num_context=config.num_context,
            num_target=config.num_target,
            context_min_distance=config.ctxt_min,
            context_max_distance=config.ctxt_max,
            max_scenes=config.max_scenes,
            stage=config.stage,
            image_size=config.image_size,
            language_encoder=le,
            use_first_frame_prob=config.clevr_first_frame_prob,
            start_frame_id=config.clevr_start_frame_id,
            raw_dataset_dir=config.dataset.raw_dataset_dir,
            semantic_config=config.semantic_config,
            adjacent_angle=config.adjacent_angle,
            adjacent_distance=config.adjacent_distance,
            num_intermediate=config.num_intermediate,
        )
    elif name == "spoc":
        paths = config.dataset.root_dir if config.dataset.root_dir!="" else get_path(name)
        ctxt_min = config.dataset.ctxt_min if hasattr(config.dataset, "ctxt_min") else config.ctxt_min
        ctxt_max = config.dataset.ctxt_max if hasattr(config.dataset, "ctxt_max") else config.ctxt_max
        le = _get_lang()
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
            language_encoder=le,
            adjacent_angle=config.adjacent_angle,
            overfit_to_index=config.overfit_to_index,
            use_depth_supervision=config.use_depth_supervision,
        )
    elif name == "spoc_seq":
        paths = config.dataset.root_dir if config.dataset.root_dir!="" else get_path(name)
        ctxt_min = config.dataset.ctxt_min if hasattr(config.dataset, "ctxt_min") else config.ctxt_min
        ctxt_max = config.dataset.ctxt_max if hasattr(config.dataset, "ctxt_max") else config.ctxt_max
        le = _get_lang()
        return SPOCDatasetSeq(
            root=paths,
            num_context=config.num_context,
            num_target=config.num_target,
            context_min_distance=config.ctxt_min,
            context_max_distance=config.ctxt_max,
            max_scenes=config.max_scenes,
            stage=config.stage,
            image_size=config.image_size,
            language_encoder=le,
            overfit_to_index=config.overfit_to_index,
            adjacent_angle=config.adjacent_angle,
            adjacent_distance=config.adjacent_distance,
            num_intermediate=config.num_intermediate,
        )
    elif name == "dl3dv":
        paths = config.dataset.root_dir if config.dataset.root_dir!="" else get_path(name)
        ctxt_min = config.dataset.ctxt_min if hasattr(config.dataset, "ctxt_min") else config.ctxt_min
        ctxt_max = config.dataset.ctxt_max if hasattr(config.dataset, "ctxt_max") else config.ctxt_max
        le = _get_lang()
        return DL3DVDataset(
            root=paths,
            num_context=config.num_context,
            num_target=config.num_target,
            context_min_distance=config.ctxt_min,
            context_max_distance=config.ctxt_max,
            max_scenes=config.max_scenes,
            stage=config.stage,
            image_size=config.image_size,
            language_encoder=le,
            intermediate=config.intermediate,
            num_intermediate=config.num_intermediate,
            adjacent_angle=config.adjacent_angle,
            use_depth_supervision=config.use_depth_supervision,
        )
    raise NotImplementedError(f'Dataset "{name}" not supported.')
