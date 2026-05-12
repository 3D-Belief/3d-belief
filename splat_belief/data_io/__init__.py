import copy

from omegaconf import OmegaConf, DictConfig
from typing import Any

def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))

def get_path(dataset_name: str) -> str:
    if dataset_name == "realestate":
        return ""
    if dataset_name == "realestate_seq":
        return ""
    elif dataset_name == "spoc":
        return ""
    elif dataset_name == "spoc_seq":
        return ""
    raise NotImplementedError(f'Dataset "{dataset_name}" not supported.')

def get_dataset(config: DictConfig) -> Any:
    name = config.dataset.name  # <-- do NOT delete it anymore

    # -----------------------------
    # composite dataset branch
    # -----------------------------
    if name == "composite":
        from data_io.composite import CompositeDataset
        from splat_belief.splat.layers import T5Encoder

        # Expect:
        # config.dataset.datasets = [
        #   {name: "spoc", weight: 1.0, root_dir: "...", <optional overrides>...},
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

def _get_single_dataset(config: DictConfig, language_encoder=None) -> Any:
    
    name = config.dataset.name

    def _get_lang():
        from splat_belief.splat.layers import T5Encoder

        return language_encoder if language_encoder is not None else T5Encoder()

    if name == "realestate":
        from data_io.realestate import RealEstate10kDatasetOM

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
            adjacent_distance=config.adjacent_distance,
            overfit_to_index=config.overfit_to_index,
            use_depth_supervision=config.use_depth_supervision,
        )
    elif name == "realestate_seq":
        from data_io.realestate_seq import RealEstate10kDatasetSeq

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
    elif name == "spoc":
        from data_io.spoc import SPOCDataset

        paths = config.dataset.root_dir if config.dataset.root_dir!="" else get_path(name)
        ctxt_min = config.dataset.ctxt_min if hasattr(config.dataset, "ctxt_min") else config.ctxt_min
        ctxt_max = config.dataset.ctxt_max if hasattr(config.dataset, "ctxt_max") else config.ctxt_max
        le = _get_lang()
        # Class-id maps are needed when supervision uses class-text targets.
        load_class_maps = False
        class_name_to_id = None
        try:
            mode = config.semantic_supervision_mode
            if mode in ("class_text_only", "class_text_image"):
                load_class_maps = True
                tt = config.get("class_text_table_path", None)
                if tt:
                    import torch as _torch
                    payload = _torch.load(tt, map_location="cpu", weights_only=False)
                    class_name_to_id = payload["name_to_id"]
        except Exception:
            pass
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
            adjacent_distance=config.adjacent_distance,
            overfit_to_index=config.overfit_to_index,
            use_depth_supervision=config.use_depth_supervision,
            load_class_maps=load_class_maps,
            class_name_to_id=class_name_to_id,
        )
    elif name == "spoc_seq":
        from data_io.spoc_seq import SPOCDatasetSeq

        paths = config.dataset.root_dir if config.dataset.root_dir!="" else get_path(name)
        ctxt_min = config.dataset.ctxt_min if hasattr(config.dataset, "ctxt_min") else config.ctxt_min
        ctxt_max = config.dataset.ctxt_max if hasattr(config.dataset, "ctxt_max") else config.ctxt_max
        le = language_encoder
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
    raise NotImplementedError(f'Dataset "{name}" not supported.')
