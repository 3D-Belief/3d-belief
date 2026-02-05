from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


class CompositeDataset(Dataset):
    """
    Samples from one of N datasets at each __getitem__.

    - Choose dataset according to `weights` (normalized).
    - Choose sample index within that dataset either:
        * random (recommended if underlying datasets already ignore `index`), or
        * deterministic via modulo.

    Optionally returns dataset name for logging/debug.
    """

    def __init__(
        self,
        datasets: Sequence[Tuple[str, Dataset]],
        weights: Optional[Sequence[float]] = None,
        *,
        length: Optional[int] = None,
        sample_index_within_dataset: str = "random",  # "random" | "mod"
        return_dataset_name: bool = False,
        base_seed: int = 0,
        image_size: Union[int, Tuple[int, int]] = 128,
    ) -> None:
        super().__init__()
        assert len(datasets) > 0, "CompositeDataset needs at least one child dataset."
        assert sample_index_within_dataset in ("random", "mod")
        self.image_size = image_size

        self.names: List[str] = [n for n, _ in datasets]
        self.datasets: List[Dataset] = [d for _, d in datasets]
        self.return_dataset_name = return_dataset_name
        self.sample_index_within_dataset = sample_index_within_dataset
        self.base_seed = int(base_seed)
        self.epoch = 0

        if weights is None:
            w = np.ones(len(self.datasets), dtype=np.float64)
        else:
            w = np.asarray(list(weights), dtype=np.float64)
            assert w.shape == (len(self.datasets),)
            assert np.all(w >= 0), "weights must be non-negative."
            assert np.sum(w) > 0, "at least one weight must be > 0."
        self.weights = (w / np.sum(w)).astype(np.float64)

        # define an effective length (DataLoader uses this for epoch sizing)
        if length is not None:
            self._length = int(length)
        else:
            lens = [len(d) for d in self.datasets]
            self._length = int(sum(lens))

    def set_epoch(self, epoch: int) -> None:
        """Call this from the training loop each epoch if per-epoch reshuffling."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    def _make_rng(self, index: int) -> np.random.Generator:
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0

        # If in DDP, mix rank in as well (optional).
        # This keeps different ranks from sampling identically.
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        except Exception:
            rank = 0

        # Mix everything into a 32-bit seed
        seed = (
            (hash((self.base_seed, self.epoch, index, worker_id, rank)) % (2**32))
        )
        return np.random.default_rng(seed)

    def __getitem__(self, index: int):
        rng = self._make_rng(index)

        ds_id = int(rng.choice(len(self.datasets), p=self.weights))
        ds = self.datasets[ds_id]

        if self.sample_index_within_dataset == "random":
            sub_index = int(rng.integers(0, len(ds)))
        else:
            sub_index = int(index % len(ds))

        item = ds[sub_index]

        if self.return_dataset_name:
            # try to be minimally invasive:
            # - if item is (ret_dict, trgt_rgb) like thes, add a key into ret_dict
            if isinstance(item, tuple) and len(item) >= 1 and isinstance(item[0], dict):
                item0 = dict(item[0])
                item0["dataset_name"] = self.names[ds_id]
                return (item0,) + item[1:]
            return item, self.names[ds_id]

        return item
