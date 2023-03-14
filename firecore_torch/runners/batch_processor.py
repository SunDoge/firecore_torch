from typing import Optional, List, Dict, Union
from firecore.adapter import adapt
from torch import Tensor
import torch


class BatchProcessor:

    def __init__(
        self,
        names: Optional[List[str]] = None,
        batch_size_key: Optional[str] = None,
        batch_size_index: int = 0,
        rules: Optional[Dict[str, str]] = None,
    ):
        self._names = names
        self._batch_size_key = batch_size_key
        self._batch_size_index = batch_size_index
        self._rules = rules

    def __call__(
        self,
        batch: Union[List[Tensor], Dict[str, Tensor]],
        device: torch.device,
        **kwargs
    ):
        if self._names:
            batch = self.name_batch(batch)

        if self._rules:
            batch = adapt(batch, self._rules)

        if self._batch_size_key:
            tensor = batch[self._batch_size_key]
        else:
            tensor = next(iter(batch.values()))

        batch_size = tensor.size(self._batch_size_index)

        batch_on_device = {
            k: v.to(device, non_blocking=True)
            for k, v in batch.items()
        }
        return batch_on_device, batch_size

    def name_batch(self, batch: List[Tensor]):
        assert isinstance(batch, list)
        assert len(self._names) == len(batch)
        return {k: v for k, v in zip(self._names, batch)}
