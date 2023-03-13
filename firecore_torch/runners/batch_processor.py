from typing import Optional, List, Dict, Union
from firecore.adapter import adapt
from torch import Tensor
import torch


class BatchProcessor:

    def __init__(
        self,
        names: Optional[List[str]] = None,
        rules: Optional[Dict[str, str]] = None,
    ):
        self._names = names
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

        batch_on_device = {
            k: v.to(device, non_blocking=True)
            for k, v in batch.items()
        }
        return batch_on_device

    def name_batch(self, batch: List[Tensor]):
        assert isinstance(batch, list)
        assert len(self._names) == len(batch)
        return {k: v for k, v in zip(self._names, batch)}
