import torch
from typing import Optional, List, Dict, Union
from torch import Tensor
from firecore import adapter


class BatchSizeExtractor:

    def __init__(
        self,
        batch_size_key: Optional[str] = None,
        batch_size_index: int = 0,
    ) -> None:
        self._batch_size_key = batch_size_key
        self._batch_size_index = batch_size_index

    def __call__(self, batch: Dict[str, Tensor]):
        assert isinstance(batch, dict), 'input must be dict'
        if self._batch_size_key is not None:
            tensor = batch[self._batch_size_key]
        else:
            tensor = next(iter(batch.values()))

        batch_size = tensor.size(self._batch_size_index)
        return batch_size


class BatchProcessor:

    def __init__(
        self,
        device: torch.device,
        names: Optional[List[str]] = None,
        rules: Optional[List[str]] = None,
        batch_size_key: Optional[str] = None,
        batch_size_index: int = 0,
    ) -> None:
        """
        Args:
            device,
            names: for Tuple input
            rules: for Dict input
        """
        self._device = device
        self._names = names
        self._rules = rules
        self._batch_size_extractor = BatchSizeExtractor(
            batch_size_key=batch_size_key,
            batch_size_index=batch_size_index,
        )

    def __call__(self, batch: Union[List[Tensor], Dict[str, Tensor]]):
        if self._names is not None:
            batch = adapter.nameing(batch, self._names)

        if self._rules is not None:
            batch = adapter.extract(batch, self._rules)

        batch_size = self._batch_size_extractor(batch)

        return batch, batch_size

    def get_batch_size(self, batch: Dict[str, Tensor]):
        if self._batch_size_key:
            tensor = batch[self._batch_size_key]
        else:
            tensor = next(iter(batch.values()))
        batch_size = tensor.size(self._batch_size_index)
        return batch_size
