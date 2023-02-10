from torch.utils.data import Dataset
import torch

from typing import List, TypedDict, Dict, Callable
from torch import Tensor
import copy

class FakeDatset(Dataset):

    def __init__(
        self,
        length: int = 100,
        **kwargs: Callable[[], Tensor],
    ) -> None:
        super().__init__()

        samples = []
        for _ in range(length):
            samples.append({k: v() for k, v in kwargs.items()})

        self._length = length
        self._samples = samples


    def __len__(self):
        return self._length
    
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return self._samples[index]



        