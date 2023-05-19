from dataclasses import dataclass
from typing import Optional


@dataclass
class Ctx:
    epoch: int
    max_epochs: int
    batch_idx: int
    epoch_length: Optional[int]
    stage: str

    @property
    def max_iterations(self) -> int:
        assert self.epoch_length is not None, "training must have epoch length"
        return self.epoch_length * self.max_epochs

    @property
    def iteration(self) -> int:
        assert self.epoch_length is not None, "training must have epoch length"
        return self.epoch * self.epoch_length + self.batch_idx



"""
optimize 1000 * 1000 iters
eval every 1000 iters
base batch_size 256

optimize 100 epochs
eval every 1 or 2 epochs
base batch_size 256
-> epoch_length (training)
-> iterations (training) 

epoch_length = 1000 for each epoch with base bs 256
    1000 * 100 * 256 / bs
"""
class TrainingContext:
    epoch: int

