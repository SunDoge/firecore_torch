from dataclasses import dataclass
from typing import Optional


@dataclass
class Ctx:
    epoch: int
    max_epochs: int
    batch_idx: int
    epoch_length: Optional[int]

    @property
    def max_iterations(self) -> int:
        assert self.epoch_length is not None, "training must have epoch length"
        return self.epoch_length * self.max_epochs

    @property
    def iteration(self) -> int:
        assert self.epoch_length is not None, "training must have epoch length"
        return self.epoch * self.epoch_length + self.batch_idx
