from .base import BaseMetric
from typing import Dict
from torch import Tensor
import time


class Speed(BaseMetric):

    def __init__(self, fmt: str = '.2f', in_rules: Dict[str, str] = {}, out_rules: Dict[str, str] = {}) -> None:
        super().__init__(fmt, in_rules, out_rules)

        self._num_samples = 0
        self._start_time = time.perf_counter()

    def _update(self, batch_size: int, **kwargs):
        self._num_samples += batch_size

    def _compute(self) -> Dict[str, Tensor]:
        duration = time.perf_counter()
        speed = self._num_samples / duration
        return {'spl_per_sec': speed}

    def _reset(self):
        self._num_samples = 0
        self._start_time = time.perf_counter()
