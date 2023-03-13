from .base import BaseMetric
from typing import Dict
from torch import Tensor
import time


# class Speed(BaseMetric):

#     def __init__(self, in_rules: Dict[str, str] = {}, out_rules: Dict[str, str] = {}) -> None:
#         super().__init__(in_rules, out_rules)

#         self._num_samples = 0
#         self._start_time = time.perf_counter()

#     def _update(self, target: Tensor, **kwargs):
#         batch_size = target.size(0)

#         self._num_samples += batch_size
