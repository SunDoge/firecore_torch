from typing import List, Optional
from .base import BaseMetric
from torch import Tensor
import torch
import torch.distributed as dist


class SampleCounter(BaseMetric):

    _num_samples: Tensor

    def __init__(self, in_rules: Optional[List[str]] = None, out_rules: Optional[List[str]] = None, fmt: str = 'd') -> None:
        if in_rules == None:
            in_rules = ['batch_size']
        if out_rules == None:
            out_rules = ['num_samples']
        super().__init__(in_rules, out_rules, fmt)

        self.register_buffer(
            '_num_samples', torch.tensor(0, dtype=torch.int64)
        )

    def _update(self, batch_size: int):
        self._num_samples.add_(batch_size)

    def _compute(self):
        return self._num_samples

    def _sync(self) -> Optional[torch.futures.Future]:
        fut = dist.all_reduce(
            self._num_samples, op=dist.ReduceOp.SUM,
            async_op=True
        ).get_future()
        return fut

    def _reset(self):
        self._num_samples.fill_(0)
