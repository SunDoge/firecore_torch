from .base import BaseMetric
from typing import Dict
import torch
from torch import Tensor
import torch.distributed as dist


class Average(BaseMetric):

    _sum: Tensor
    _count: Tensor
    _val: Tensor

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.register_buffer('_sum', torch.tensor(0., dtype=torch.float))
        self.register_buffer('_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_val', torch.tensor(0., dtype=torch.float))

    def _update(self, output: Tensor, n: int):
        # print(output)
        # device = output.device

        # if self._val.device != device:
        #     self._val = self._val.to(device)
        #     self._count = self._count.to(device)
        #     self._sum = self._sum.to(device)

        self._val.copy_(output)
        self._sum.add_(output, alpha=n)
        self._count.add_(n)

    def _compute(self):
        avg = self._sum / self._count
        return avg

    def _sync(self) -> torch.futures.Future:
        return torch.futures.collect_all([
            dist.all_reduce(self._count, op=dist.ReduceOp.SUM,
                            async_op=True).get_future(),
            dist.all_reduce(self._sum, op=dist.ReduceOp.SUM,
                            async_op=True).get_future()
        ])

    def _reset(self):
        self._val.fill_(0.)
        self._count.fill_(0)
        self._sum.fill_(0.)
