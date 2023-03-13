from .base import BaseMetric
from typing import Dict
import torch
from torch import Tensor
import torch.distributed as dist


class Average(BaseMetric):

    def __init__(self, in_rules: Dict[str, str] = {}, out_rules: Dict[str, str] = {}) -> None:
        super().__init__(in_rules, out_rules)
        self._sum = torch.tensor(0., dtype=torch.float)
        self._count = torch.tensor(0, dtype=torch.long)
        self._val = torch.tensor(0., dtype=torch.float)

    def _update(self, output: Tensor, target: Tensor, **kwargs):
        # print(output)
        batch_size = target.size(0)
        device = output.device

        if self._val.device != device:
            self._val = self._val.to(device)
            self._count = self._count.to(device)
            self._sum = self._sum.to(device)

        self._val.copy_(output)
        self._sum.add_(output)
        self._count.add_(batch_size)

    def _compute(self):
        avg = self._sum / self._count
        return {'avg': avg, 'val': self._val}

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
