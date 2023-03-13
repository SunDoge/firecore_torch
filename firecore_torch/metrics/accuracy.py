from .base import BaseMetric
from typing import Dict, List
import torch
from torch import Tensor
from . import functional as F
import torch.distributed as dist


class Accuracy(BaseMetric):

    def __init__(self, topk: List[int] = [1], in_rules: Dict[str, str] = {}, out_rules: Dict[str, str] = {}) -> None:
        super().__init__(in_rules, out_rules)

        self._topk = topk

        self._count = torch.tensor(0, dtype=torch.long)
        self._sum = torch.zeros(len(topk), dtype=torch.float)

    def _update(self, output: Tensor, target: Tensor, **kwargs):
        device = output.device
        batch_size = target.size(0)

        if self._count.device != device:
            self._count = self._count.to(device)
            self._sum = self._sum.to(device)

        corrects = F.topk_correct(output, target, topk=self._topk)

        self._sum.add_(torch.as_tensor(corrects, device=device))
        self._count.add_(batch_size)

    def _compute(self):
        result = {}

        acc = self._sum / self._count

        for i, k in enumerate(self._topk):
            result['acc{}'.format(k)] = acc[i]
        return result

    def sync(self) -> torch.futures.Future:
        return torch.futures.collect_all([
            dist.all_reduce(self._sum, op=dist.ReduceOp.SUM,
                            async_op=True).get_future(),
            dist.all_reduce(self._count, op=dist.ReduceOp.SUM,
                            async_op=True).get_future(),
        ])

    def _reset(self):
        self._sum.fill_(0.)
        self._count.fill_(0)
