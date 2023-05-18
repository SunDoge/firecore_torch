from .base import BaseMetric
from typing import List, Dict
import torch
import logging
from torch import Tensor
from typing import Optional, Union
import torch
from .base import BaseMetric
from torch import nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


class MetricCollection(BaseMetric):

    _num_samples: Tensor

    def __init__(
        self,
        metrics: List[BaseMetric]
    ) -> None:
        super().__init__()

        self._metrics: List[BaseMetric] = nn.ModuleList(metrics)

        self.register_buffer('_num_samples', torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, **kwargs):
        batch_size = kwargs.get('batch_size')
        assert batch_size, f'batch_size should not be none or 0, {batch_size}'

        self._num_samples.add_(batch_size)

        for metric in self._metrics:
            metric.update(**kwargs)

    @torch.no_grad()
    def _compute(self) -> Dict[str, Tensor]:
        outputs = {}
        for metric in self._metrics:
            outputs.update(
                metric.compute()
            )
        outputs['num_samples'] = self._num_samples
        return outputs

    @torch.no_grad()
    def _sync(self) -> torch.futures.Future:
        fut = dist.all_reduce(
            self._num_samples, op=dist.ReduceOp.SUM,
            async_op=True).get_future()

        futs = [fut]

        for metric in self._metrics:
            fut = metric._sync()
            futs.append(fut)
        return torch.futures.collect_all(futs)

    def reset(self):
        for metric in self._metrics:
            metric.reset()

    def display(self) -> Dict[str, str]:
        outputs = {}
        for metric in self._metrics:
            outputs.update(metric.display())
        outputs.update(super().display())
        return outputs
