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

    def __init__(
        self,
        metrics: List[BaseMetric]
    ) -> None:
        super().__init__()
        self._metrics: List[BaseMetric] = nn.ModuleList(metrics)

    @torch.no_grad()
    def update(self, **kwargs):
        """
        We have to skip adapter.extract
        """
        self._cached_result = None
        self._is_synced = False

        for metric in self._metrics:
            metric.update(**kwargs)

    @torch.no_grad()
    def compute(self) -> Dict[str, Tensor]:
        """
        We have to 
        """
        if self._cached_result is None:
            outputs = {}
            for metric in self._metrics:
                outputs.update(
                    metric.compute()
                )
            self._cached_result = outputs

        return self._cached_result

    @torch.no_grad()
    def _sync(self) -> torch.futures.Future:
        futs = []
        for metric in self._metrics:
            fut = metric._sync()
            if fut is not None:
                futs.append(fut)
        return torch.futures.collect_all(futs)

    @torch.no_grad()
    def _reset(self):
        for metric in self._metrics:
            metric.reset()

    @torch.no_grad()
    def display(self) -> Dict[str, str]:
        outputs = {}
        for metric in self._metrics:
            outputs.update(metric.display())
        return outputs
