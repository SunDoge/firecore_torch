from .base import BaseMetric
from typing import List, Dict
import torch
import logging
from torch import Tensor
from typing import Optional

logger = logging.getLogger(__name__)


class MetricCollection:

    def __init__(
        self,
        metrics: Dict[str, BaseMetric],
    ) -> None:
        self._metrics = metrics

    def update(self, **kwargs):
        for metric in self._metrics.values():
            metric.update_adapted(**kwargs)

    def compute(self):
        out = {}
        for metric in self._metrics.values():
            out.update(metric.compute_adapted())
        return out

    def sync(self) -> torch.futures.Future:
        logger.debug('Sync all metrics', metrics=self._metrics)
        return torch.futures.collect_all([m.sync() for m in self._metrics.values()])

    def reset(self):
        logger.debug('Reset all metrics', metrics=self._metrics)
        for name, metric in self._metrics.items():
            metric.reset()

    def compute_by_keys(self, keys: List[str]) -> Dict[str, Tensor]:
        res = {}
        for key in keys:
            res.update(self._metrics[key].compute_adapted())
        return res
