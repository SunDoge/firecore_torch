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
        partial_keys: Optional[List[str]] = None,
    ) -> None:

        # Compute all metrics by default
        if not partial_keys:
            partial_keys = list(metrics.keys())
            logger.info('no partial keys, use all metrics by default')

        self._metrics = metrics
        self._partial_keys = partial_keys

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
        for metric in self._metrics:
            metric.reset()

    def compute_by_keys(self, keys: List[str]) -> Dict[str, Tensor]:
        res = {}
        for key in keys:
            res.update(self._metrics[key].compute_adapted())
        return res

    def compute_partial(self) -> Dict[str, Tensor]:
        return self.compute_by_keys(self._partial_keys)
