from .base import BaseMetric
from typing import List
import torch
# from firecore.logging import get_logger
import logging

logger = logging.getLogger(__name__)


class MetricCollection:

    def __init__(self, metrics: List[BaseMetric]) -> None:
        self._metrics = metrics

    def update(self, **kwargs):
        for metric in self._metrics:
            metric.update_adapted(**kwargs)

    def compute(self):
        out = {}
        for metric in self._metrics:
            out.update(metric.compute_adapted())
        return out

    def sync(self) -> torch.futures.Future:
        logger.debug('Sync all metrics', metrics=self._metrics)
        return torch.futures.collect_all([m.sync() for m in self._metrics])

    def reset(self):
        logger.debug('Reset all metrics', metrics=self._metrics)
        for metric in self._metrics:
            metric.reset()

    
