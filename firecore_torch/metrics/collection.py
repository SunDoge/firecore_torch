from .base import BaseMetric
from typing import List
import torch


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
        return torch.futures.collect_all([m.sync() for m in self._metrics])

    def reset(self):
        for metric in self._metrics:
            metric.reset()
