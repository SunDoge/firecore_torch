from .base import BaseMetric
from typing import List, Dict
import torch
import logging
from torch import Tensor
from typing import Optional, Union
import torch

logger = logging.getLogger(__name__)


class MetricCollection:

    def __init__(
        self,
        metrics: Dict[str, BaseMetric],
    ) -> None:
        self._metrics = metrics

    @torch.inference_mode()
    def update(self, **kwargs):
        for metric in self._metrics.values():
            metric.update_adapted(**kwargs)

    @torch.inference_mode()
    def compute(self, fmt: bool = False) -> Union[Dict[str, Tensor], Dict[str, str]]:
        out = {}
        for metric in self._metrics.values():
            out.update(metric.compute_adapted(fmt=fmt))
        return out

    def sync(self) -> torch.futures.Future:
        logger.debug('Sync all metrics', metrics=self._metrics)
        futs = []
        for name, metric in self._metrics.items():
            fut = metric.sync()
            if fut is not None:
                futs.append(fut)
        return torch.futures.collect_all(futs)

    # Fix RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed
    @torch.inference_mode()
    def reset(self):
        logger.debug('Reset all metrics', metrics=self._metrics)
        for name, metric in self._metrics.items():
            metric.reset()

    def compute_by_keys(self, keys: List[str], fmt: bool = False) -> Union[Dict[str, Tensor], Dict[str, str]]:
        res = {}
        for key in keys:
            res.update(self._metrics[key].compute_adapted(fmt=fmt))
        return res
