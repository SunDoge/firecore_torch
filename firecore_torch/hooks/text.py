from .base import BaseHook
from firecore_torch.metrics import MetricCollection
import logging
from typing import List, Dict, TypedDict, Optional
from torch import Tensor
import torch

logger = logging.getLogger(__name__)


class FmtCfg(TypedDict):
    key: str
    fmt: str


class TextLoggerHook(BaseHook):

    # TODO: change metric_keys to other name
    def __init__(
        self,
        interval: int = 100,
        metric_keys: Optional[List[str]] = None
    ) -> None:
        """
        Args:
            metric_keys: select keys when itering, default: ['loss']
        """
        super().__init__()

        if metric_keys is None:
            metric_keys = ['loss']

        self._interval = interval
        self._metric_keys = metric_keys

    def after_epoch(self, epoch: int, metrics: MetricCollection, max_epochs: int, stage: str, **kwargs):

        metric_outputs = metrics.compute(fmt=True)
        formatted_outputs = self._format_metrics(metric_outputs)
        prefix = f'{stage} {epoch}/{max_epochs}'
        logger.info('{} {}'.format(prefix, ' '.join(formatted_outputs)))

    def after_iter(self, metrics: MetricCollection, batch_idx: int, epoch_length: int, stage: str, **kwargs):
        if batch_idx > 0 and batch_idx % self._interval != 0:
            return

        metric_outputs = metrics.compute_by_keys(self._metric_keys, fmt=True)
        formatted_outputs = self._format_metrics(metric_outputs)
        prefix = f'{stage} {batch_idx}/{epoch_length}'
        logger.info('{} {}'.format(prefix, ' '.join(formatted_outputs)))

    def _format_metrics(self, outputs: Dict[str, str]) -> List[str]:
        res = []
        for key, value in outputs.items():
            res.append('{}: {}'.format(key, value))
        return res
