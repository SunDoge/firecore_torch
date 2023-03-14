from .base import BaseHook
from firecore_torch.metrics import MetricCollection
import logging
from typing import List, Dict, TypedDict, Optional
from torch import Tensor
import torch
meter
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
        self._rate_meter = Meter()

    def before_epoch(self, **kwargs):
        self._rate_meter.reset()

    def after_epoch(self, epoch: int, metrics: MetricCollection, max_epochs: int, stage: str, **kwargs):

        metric_outputs = metrics.compute(fmt=True)
        formatted_outputs = self._format_metrics(metric_outputs)
        rate = self._rate_meter.rate
        prefix = f'{stage} {epoch}/{max_epochs} {rate:.1f} spl/s'
        logger.info('{} {}'.format(prefix, ' '.join(formatted_outputs)))

    def after_iter(
        self,
        metrics: MetricCollection,
        batch_idx: int,
        epoch_length: int,
        stage: str,
        batch_size: int,
        eta_meter: Meter,
        **kwargs
    ):
        self._rate_meter.step(n=batch_size)

        if batch_idx > 0 and batch_idx % self._interval != 0:
            return

        metric_outputs = metrics.compute_by_keys(self._metric_keys, fmt=True)
        formatted_outputs = self._format_metrics(metric_outputs)
        rate = self._rate_meter.rate
        prefix = f'{stage} {batch_idx}/{epoch_length} {rate:.1f} spl/s'

        if eta_meter.is_updated:
            prefix += ' etr: {}'.format(eta_meter.remaining_timedelta)

        logger.info('{} {}'.format(prefix, ' '.join(formatted_outputs)))

    def _format_metrics(self, outputs: Dict[str, str]) -> List[str]:
        res = []
        for key, value in outputs.items():
            res.append('{}: {}'.format(key, value))
        return res
