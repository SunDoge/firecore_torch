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
            fmt: List[FmtCfg], interval: int = 100, metric_keys: Optional[List[str]] = None) -> None:
        """
        Args:
            metric_keys: select keys when itering, default: ['loss']
        """
        super().__init__()

        if metric_keys is None:
            metric_keys = ['loss']

        self._interval = interval
        self._fmt = fmt
        self._metric_keys = metric_keys

    def after_epoch(self, epoch: int, metric_outputs: Dict[str, Tensor], max_epochs: int, **kwargs):
        formatted_outputs = self._format_metrics(metric_outputs)
        prefix = f'{epoch}/{max_epochs}'
        logger.info('{} {}'.format(prefix, ' '.join(formatted_outputs)))

    def after_iter(self, metrics: MetricCollection, batch_idx: int, epoch_length: int, **kwargs):
        if batch_idx > 0 and batch_idx % self._interval != 0:
            return
        with torch.inference_mode():
            metric_outputs = metrics.compute_by_keys(self._metric_keys)
        formatted_outputs = self._format_metrics(metric_outputs)
        prefix = f'{batch_idx}/{epoch_length}'
        logger.info('{} {}'.format(prefix, ' '.join(formatted_outputs)))

    def _format_metrics(self, outputs: Dict[str, Tensor]) -> List[str]:
        res = []
        for fmt_cfg in self._fmt:
            key = fmt_cfg['key']

            if key not in outputs:
                continue

            fmt = fmt_cfg['fmt']
            template = '{key}: {val' + fmt + '}'
            tensor = outputs[key]
            val = tensor.tolist()
            fmt_str = template.format(key=key, val=val)
            res.append(fmt_str)
        return res
