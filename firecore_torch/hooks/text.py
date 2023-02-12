from .base import BaseHook
from firecore_torch.metrics import MetricCollection
import logging
from typing import List, Dict, TypedDict, Optional
from torch import Tensor

logger = logging.getLogger(__name__)


class FmtCfg(TypedDict):
    key: str
    fmt: str


class TextLoggerHook(BaseHook):

    def __init__(self,  fmt: List[FmtCfg], interval: int = 100, metric_keys: Optional[List[str]] = None) -> None:
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

    def after_epoch(self, metrics: MetricCollection, epoch: int, **kwargs):
        metric_outputs = metrics.compute()
        formatted_outputs = self._format_metrics(metric_outputs)
        logger.info('{}'.format(' '.join(formatted_outputs)))

    def after_iter(self, metrics: MetricCollection, batch_idx: int, **kwargs):
        metric_outputs = metrics.compute_by_keys(self._metric_keys)
        formatted_outputs = self._format_metrics(metric_outputs)
        logger.info('{}'.format(' '.join(formatted_outputs)))

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
