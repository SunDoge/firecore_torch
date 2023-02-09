from .base import BaseHook
from firecore_torch.metrics import MetricCollection
import logging
from typing import List, Dict, TypedDict

logger = logging.getLogger(__name__)


class FmtCfg(TypedDict):
    key: str
    fmt: str


class TextLoggerHook(BaseHook):

    def __init__(self, fmt: List[FmtCfg], interval: int = 100) -> None:
        super().__init__()
        self._interval = interval
        self._fmt = fmt

    def before_epoch(self, **kwargs):
        return super().before_epoch(**kwargs)

    def before_iter(self, **kwargs):
        return super().before_iter(**kwargs)

    def after_iter(self, metrics: MetricCollection, **kwargs):
        pass

    def after_epoch(self, **kwargs):
        return super().after_epoch(**kwargs)

    def after_metrics(self, metrics: MetricCollection, **kwargs):
        metric_outputs = metrics.compute()
        formatted_outputs = []
        for fmt_cfg in self._fmt:
            metric_value = metric_outputs[fmt_cfg['key']]
            fmt_str = '{}: {' + fmt_cfg['fmt'] + '}'
            out = fmt_str.format(fmt_cfg['key'], metric_value.item())
            formatted_outputs.append(out)

        logger.info('{}'.format(' '.join(formatted_outputs)))
