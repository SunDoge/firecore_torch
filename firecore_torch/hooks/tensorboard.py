from .base import BaseHook
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import logging
from torch import Tensor
from typing import Dict
from firecore_torch.helpers import rank_zero
from firecore_torch.metrics import MetricCollectionV2

logger = logging.getLogger(__name__)


class TbForMetricsHook(BaseHook):

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @rank_zero
    def after_epoch(self, summary_writer: SummaryWriter, metrics: MetricCollectionV2, stage: str, epoch: int, **kwargs):
        metric_outputs = metrics.compute()
        for key, value in metric_outputs.items():
            if value.ndim == 0:
                logger.info(
                    f'summary_writer.add_scaler(key={key}, epoch={epoch})'
                )
                summary_writer.add_scalar(
                    '{}/{}'.format(stage, key),
                    value,
                    global_step=epoch,
                )
