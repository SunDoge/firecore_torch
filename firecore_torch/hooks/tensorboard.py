from .base import BaseHook
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import logging
from torch import Tensor
from typing import Dict
from firecore_torch.helpers import rank_zero

logger = logging.getLogger(__name__)


class TensorboardHook(BaseHook):

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @rank_zero
    def after_epoch(self, summary_writer: SummaryWriter, metric_outputs: Dict[str, Tensor], stage: str, epoch: int, **kwargs):

        for key, value in metric_outputs.items():
            if value.ndim == 0:
                logger.info(
                    'summary_writer.add_scaler(key={}, epoch={})'.format(
                        key, epoch
                    )
                )
                summary_writer.add_scalar(
                    '{}/{}'.format(stage, key),
                    value,
                    global_step=epoch,
                )
