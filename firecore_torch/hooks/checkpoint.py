from .base import BaseHook
from pathlib import Path
from torch import nn
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LrScheduler
from torch.cuda.amp.grad_scaler import GradScaler
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PeriodicalCheckpointHook(BaseHook):

    def __init__(
        self,
        interval: int,
    ) -> None:
        super().__init__()


class CheckpointHook(BaseHook):

    def __init__(self) -> None:
        super().__init__()

    def on_init(self, work_dir: Path, **kwargs):
        self._work_dir = work_dir

    def after_epoch(
        self,
        base_model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: Optional[LrScheduler] = None,
        grad_scaler: Optional[GradScaler] = None,
        **kwargs
    ):
        state_dict = {
            'base_model': base_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
            'grad_scaler': grad_scaler.state_dict() if grad_scaler else None,
        }
        filename = self._work_dir / 'checkpoint.pth.tar'
        logger.info('save checkpoint: %s', filename)
        torch.save(
            state_dict, str(filename)
        )
