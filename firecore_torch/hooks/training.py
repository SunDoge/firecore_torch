from .base import BaseHook
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from typing import Optional
from torch.optim.lr_scheduler import _LRScheduler as LrScheduler

logger = logging.getLogger(__name__)


class TrainingHook(BaseHook):

    def __init__(
        self,
        update_lr: str = 'epoch',
    ) -> None:
        super().__init__()
        assert update_lr in ['epoch', 'iter', 'never'], update_lr
        self._update_lr = update_lr

    def before_epoch(self, model: nn.Module, data, epoch: int, **kwargs):
        if isinstance(data, DataLoader):
            if isinstance(data.sampler, DistributedSampler):
                logger.info('data.sampler.set_epoch: {}'.format(epoch))
                data.sampler.set_epoch(epoch)

        logger.info('model.train()')
        model.train()

    def after_forward(self, loss: Tensor, optimizer: Optimizer, **kwargs):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def after_iter(self, lr_scheduler: LrScheduler = None, **kwargs):
        if self._update_lr == 'iter':
            lr_scheduler.step()

    def after_epoch(self, lr_scheduler: LrScheduler = None, **kwargs):
        if self._update_lr == 'epoch':
            lr_scheduler.step()
