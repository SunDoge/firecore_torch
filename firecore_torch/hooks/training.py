from .base import BaseHook
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging


logger = logging.getLogger(__name__)


class TrainingHook(BaseHook):

    def before_epoch(self, model: nn.Module, data, epoch: int, **kwargs):
        if isinstance(data, DataLoader):
            if isinstance(data.sampler, DistributedSampler):
                logger.info('data.sampler.set_epoch: {}'.format(epoch))
                data.sampler.set_epoch(epoch)

        logger.info('model.train()')
        model.train()

    def before_iter(self, **kwargs):
        return super().before_iter(**kwargs)

    def after_iter(self, loss: Tensor, optimizer: Optimizer, **kwargs):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def after_epoch(self, **kwargs):
        return super().after_epoch(**kwargs)
