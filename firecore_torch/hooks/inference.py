from .base import BaseHook
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging

import torch




logger = logging.getLogger(__name__)


class InferenceHook(BaseHook):

    def __init__(self) -> None:
        super().__init__()

    def before_epoch(self, model: nn.Module, **kwargs):
        torch.set_grad_enabled(False)
        logger.info('model.eval()')
        model.eval()
       

    def after_epoch(self, **kwargs):
        logger.info('exit inference mode')
        torch.set_grad_enabled(True)
