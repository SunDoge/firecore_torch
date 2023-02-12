from .base import BaseHook
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging

import torch

torch.inference_mode


logger = logging.getLogger(__name__)


class InferenceHook(BaseHook):

    def __init__(self) -> None:
        super().__init__()
        self._inference_mode_raii_guard = None

    def before_epoch(self, model: nn.Module, **kwargs):
        logger.info('model.eval()')
        model.eval()
        self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    def after_epoch(self, **kwargs):
        del self._inference_mode_raii_guard
