from .base import BaseWorkflow
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LrScheduler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from firecore_torch.metrics import MetricCollection
import logging
from firecore_torch import helpers
import torch
from typing import Dict

torch.inference_mode
torch.no_grad

logger = logging.getLogger(__name__)


class Trainer(BaseWorkflow):
    """
    Not very generic
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LrScheduler,
        data: DataLoader,
        metrics: MetricCollection,
        device: torch.device,
        log_interval: int,
        **kwargs
    ) -> None:
        super().__init__()
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._data = data
        self._metrics = metrics
        self._device = device
        self._log_interval = log_interval

    def step(self, epoch: int):
        self._data.sampler.set_epoch(epoch)
        self._model.train()
        self._metrics.reset()

        for batch_idx, batch in enumerate(self._data):
            batch = helpers.copy_to_device(batch, self._device)
            outputs: Dict[str, Tensor] = self._model(**batch)
            losses: Dict[str, Tensor] = self._criterion(**outputs, **batch)
            losses['loss'].backward()
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)

            # Metrics
            losses = helpers.detach_all(losses)
            self._metrics.update(**losses, **outputs, **batch)

            if batch_idx % self._log_interval == 0:
                metrics = self._metrics.compute()
                metrics = helpers.copy_to_py(metrics)
                # logger.info('show metrics', batch_idx=batch_idx, **metrics)

        self._metrics.sync().wait()
        metrics = self._metrics.compute()
        metrics = helpers.copy_to_py(metrics)
        # logger.info('show metrics', epoch=epoch, **metrics)
