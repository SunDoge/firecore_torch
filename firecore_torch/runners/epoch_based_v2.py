from .base import BaseRunner2
from torch import nn, Tensor
from typing import Iterable, Dict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

logger = logging.getLogger(__name__)


class TrainRunner(BaseRunner2):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def step(self, epoch: int):

        self.call_method(
            self.before_epoch,
            epoch=epoch
        )

        self.call_method(
            self.start_loop,
            epoch=epoch
        )

        self.call_method(
            self.after_epoch,
            epoch=epoch,
        )

    def _step(
        self,
        **kwargs,
    ):
        pass

    def start_loop(
        self,
        device: torch.device,
        epoch: int,
        data_source: Iterable[Dict[str, Tensor]],
        **kwargs
    ):
        for batch_idx, batch in enumerate(
            data_source, start=1
        ):
            self.call_method(
                self.before_iter,
                epoch=epoch,
                batch_idx=batch_idx,
                **batch,
            )

            batch_on_device = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
            }

            self.call_method(
                self.before_forward,
                epoch=epoch,
                batch_idx=batch_idx,
                **batch_on_device,
            )

            self.call_method(
                self.forward,
                epoch=epoch,
                batch_idx=batch_idx,
                **batch_on_device,
            )

            self.call_method(

            )

    def before_epoch(
        self,
        model: nn.Module,
        data_source: Iterable[Dict[str, Tensor]],
        epoch: int,
        **kwargs,
    ):
        model.train()
        logger.info('model.training=%s', model.training)

        if isinstance(data_source, DataLoader):
            if isinstance(data_source.sampler, DistributedSampler):
                logger.info('data.sampler.set_epoch: {}'.format(epoch))
                data_source.sampler.set_epoch(epoch)

    def after_epoch(
        self,
        **kwargs,
    ):
        pass

    def before_iter(
        self,
        **kwargs,
    ):
        pass

    def after_iter(
        self,
        **kwargs
    ):
        pass

    def before_forward(
        self,
        **kwargs,
    ):
        pass

    def after_forward(
        self,
        **kwargs,
    ):
        pass

    def forward(
        self,
        model: nn.Module,
        criterion: nn.Module,
        **kwargs,
    ):
        outputs = model(**kwargs)
        losses = criterion(**outputs, **kwargs)
        return outputs, losses
