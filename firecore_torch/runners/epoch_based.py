from .base import BaseRunner, BaseRunner2
from torch import nn, Tensor
from typing import Dict, Iterable, List, Callable
import torch
from firecore_torch import helpers
from firecore_torch.metrics import MetricCollectionV2
import torch.distributed as dist
from icecream import ic
from .batch_processor import BatchProcessor
from firecore.meter import Meter

TensorDict = Dict[str, Tensor]


def default_forward_fn(
    model: nn.Module,
    criterion: nn.Module,
    **kwargs
):
    outputs = model(**kwargs)
    losses = criterion(**outputs, **kwargs)
    return outputs, losses


class EpochBasedRunner(BaseRunner):

    def __init__(
        self,
        base_model: nn.Module,
        model: nn.Module,
        criterion: nn.Module,
        data_source: Iterable[Dict[str, Tensor]],
        metrics: MetricCollectionV2,
        max_epochs: int,
        batch_cfg: dict,

        # Auto fill
        device: torch.device,
        hooks: list,
        forward_fn: Callable = default_forward_fn,
        **kwargs
    ) -> None:
        super().__init__(hooks, **kwargs)

        epoch_length = len(data_source)

        self.base_model = base_model
        self.model = model
        self.criterion = criterion
        self.data_source = data_source
        self.device = device
        self.metrics = metrics
        self.eta_meter = Meter(
            total=max_epochs * epoch_length
        )

        # TODO: better epoch_length
        self.epoch_length = epoch_length
        self.max_epochs = max_epochs

        self._forward_fn = forward_fn
        self._batch_processor = BatchProcessor(**batch_cfg)

        self.call_hook('on_init')

    def step(self, epoch: int):
        self.metrics.reset()
        self.call_hook('before_epoch', epoch=epoch)

        for batch_idx, batch in enumerate(self.data_source, start=1):
            self.call_hook(
                'before_iter',
                epoch=epoch,
                batch_idx=batch_idx,
            )

            batch_on_device, batch_size = self.call_method(
                self._batch_processor,
                batch=batch,
            )

            self.call_hook(
                'before_forward',
                epoch=epoch,
                batch_idx=batch_idx,
                batch_size=batch_size,
                **batch_on_device
            )

            outputs, losses = self.call_method(
                self._forward_fn,
                **batch_on_device
            )

            self.call_hook(
                'after_forward',
                epoch=epoch,
                batch_idx=batch_idx,
                batch_size=batch_size,
                **batch_on_device,
                **outputs,
                **losses
            )

            self.metrics.update(
                batch_size=batch_size,
                **losses,
                **outputs,
                **batch_on_device
            )

            self.call_hook(
                'after_iter',
                epoch=epoch,
                batch_idx=batch_idx,
                batch_size=batch_size,
                **batch_on_device,
                **outputs,
                **losses
            )

        if dist.is_available() and dist.is_initialized():
            self.metrics.sync().wait()

        self.call_hook(
            'after_epoch',
            epoch=epoch,
        )
