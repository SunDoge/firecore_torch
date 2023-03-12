from .base import BaseRunner, BaseRunner2
from torch import nn, Tensor
from typing import Dict, Iterable, List, Callable
import torch
from firecore_torch import helpers
from firecore_torch.metrics import MetricCollection
import torch.distributed as dist
from icecream import ic

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
        metrics: MetricCollection,
        max_epochs: int,

        # Auto fill
        device: torch.device,
        hooks: list,
        forward_fn: Callable = default_forward_fn,
        **kwargs
    ) -> None:
        super().__init__(hooks, **kwargs)

        self.base_model = base_model
        self.model = model
        self.criterion = criterion
        self.data_source = data_source
        self.device = device
        self.metrics = metrics

        # TODO: better epoch_length
        self.epoch_length = len(data_source)
        self.max_epochs = max_epochs

        self._forward_fn = forward_fn

        self.call_hook('on_init')

    def step(self, epoch: int, stage: str = ''):
        self.metrics.reset()
        self.call_hook('before_epoch', epoch=epoch, stage=stage)

        for batch_idx, batch in enumerate(self.data_source, start=1):
            self.call_hook(
                'before_iter',
                epoch=epoch,
                batch_idx=batch_idx,
                stage=stage,
                **batch
            )
            # TODO: maybe a filter
            batch_on_device = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch.items()
            }

            self.call_hook(
                'before_forward',
                epoch=epoch,
                batch_idx=batch_idx,
                stage=stage,
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
                stage=stage,
                **batch_on_device,
                **outputs,
                **losses
            )

            with torch.inference_mode():
                self.metrics.update(
                    **losses, **outputs, **batch_on_device
                )

            self.call_hook(
                'after_iter',
                epoch=epoch,
                batch_idx=batch_idx,
                stage=stage,
                **batch_on_device,
                **outputs,
                **losses
            )

            if batch_idx == 1000:
                break

        if dist.is_available() and dist.is_initialized():
            self.metrics.sync().wait()

        with torch.inference_mode():
            metric_outputs = self.metrics.compute()
        self.call_hook(
            'after_epoch',
            epoch=epoch,
            stage=stage,
            metric_outputs=metric_outputs
        )
