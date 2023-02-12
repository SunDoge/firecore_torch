from .base import BaseRunner
from torch import nn, Tensor
from typing import Dict, Iterable, List
import torch
from firecore_torch import helpers
from firecore_torch.metrics import MetricCollection
import torch.distributed as dist
from icecream import ic

TensorDict = Dict[str, Tensor]


class EpochBasedRunner(BaseRunner):

    def __init__(
        self,
        base_model: nn.Module,
        model: nn.Module,
        criterion: nn.Module,
        data: Iterable[Dict[str, Tensor]],
        metrics: MetricCollection,
        # Auto fill
        device: torch.device,
        hooks: list,
        **kwargs
    ) -> None:
        super().__init__(hooks, **kwargs)

        self.base_model = base_model
        self.model = model
        self.criterion = criterion
        self.data = data
        self.device = device
        self.metrics = metrics
        self.max_iters = len(data)

    def step(self, epoch: int, stage: str = ''):
        self.call_hook('before_epoch', epoch=epoch, stage=stage)
        for batch_idx, batch in enumerate(self.data):
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
            outputs: TensorDict = self.model(**batch_on_device)
            losses: TensorDict = self.criterion(**outputs, **batch_on_device)
            self.call_hook(
                'after_forward',
                epoch=epoch,
                batch_idx=batch_idx,
                stage=stage,
                **batch_on_device,
                **outputs,
                **losses
            )

            # import ipdb; ipdb.set_trace()
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

        if dist.is_available() and dist.is_initialized():
            self.metrics.sync().wait()

        metric_outputs = self.metrics.compute()
        self.call_hook(
            'after_epoch',
            epoch=epoch,
            stage=stage,
            metric_outputs=metric_outputs
        )

    def run_iter(self, batch: Dict[str, Tensor], epoch: int, batch_idx: int):
        pass
        # return outputs, losses
