from .base import BaseRunner
from typing import List, Optional, Dict, Union
from firecore_torch.hooks.base import BaseHook
from firecore_torch.batch_processor import BatchProcessor
from icecream import ic
from firecore_torch.metrics.collection import MetricCollection
from torch import Tensor, nn
from firecore.meter import Meter


class EpochBasedRunner(BaseRunner):
    @staticmethod
    def default_forward_fn(
        model: nn.Module,
        criterion: nn.Module,
        **kwargs
    ):
        outputs = model(**kwargs)
        losses = criterion(**outputs, **kwargs)
        return outputs, losses

    def __init__(self, hooks: List[BaseHook], **kwargs) -> None:
        super().__init__(hooks, **kwargs)
        ic([(k, type(v)) for k, v in kwargs.items()])

    def step(self, epoch: int, epoch_length: Optional[int]):
        self.call_method(
            self.one_epoch,
            epoch=epoch,
            epoch_length=epoch_length
        )

    def one_epoch(
        self,
        epoch: int,
        epoch_length: Optional[int],
        # From self
        data_source,
        metrics: MetricCollection,
        **kwargs
    ):
        self.call_hook(
            "before_epoch",
            epoch=epoch,
            epoch_length=epoch_length,
        )

        metrics.reset()

        if epoch_length is not None:
            data_source_iter = iter(data_source)
            for batch_idx in range(epoch_length):
                batch = next(data_source_iter)
                self.call_method(
                    self.one_iteration,
                    epoch=epoch,
                    epoch_length=epoch_length,
                    batch=batch,
                    batch_idx=batch_idx,
                )
        else:
            for batch_idx, batch in enumerate(data_source):
                self.call_method(
                    self.one_iteration,
                    epoch=epoch,
                    epoch_length=epoch_length,
                    batch=batch,
                    batch_idx=batch_idx,
                )

        self.call_hook(
            "after_epoch",
            epoch=epoch,
            epoch_length=epoch_length,
        )

    def one_iteration(
        self,
        epoch: int,
        batch_idx: int,
        epoch_length: Optional[int],
        batch: Dict[str, Tensor],
        # From self
        batch_processor: BatchProcessor,
        forward_fn,
        metrics: MetricCollection,
        **kwargs
    ):
        self.call_hook(
            "before_iter",
            epoch=epoch,
            batch_idx=batch_idx,
            epoch_length=epoch_length,
        )

        batch_on_device, batch_size = batch_processor(batch)

        self.call_hook(
            "before_forward",
            epoch=epoch,
            batch_idx=batch_idx,
            batch_size=batch_size,
            epoch_length=epoch_length,
            **batch_on_device,
        )

        outputs, losses = self.call_method(forward_fn, **batch_on_device)

        self.call_hook(
            "after_forward",
            epoch=epoch,
            batch_idx=batch_idx,
            batch_size=batch_size,
            epoch_length=epoch_length,
            **batch_on_device,
            **outputs,
            **losses,
        )

        metrics.update(
            **batch_on_device,
            **outputs,
            **losses,
            batch_size=batch_size,
            epoch_length=epoch_length,
        )

        self.call_hook(
            "after_iter",
            epoch=epoch,
            batch_idx=batch_idx,
            batch_size=batch_size,
            epoch_length=epoch_length,
            **batch_on_device,
            **outputs,
            **losses,
        )
