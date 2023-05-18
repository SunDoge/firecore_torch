from .base import BaseHook
from torch import nn, Tensor
from deepspeed.runtime.engine import DeepSpeedEngine
from firecore.meter import Meter


class DeepspeedTraining(BaseHook):

    def __init__(self) -> None:
        super().__init__()

    def before_epoch(
        self,
        model: nn.Module,
        data_source,
        epoch: int,
        **kwargs
    ):

        model.train()

    def after_forward(self, loss: Tensor, model: DeepSpeedEngine,  **kwargs):
        model.backward(loss)
        model.step()

    def after_iter(self, eta_meter: Meter, **kwargs):
        eta_meter.step()


class DeepspeedInference(BaseHook):

    def __init__(self) -> None:
        super().__init__()


    def before_epoch(self, **kwargs):
        return super().before_epoch(**kwargs)
