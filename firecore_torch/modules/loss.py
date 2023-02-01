from .base import BaseModel
from torch import nn, Tensor

nn.CrossEntropyLoss()


class Loss(BaseModel):

    def __init__(self, loss_fn: nn.Module, in_rules=None, out_rules=None) -> None:
        super().__init__(in_rules, out_rules)
        self.loss_fn = loss_fn

    def _forward(self, output: Tensor, target: Tensor):
        return {'loss': self.loss_fn(output, target)}
