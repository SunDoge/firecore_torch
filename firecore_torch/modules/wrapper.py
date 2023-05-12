from torch import nn, Tensor
from typing import List
from firecore import adapter


class Model(nn.Module):

    def __init__(self, module: nn.Module, in_rules: List[str], out_rules: List[str]) -> None:
        super().__init__()
        self._in_rules = in_rules
        self._out_rules = out_rules
        self.module = module

    def forward(self, **kwargs):
        new_args = adapter.extract(kwargs, self._in_rules)
        outputs = self.module(*new_args)
        new_outputs = adapter.nameing(outputs, self._out_rules)
        return new_outputs
