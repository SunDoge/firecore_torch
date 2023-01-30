from torch import Tensor
import torch
from typing import Dict
from firecore.adapter import adapt

torch.nn.CrossEntropyLoss


class BaseMetric:

    def __init__(self, in_rules: Dict[str, str] = {}, out_rules: Dict[str, str] = {}) -> None:
        self._in_rules = in_rules
        self._out_rules = out_rules

    def update(self, output: Tensor, target: Tensor):
        pass

    def update_adapted(self, **kwargs):
        new_kwargs = adapt(kwargs, self._in_rules)
        self.update(**new_kwargs)

    def compute(self):
        pass

    def compute_adapted(self):
        out = self.compute()
        assert isinstance(out, dict)
        new_out = adapt(out, self._out_rules)
        return new_out

    def sync(self) -> torch.Future:
        pass
