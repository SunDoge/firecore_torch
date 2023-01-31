from torch import nn, Tensor
from typing import Dict
from firecore.adapter import adapt


class BaseModel(nn.Module):

    def __init__(self, in_rules=None, out_rules=None) -> None:
        super().__init__()
        self._in_rules = in_rules if in_rules else {}
        self._out_rules = out_rules if out_rules else {}

    def forward(self, *args, **kwargs):
        new_kwargs = adapt(kwargs, self._in_rules)
        out = self._forward(**new_kwargs)
        assert isinstance(out, dict)
        new_out = adapt(out, self._out_rules)
        return new_out

    def _forward(self, **kwargs):
        pass
