from torch import Tensor
import torch
from typing import Dict, Optional, Union, List, Sequence
from firecore import adapter
import logging
from torch import nn

logger = logging.getLogger(__name__)


def make_empty_future():
    fut = torch.futures.Future()
    fut.set_result(None)
    return fut


class BaseMetric(nn.Module):

    def __init__(
        self,
        in_rules: Optional[List[str]] = None,
        out_rules: Optional[List[str]] = None,
        fmt: str = '.4f',
    ) -> None:
        super().__init__()

        if in_rules is None:
            in_rules = []
        if out_rules is None:
            out_rules = []

        assert isinstance(in_rules, list)
        assert isinstance(out_rules, list)

        self._fmt = fmt
        self._in_rules = in_rules
        self._out_rules = out_rules
        self._cached_result: Optional[Dict[str, Tensor]] = None
        self._is_synced: bool = False

    def update(self, **kwargs):
        self._cached_result = None
        self._is_synced = False

        new_args = adapter.extract(kwargs, self._in_rules)
        self._update(*new_args)

    def compute(self) -> Dict[str, Tensor]:
        if self._cached_result is None:
            outputs = self._compute()
            self._cached_result = adapter.nameing(
                outputs,
                self._out_rules
            )

        return self._cached_result

    def sync(self):
        if self._is_synced:
            logger.info('Already synced, skip')
        else:
            self._is_synced = True
            fut = self._sync()
            if fut is not None:
                fut.wait()

    def reset(self):
        self._cached_result = None
        self._is_synced = False
        self._reset()

    def display(self) -> Dict[str, str]:
        tmpl = "{:" + self._fmt + "}"
        outputs = {}
        for key, value in self.compute().items():
            outputs[key] = tmpl.format(value.item())
        return outputs

    def _update(self, *args):
        """
        overwrite
        """
        pass

    def _compute(self) -> Union[Tensor, Sequence[Tensor]]:
        """
        overwrite
        """
        pass

    def _reset(self):
        pass

    def _sync(self) -> Optional[torch.futures.Future]:
        pass
