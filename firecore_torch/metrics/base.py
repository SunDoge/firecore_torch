from torch import Tensor
import torch
from typing import Dict, Optional, Union, List
from firecore.adapter import extract
import logging

logger = logging.getLogger(__name__)


def make_empty_future():
    fut = torch.futures.Future()
    fut.set_result(None)
    return fut


class BaseMetric:

    def __init__(
        self,
        in_rules: List[str],
        out_rules: List[str],
        fmt: str = '.4f',
    ) -> None:
        self._fmt = fmt
        self._in_rules = in_rules
        self._out_rules = out_rules
        self._cached_result: Optional[Dict[str, Tensor]] = None
        self._is_synced: bool = False

    def update(self, **kwargs):
        self._cached_result = None
        self._is_synced = False

        new_args = extract(kwargs, self._in_rules)
        self._update(*new_args)

    def compute(self) -> Dict[str, Tensor]:
        if self._cached_result is None:
            outputs = self._compute()
            assert len(outputs) == len(self._out_rules)
            
        return self._cached_result

    def sync(self) -> Optional[torch.futures.Future]:
        if self._is_synced:
            logger.info('Already synced, skip')
            return
        else:
            self._is_synced = True
            return self._sync()

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

    def _update(self, output: Tensor, target: Tensor):
        """
        overwrite
        """
        pass

    def _compute(self) -> Dict[str, Tensor]:
        """
        overwrite
        """
        pass

    def _reset(self):
        pass

    def _sync(self) -> Optional[torch.futures.Future]:
        pass
