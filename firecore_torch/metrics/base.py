from torch import Tensor
import torch
from typing import Dict, Optional
from firecore.adapter import adapt
import logging

logger = logging.getLogger(__name__)


def make_empty_future():
    fut = torch.futures.Future()
    fut.set_result(None)
    return fut


class BaseMetric:

    def __init__(self, in_rules: Dict[str, str] = {}, out_rules: Dict[str, str] = {}) -> None:
        self._in_rules = in_rules
        self._out_rules = out_rules
        self._cached_result: Optional[Dict[str, Tensor]] = None
        self._is_synced: bool = False

    def update(self, *args, **kwargs):
        self._cached_result = None
        self._is_synced = False
        self._update(*args, **kwargs)

    def update_adapted(self, **kwargs):
        new_kwargs = adapt(kwargs, self._in_rules)
        self.update(**new_kwargs)

    def compute(self) -> Dict[str, Tensor]:
        if self._cached_result is None:
            self._cached_result = self._compute()
        return self._cached_result

    def compute_adapted(self):
        out = self.compute()
        assert isinstance(out, dict)
        new_out = adapt(out, self._out_rules)
        return new_out

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
