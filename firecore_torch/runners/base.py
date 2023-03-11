from firecore_torch.hooks.base import BaseHook
from typing import List
import logging

logger = logging.getLogger(__name__)


class BaseRunner:

    def __init__(self, hooks: List[BaseHook], **kwargs) -> None:
        self._kwargs = kwargs
        self._hooks = hooks

    def step(self, epoch: int, stage: str = ''):
        pass

    def call_hook(self, method: str, **kwargs):
        logger.debug('call hook: %s', method)
        for hook in self._hooks:
            self.call_method(getattr(hook, method), **kwargs)

    def call_method(self, func, **kwargs):
        return func(**self._kwargs, **kwargs)

    def register_hook(self, hook: BaseHook):
        self._hooks.append(hook)


class BaseRunner2:

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    def step(self, epoch: int):
        pass

    def call_method(self, func, **kwargs):
        return func(**self.__dict__, **self._kwargs, **kwargs)
