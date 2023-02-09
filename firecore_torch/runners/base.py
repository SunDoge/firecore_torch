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
            getattr(hook, method)(**self.__dict__, **kwargs, **self._kwargs)

    def register_hook(self, hook: BaseHook):
        self._hooks.append(hook)
