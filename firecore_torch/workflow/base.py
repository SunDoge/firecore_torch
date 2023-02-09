from firecore_torch.hooks.base import BaseHook
from typing import List
import logging

logger = logging.getLogger(__name__)


class BaseWorkflow:

    def __init__(self, prefix: str, hooks: List[BaseHook], **kwargs) -> None:
        self._kwargs = kwargs
        self._hooks = hooks
        self.prefix = prefix

    def step(self, epoch: int):
        pass

    def call_hook(self, method: str, **kwargs):
        logger.debug('call hook: %s', method)
        for hook in self._hooks:
            getattr(hook, method)(**self.__dict__, **kwargs, **self._kwargs)

    def register_hook(self, hook: BaseHook):
        self._hooks.append(hook)
