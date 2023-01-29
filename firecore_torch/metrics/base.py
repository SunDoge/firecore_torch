from torch import Tensor
import torch
from typing import Dict

torch.nn.CrossEntropyLoss


class BaseMetric:

    def __init__(self, in_rules: Dict[str, str] = {}, out_rules: Dict[str, str] = {}) -> None:
        pass

    def update(self, preds: Tensor, target: Tensor):
        pass

    def update_adapted(self, **kwargs):
        pass

    def compute(self):
        pass
