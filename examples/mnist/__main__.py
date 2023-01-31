from firecore_torch.metrics import Accuracy, Average
import firecore
from firecore.logging import get_logger
import typed_args as ta
from dataclasses import dataclass
from pathlib import Path
from firecore_torch.modules.base import BaseModel
from typing import Dict
from torch import nn
import torch.nn.functional as F
import torch
from icecream import ic
from firecore_torch.helpers.distributed import init_process_group


logger = get_logger(__name__)


@dataclass
class Args(ta.TypedArgs):
    config: Path = ta.add_argument('-c', '--config', type=Path, required=True)
    device: str = ta.add_argument('-d', '--device', default='cpu')


class Net(BaseModel):

    def __init__(self, in_rules=None, out_rules=None) -> None:
        super().__init__(in_rules, out_rules)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def _forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return {'output': x}


def train():
    pass


def test():
    pass


def get_backend(device_type: str):
    return {
        'cuda': 'nccl',
        'cpu': 'gloo',
    }[device_type]


@firecore.main_fn
def main():
    firecore.logging.init()

    args = Args.from_args()

    device = torch.device(args.device)
    backend = get_backend(device.type)

    init_process_group(backend)

    ic(args)

    cfg = firecore.config.from_file(str(args.config), jpathdir='.')

    ic(cfg)

    data = firecore.resolve(cfg['data'])
    ic(data['train'])
