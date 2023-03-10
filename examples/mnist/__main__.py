from firecore_torch.metrics import Accuracy, Average
import firecore
import typed_args as ta
from dataclasses import dataclass
from pathlib import Path
from firecore_torch.modules.base import BaseModel
from typing import Dict, List
from torch import nn
import torch.nn.functional as F
import torch
from icecream import ic
from firecore_torch.helpers.distributed import init_process_group
from torch import Tensor
from firecore_torch.runners import Trainer
from firecore_torch import helpers
import logging
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class Args(ta.TypedArgs):
    config: Path = ta.add_argument('-c', '--config', type=Path, required=True)
    device: str = ta.add_argument('-d', '--device', default='cpu')
    work_dir: Path = ta.add_argument(
        '-w', '--work-dir', type=Path, required=True)


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


def get_backend(device_type: str):
    return {
        'cuda': 'nccl',
        'cpu': 'gloo',
    }[device_type]


@firecore.main_fn
def main():
    # import tracemalloc

    # tracemalloc.start()

    # firecore.logging.init(level='INFO')

    args = Args.from_args()

    args.work_dir.mkdir(parents=True)

    firecore.logging.init(
        filename=str(args.work_dir / 'run.log'),
        level=logging.INFO,
    )

    device = torch.device(args.device)
    backend = get_backend(device.type)

    init_process_group(backend)

    ic(args)

    cfg = firecore.config.from_file(str(args.config), jpathdir='.')

    ic(cfg)

    base_model: nn.Module = firecore.resolve(cfg['model'])
    model = helpers.make_dist_model(base_model, device)
    ic(base_model)
    criterion: nn.Module = firecore.resolve(cfg['criterion'])
    criterion.to(device)
    ic(criterion)
    params = firecore.resolve(cfg['params'], model=base_model)
    ic(params)
    optimizer = firecore.resolve(cfg['optimizer'], params=params)
    ic(optimizer)
    lr_scheduler = firecore.resolve(cfg['lr_scheduler'], optimizer=optimizer)
    ic(lr_scheduler)

    summary_writer = SummaryWriter(
        log_dir=str(args.work_dir/'tf_logs')
    )

    plans: List[Dict] = cfg['plans']

    shared = dict(
        base_model=base_model,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        max_epochs=cfg['base']['max_epochs'],
        summary_writer=summary_writer,
    )

    workflows: Dict[str, Trainer] = {}
    for plan in plans:
        key = plan['key']
        workflows[key] = firecore.resolve(
            cfg[key]
        )(**shared)

    for epoch in range(1, 2 + 1):
        for plan in plans:
            if epoch % plan['interval'] == 0:
                workflow = workflows[plan['key']]
                workflow.step(epoch)

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')

    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     print(stat)
