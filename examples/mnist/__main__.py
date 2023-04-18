from firecore_torch.metrics import Accuracy, Average
import firecore
import typed_args as ta
from pathlib import Path

from typing import Dict, List
from torch import nn

import torch
from icecream import ic
from firecore_torch.helpers.distributed import init_process_group
from torch import Tensor
from firecore_torch.runners import Trainer
from firecore_torch import helpers
import logging
from torch.utils.tensorboard import SummaryWriter
from firecore_torch.helpers.arguments import Args

logger = logging.getLogger(__name__)





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

    args = Args.parse_args()

    args.work_dir.mkdir(parents=True)

    firecore.logging.init(
        filename=str(args.work_dir / 'run.log'),
        level=logging.INFO,
    )

    device = args.device
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

    max_epochs: int = cfg['base']['max_epochs']

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
        max_epochs=max_epochs,
        summary_writer=summary_writer,
        work_dir=args.work_dir,
    )

    workflows: Dict[str, Trainer] = {}
    for plan in plans:
        key = plan['key']
        workflows[key] = firecore.resolve(
            cfg[key]
        )(**shared, stage=key)

    for epoch in range(1, 2 + 1):
        for plan in plans:
            if epoch % plan['interval'] == 0:
                workflow = workflows[plan['key']]
                workflow.step(epoch)
