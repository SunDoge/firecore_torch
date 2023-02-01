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
from torch import Tensor

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


class Workflow:

    def step(self, epoch: int):
        pass


class TrainWorkflow(Workflow):

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        data,
        metric,
        device,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data = data
        self.metric = metric
        self.device = device

    def step(self, epoch: int):
        self.model.train()
        self.data.sampler.set_epoch(epoch)
        self.metric.reset()

        for batch_idx, batch in enumerate(self.data):
            batch = {k: v.to(self.device, non_blocking=True)
                     for k, v in batch.items()}
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            losses: Dict[str, Tensor] = self.criterion(**outputs, **batch)
            losses['loss'].backward()
            self.optimizer.step()

            losses = {k: v.detach() for k, v in losses.items()}
            self.metric.update(**losses, **outputs, **batch)

            if batch_idx % 10 == 0:
                metrics = self.metric.compute()
                logger.info('show metrics', batch_idx=batch_idx,
                            **metrics)

        self.metric.sync()
        metrics = self.metric.compute()
        logger.info('show metrics', epoch=epoch, **metrics)


class TestWorkflow(Workflow):

    def __init__(
        self
    ) -> None:
        super().__init__()

    def step(self, epoch: int):
        return super().step(epoch)


def get_backend(device_type: str):
    return {
        'cuda': 'nccl',
        'cpu': 'gloo',
    }[device_type]


@firecore.main_fn
def main():
    firecore.logging.init(level='DEBUG')

    args = Args.from_args()

    device = torch.device(args.device)
    backend = get_backend(device.type)

    init_process_group(backend)

    ic(args)

    cfg = firecore.config.from_file(str(args.config), jpathdir='.')

    ic(cfg)

    model: nn.Module = firecore.resolve(cfg['model'])
    model.to(device)
    ic(model)
    criterion: nn.Module = firecore.resolve(cfg['criterion'])
    criterion.to(device)
    ic(criterion)
    params = firecore.resolve(cfg['params'], model=model)
    ic(params)
    optimizer = firecore.resolve(cfg['optimizer'], params=params)
    ic(optimizer)
    lr_scheduler = firecore.resolve(cfg['lr_scheduler'], optimizer=optimizer)
    ic(lr_scheduler)

    workflow = cfg['workflow']
    ic(workflow)
    pipelines = [firecore.resolve(cfg[k]) for k in workflow.keys()]
    ic(pipelines)

    train_workflow = TrainWorkflow(
        model, criterion, optimizer, lr_scheduler, pipelines[
            1]['data'], pipelines[1]['metric'], device,
    )
    train_workflow.step(0)
