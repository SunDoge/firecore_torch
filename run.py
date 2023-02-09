from firecore_torch.hooks import TrainingHook, InferenceHook, TextLoggerHook
from firecore_torch.runners.basic import EpochBasedRunner
import torch
from firecore_torch.modules.base import BaseModel
from firecore_torch.metrics import MetricCollection, Average

import firecore
from firecore_torch.modules.loss import Loss
import logging


class SimpleModel(BaseModel):

    def __init__(self, in_rules=None, out_rules=None) -> None:
        super().__init__(in_rules, out_rules)
        self.linear = torch.nn.Linear(4, 4)

    def _forward(self, x: torch.Tensor, **kwargs):
        y = self.linear(x)
        return {'output': y}


def main():
    firecore.logging_v2.init(level=logging.DEBUG)
    base_model = SimpleModel()
    model = base_model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device('cpu')
    data = [dict(
        x=torch.rand(10, 4),
        y=torch.rand(10, 4),
    )]
    basic_runner = EpochBasedRunner(
        base_model=base_model,
        model=model,
        criterion=Loss(
            torch.nn.MSELoss(),
            in_rules={'output': 'output', 'target': 'y'}
        ),
        data=data,
        metrics=MetricCollection(dict(
            loss=Average(
                in_rules={'output': 'loss', 'target': 'y'},
                out_rules={'loss': 'avg'}
            ),
            loss2=Average(
                in_rules={'output': 'loss', 'target': 'y'},
                out_rules={'loss2': 'avg'}
            )
        )),
        device=device,
        prefix='basic',
        hooks=[TextLoggerHook(
            [dict(key='loss', fmt=':.4f'), dict(key='loss2', fmt=':.4f')])],
        optimizer=optimizer
    )
    basic_runner.step(1, stage='basic')

    # basic_workflow._hooks = [InferenceHook()]
    # basic_workflow.step(1)

    # basic_workflow._hooks = [TrainingHook()]
    # basic_workflow.step(1)


if __name__ == '__main__':
    main()
