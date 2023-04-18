from torchvision.datasets import MNIST as Base
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from firecore_torch.modules.base import BaseModel
import torch


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
        return {"output": x}


def train_transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])]
    )


def test_transform():
    return train_transform()


def get_params(model: nn.Module):
    return model.parameters()
