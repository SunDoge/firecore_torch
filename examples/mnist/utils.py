from torchvision.datasets import MNIST as Base
from torchvision import transforms
from torch import nn


class Mnist(Base):

    def __getitem__(self, index: int):
        data, target = super().__getitem__(index)
        return dict(data=data, target=target)


def train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ])


def test_transform():
    return train_transform()


def get_params(model: nn.Module):
    return model.parameters()
