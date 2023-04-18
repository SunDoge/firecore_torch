from .utils import Net
from firecore.config import LazyCall
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


train_transform = LazyCall(transforms.Compose)(
    transforms=[
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081]),
    ]
)
test_transform = train_transform

train_dataset = LazyCall(MNIST)(
    root="./data",
    transform=train_transform,
    train=True,
    download=True,
)
test_dataset = LazyCall(MNIST)(
    root="./data",
    train=False,
    download=True,
    transform=test_transform,
)

train_loader = LazyCall(DataLoader)(
    dataset=train_dataset,
    batch_size=64,
    num_workers=1,
    shuffle=True,
)
test_loader = LazyCall(DataLoader)(
    dataset=test_dataset,
    batch_size=128,
    num_workers=2,
    shuffle=False,
)

model = LazyCall(Net)(
    in_rules=dict(x="image"),
)


