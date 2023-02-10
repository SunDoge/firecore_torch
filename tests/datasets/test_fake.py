from firecore_torch.datasets.fake import FakeDatset
import torch


def test_fake():
    fake_ds = FakeDatset(
        image=lambda: torch.rand([2, 3]),
        target=lambda: torch.zeros([], dtype=torch.long)
    )
    assert fake_ds[10]['image'].shape == (2, 3)
    assert fake_ds[10]['target'] == 0
