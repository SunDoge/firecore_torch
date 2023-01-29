import firecore_torch.metrics.functional as F
import torch


def test_topk_correct():
    x = torch.rand(10, 10)
    x[:, 0] = 1.0
    x[1:, 1] = 2.0
    y = torch.zeros(10, dtype=torch.long)
    corrects = F.topk_correct(x, y, topk=[1, 5])
    assert corrects[0] == 1
    assert corrects[1] == 10
