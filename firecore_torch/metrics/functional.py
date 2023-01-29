from torch import Tensor
from typing import List


def topk_correct(output: Tensor, target: Tensor, topk: List[int] = [1]) -> List[Tensor]:
    maxk = max(topk)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    # [K, B]
    pred: Tensor = pred.t()
    # [B] -> [1, B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k)
    return res
