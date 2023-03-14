from firecore_torch.metrics.average import Average
from firecore_torch.metrics.accuracy import Accuracy
import torch
from firecore_torch.testing.distributed import init_cpu_process_group


def test_avg():
    with init_cpu_process_group():
        meter = Average()
        num_samples = 10
        x = torch.rand(num_samples)
        for val in x:
            meter.update(val, 1)

        result = meter.compute()
        assert torch.allclose(result['val'], x[-1])
        assert torch.allclose(result['avg'], x.mean())

        meter.sync().wait()
        result = meter.compute()
        assert torch.allclose(result['val'], x[-1])
        assert torch.allclose(result['avg'], x.mean())


def test_accuracy():
    with init_cpu_process_group():
        acc = Accuracy(topk=[1, 5])

        x = torch.rand(10, 10)
        x[:, 0] = 1.0
        x[1:, 1] = 2.0
        y = torch.zeros(10, dtype=torch.long)

        acc.update(x, y)
        res = acc.compute()
        assert res['acc1'] == 0.1
        assert res['acc5'] == 1.0

        acc.sync()
        res = acc.compute()
        assert res['acc1'] == 0.1
        assert res['acc5'] == 1.0
