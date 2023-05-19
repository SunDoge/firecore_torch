from firecore_torch.metrics.average import Average
from firecore_torch.metrics.accuracy import Accuracy
import firecore_torch.metrics as M

import torch
from firecore_torch.testing.distributed import init_cpu_process_group


def test_avg():
    with init_cpu_process_group():
        meter = Average()
        num_samples = 10
        x = torch.rand(num_samples)
        for val in x:
            meter._update(val, 1)

        result = meter._compute()

        assert torch.allclose(result, x.mean())

        meter.sync()
        result = meter._compute()
        assert torch.allclose(result, x.mean())


def test_accuracy():
    with init_cpu_process_group():
        acc = Accuracy(topk=[1, 5])

        x = torch.rand(10, 10)
        x[:, 0] = 1.0
        x[1:, 1] = 2.0
        y = torch.zeros(10, dtype=torch.long)

        acc._update(x, y)
        acc1, acc5 = acc._compute()
        assert acc1 == 0.1
        assert acc5 == 1.0

        acc.sync()
        acc1, acc5 = acc._compute()
        assert acc1 == 0.1
        assert acc5 == 1.0


def test_sample_counter():
    with init_cpu_process_group():
        counter = M.SampleCounter()
        counter._update(10)
        counter._update(20)
        assert counter._compute() == 30

        counter.sync()

        assert counter._compute() == 30
