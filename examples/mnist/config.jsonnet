{
  base: {
    batch_size: 64,
    max_epochs: 14,
    lr: 1.0,
    num_workers: 2,
  },
  data: {
    train: {
      _call: 'firecore_torch.helpers.data.make_data',
      transform: {
        _call: 'examples.mnist.data.train_transform',
      },
      make_dataset: {
        _partial: 'examples.mnist.data.Mnist',
        root: 'data',
        train: true,
        download: true,
      },
      make_loader: {
        _partial: 'firecore_torch.helpers.data.make_loader',
        num_workers: $.base.num_workers,
        batch_size: $.base.batch_size,
        shuffle: true,
      },
    },
    test: {
      _call: 'firecore_torch.helpers.data.make_data',
      transform: {
        _call: 'examples.mnist.data.test_transform',
      },
      make_dataset: {
        _partial: 'examples.mnist.data.Mnist',
        root: 'data',
        train: false,
        download: true,
      },
      make_loader: {
        _partial: 'firecore_torch.helpers.data.make_loader',
        num_workers: $.base.num_workers,
        batch_size: $.base.batch_size * 2,
        shuffle: false,
      },
    },
  },
  model: {

  },
  make_optimizer: {

  },
  make_scheduler: {

  },
  metrics: {
    train: {
      _call: 'firecore_torch.metrics.MetricCollection',
      metrics: [
        { _call: 'firecore_torch.metrics.Average', in_rules: { output: 'loss', target: 'target' } },
        { _call: 'firecore_torch.metrics.Accuracy', topk: [1, 5], in_rules: { output: 'output', target: 'target' } },
      ],
    },
    test: {

    },
  },
}
