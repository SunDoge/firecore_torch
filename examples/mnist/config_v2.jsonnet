{
  base: {
    batch_size: 128,
    num_workers: 2,
    max_epochs: 14,
  },
  model: {
    _call: 'examples.mnist.__main__.Net',
    in_rules: { x: 'data' },
  },
  criterion: {
    _call: 'firecore_torch.modules.loss.Loss',
    loss_fn: {
      _call: 'torch.nn.CrossEntropyLoss',
    },
    in_rules: { output: 'output', target: 'target' },
  },
  params: {
    _call: 'examples.mnist.utils.get_params',
    model: '$model',
  },
  optimizer: {
    _call: 'torch.optim.Adadelta',
    params: '$params',
    lr: 0.1,  // Should be rewrite
  },
  lr_scheduler: {
    _call: 'torch.optim.lr_scheduler.StepLR',
    optimizer: '$optimizer',
    step_size: 1,
    gamma: 0.7,
  },

  train: {
    _partial: 'firecore_torch.workflow.Trainer',
    data: {
      _call: 'firecore_torch.helpers.data.make_data',
      transform: {
        _call: 'examples.mnist.utils.train_transform',
      },
      dataset: {
        _partial: 'examples.mnist.utils.Mnist',
        root: 'data',
        train: true,
        download: true,
        transform: '_',
      },
      loader: {
        _partial: 'firecore_torch.helpers.data.make_loader',
        dataset: '_',
        num_workers: $.base.num_workers,
        batch_size: $.base.batch_size,
        shuffle: true,
      },
    },
    metrics: $.test.metrics,
    log_interval: 10,
  },
  test: {
    _partial: 'firecore_torch.workflow.Evaluator',
    data: {
      _call: 'firecore_torch.helpers.data.make_data',
      transform: {
        _call: 'examples.mnist.utils.test_transform',
      },
      dataset: {
        _partial: 'examples.mnist.utils.Mnist',
        root: 'data',
        train: false,
        transform: null,
      },
      loader: {
        _partial: 'firecore_torch.helpers.data.make_loader',
        dataset: null,
        num_workers: $.base.num_workers * 2,
        batch_size: $.base.batch_size * 2,
        shuffle: false,
      },
    },
    metrics: {
      _call: 'firecore_torch.metrics.MetricCollection',
      metrics: [
        {
          _call: 'firecore_torch.metrics.Average',
          in_rules: { output: 'loss', target: 'target' },
          out_rules: { loss: 'avg' },
        },
        {
          _call: 'firecore_torch.metrics.Accuracy',
          topk: [1, 5],
        },
      ],
    },
    log_interval: 10,
  },
  plans: [
    { key: 'train', interval: 1 },
    { key: 'test', interval: 1 },
  ],
}
