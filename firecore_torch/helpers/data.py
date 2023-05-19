from torch.utils.data import Dataset, DataLoader
from typing import Type, Callable, Dict, Any
from torch.utils.data import DistributedSampler, Sampler
from typing import Optional
import logging
import torch.multiprocessing as mp
import firecore
from torch import Tensor
import torch
from . import distributed as dist_utils
from torch.utils.data import IterDataPipe
import torch.multiprocessing as mp
from torch.utils.data.graph_settings import apply_sharding

logger = logging.getLogger(__name__)


def make_data(
    transform,
    dataset,
    loader,
):
    dataset_instance = dataset(transform=transform)
    loader_instance = loader(dataset=dataset_instance)
    return loader_instance


def make_loader(
    dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    **kwargs,
):
    if dist_utils.is_distributed():
        if shuffle:
            sampler = DistributedSampler(dataset)
        else:
            sampler = DistributedExactSampler(dataset)
    else:
        sampler = None
    logger.info('Make sampler: %s', sampler.__class__)
    multiprocessing_context = mp.get_context(
        'fork') if num_workers > 0 else None
    default_kwargs = dict(
        multiprocessing_context=multiprocessing_context,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    default_kwargs.update(kwargs)

    loader = DataLoader(
        dataset,
        **default_kwargs,
    )

    return loader


def num_samples_in_rank(num_samples: int, rank: int, num_replicas: int) -> int:
    n = num_samples - (rank + 1)
    return n // num_replicas + 1


class DistributedExactSampler(Sampler):

    def __init__(
        self,
        data_source: Dataset,
        rank: Optional[int] = None,
        num_replicas: Optional[int] = None,
    ) -> None:
        super().__init__(data_source)
        if num_replicas is None:
            num_replicas = dist_utils.get_world_size_safe()

        if rank is None:
            rank = dist_utils.get_rank_safe()

        if rank >= num_replicas or rank < 0:
            raise ValueError('invalid rank {}', rank)

        total_size = len(data_source)
        num_samples = num_samples_in_rank(total_size, rank, num_replicas)

        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = num_samples
        self.total_size = total_size

    def __iter__(self):
        indices = list(range(self.total_size))
        sub_indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(sub_indices) == self.num_samples
        return iter(sub_indices)

    def __len__(self):
        return self.num_samples


def create_data_loader_from_pipeline(
    dp: IterDataPipe,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    mp_ctx=mp.get_context('fork'),
    collate_fn=None,
    worker_init_fn=None,
    persistent_workers: bool = True,
    **kwargs,
):
    """
    worker_init_fn will configure sharding
    """

    if pin_memory and persistent_workers:
        logger.warning(
            "sometimes pin_memory + persistent_workers would be buggy"
            "If you meet some bug, please turn off one of them"
        )

    return DataLoader(
        dp,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        multiprocessing_context=mp_ctx,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        **kwargs
    )
