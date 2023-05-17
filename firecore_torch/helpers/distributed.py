import torch.distributed as dist
from dataclasses import dataclass
import os
from typing import TypeVar
from firecore.system import find_free_port
from torch.utils.data import graph_settings

T = TypeVar('T')


def _get_or_insert(key: str, value: T) -> T:
    if key not in os.environ:
        os.environ[key] = str(value)
    return type(value)(os.environ[key])


@dataclass
class DistInfo:
    rank: int = _get_or_insert('RANK', 0)
    world_size: int = _get_or_insert('WORLD_SIZE', 1)
    master_addr: str = _get_or_insert('MASTER_ADDR', '127.0.0.1')
    master_port: int = _get_or_insert('MASTER_PORT', find_free_port())


def init_process_group(backend: str):
    dist_info = DistInfo()
    dist.init_process_group(backend)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank_safe() -> int:
    if is_distributed():
        return dist.get_rank()
    else:
        return 0


def get_world_size_safe() -> int:
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1
