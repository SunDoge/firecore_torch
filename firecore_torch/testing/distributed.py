import torch.distributed as dist
from contextlib import contextmanager
from firecore.system import find_free_port
from firecore.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def init_cpu_process_group():
    dist_url = 'tcp://127.0.0.1:{}'.format(find_free_port())

    logger.info('init process group', dist_url=dist_url)
    dist.init_process_group(
        'GLOO',
        init_method=dist_url,
        world_size=1,
        rank=0,
    )

    yield

    logger.info('destroy process group')
    dist.destroy_process_group()
