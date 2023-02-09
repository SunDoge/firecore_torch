from torch.nn.parallel import DistributedDataParallel
from torch import nn
import torch
from firecore.logging import get_logger

logger = get_logger(__name__)


def make_dist_model(base_model: nn.Module, device: torch.device) -> DistributedDataParallel:
    base_model.to(device)
    device_ids = None
    if device.type == 'cuda':
        device_ids = [device]
        logger.info('set device_ids', device_ids=device_ids)
    return DistributedDataParallel(base_model, device_ids=device_ids)
