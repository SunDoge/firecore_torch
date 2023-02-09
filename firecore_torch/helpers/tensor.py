from torch import Tensor
from typing import Dict, Any
import torch
import numpy as np


def copy_to_device(batch: Dict[str, Tensor], device: torch.device, non_blocking: bool = True) -> Dict[str, Tensor]:
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def detach_all(tensor_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {k: v.detach() for k, v in tensor_dict.items()}


def copy_to_py(tensor_dict: Dict[str, Tensor]) -> Dict[str, np.ndarray]:
    return {k: v.cpu().numpy() for k, v in tensor_dict.items()}


def copy_to_py(tensor_dict: Dict[str, Tensor]) -> Dict[str, Any]:
    return {k: v.tolist() for k, v in tensor_dict.items()}
