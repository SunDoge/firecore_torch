import typed_args as ta
from pathlib import Path
import torch


@ta.argument_parser()
class Args:
    config: Path = ta.add_argument(
        '-c', '--config',
        type=Path,
        help='config path'
    )
    work_dir: Path = ta.add_argument(
        '-w', '--work-dir',
        type=Path,
        help='dir for checkpoints, logs and etc'
    )
    device: torch.device = ta.add_argument(
        '-d', '--device',
        default='cpu',
        type=torch.device,
        help='torch device'
    )
