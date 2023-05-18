import typed_args as ta
from typing import Optional
from dataclasses import dataclass


@dataclass
class DeepspeedGroup:
    deepspeed: bool = ta.add_argument(
        '--deepspeed', action='store_true',
    )
    "Enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)"

    deepspeed_config: Optional[str] = ta.add_argument(
        '--deepspeed-config', type=str,
    )
    "DeepSpeed json configuration file."

    deepspeed_mpi: bool = ta.add_argument(
        '--deepspeed-mpi', action='store_true',
    )
    """
    Run via MPI, this will attempt to discover the necessary variables to initialize torch
    distributed from the MPI environment
    """
