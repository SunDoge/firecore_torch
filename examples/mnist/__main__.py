from firecore_torch.metrics import Accuracy, Average
import firecore
from firecore.logging import get_logger
import typed_args as ta
from dataclasses import dataclass
from pathlib import Path

logger = get_logger(__name__)


@dataclass
class Args(ta.TypedArgs):
    config: Path = ta.add_argument('-c', '--config', type=Path, required=True)


@firecore.main_fn
def main():
    firecore.logging.init()

    args = Args.from_args()

    print(args)

    cfg = firecore.config.from_file(str(args.config), jpathdir='.')

    print(cfg)
