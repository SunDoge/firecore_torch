from .base import BaseHook


class EtaHook(BaseHook):
    """
    "eta" is the abbreviation of "estimated time of arrival."
    We can remove eta_meter from training hook
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO
