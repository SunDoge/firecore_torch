from typing import Dict, List


class MetricHistory:

    def __init__(
        self,
        key: str,
        
    ):
        self._history: Dict[str, List[float]] = {}


