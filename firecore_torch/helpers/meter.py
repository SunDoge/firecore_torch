from typing import Optional
import time
import datetime


class Meter:
    """
    TODO: move to firecore
    """

    def __init__(
        self,
        total: Optional[int] = None,
    ) -> None:
        self._total = total
        self._count = 0
        self._start_time = time.perf_counter()

    @property
    def count(self) -> int:
        return self._count

    @property
    def total(self) -> Optional[int]:
        return self._total

    @property
    def rate(self) -> float:
        return self.count / self.elapsed

    @property
    def remaining(self) -> float:
        assert self.total
        return (self.total - self.count) / self.rate

    @property
    def elapsed(self) -> float:
        end_time = time.perf_counter()
        return end_time - self._start_time

    @property
    def eta_datetime(self) -> datetime.datetime:
        return datetime.datetime.now() + self.remaining_timedelta

    @property
    def remaining_timedelta(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.remaining)

    @property
    def is_updated(self) -> bool:
        return self.count > 0

    def step(self, n: int = 1):
        self._count += n

    def reset(self):
        self._count = 0
        self._start_time = time.perf_counter()
