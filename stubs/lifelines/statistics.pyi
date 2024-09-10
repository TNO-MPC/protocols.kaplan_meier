from __future__ import annotations

from typing import Any, Iterable

class StatisticalResult:
    def __init__(
        self,
        p_value: Iterable[float] | float,
        test_statistic: Iterable[float] | float,
        name: str | None = ...,
        test_name: Iterable[float] | str | None = ...,
        **kwargs: Any,
    ) -> None:
        self.p_value = p_value
        self.test_statistic = test_statistic
        self.test_name = test_name

        for kw, value in kwargs.items():
            setattr(self, kw, value)

def multivariate_logrank_test(
    event_durations: Iterable[int],
    groups: Iterable[int],
    event_observed: Iterable[int] | None = None,
    weights: Iterable[float] | None = None,
    t_0: float = -1,
    weightings: str | None = None,
    **kwargs: Any,
) -> StatisticalResult: ...
