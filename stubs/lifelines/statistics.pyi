from typing import Any, Iterable, Optional, Union

class StatisticalResult:
    def __init__(
        self,
        p_value: Union[Iterable[float], float],
        test_statistic: Union[Iterable[float], float],
        name: Optional[str] = ...,
        test_name: Optional[Union[Iterable[float], str]] = ...,
        **kwargs: Any
    ) -> None:
        self.p_value = p_value
        self.test_statistic = test_statistic
        self.test_name = test_name

        for kw, value in kwargs.items():
            setattr(self, kw, value)
