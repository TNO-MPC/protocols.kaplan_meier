from typing import List, Optional, Sequence, Union

class rv_continuous:
    def __init__(
        self, a: Optional[float] = None, name: Optional[str] = None
    ) -> None: ...
    def sf(
        self, x: Union[float, Sequence[float]], *args: int, **kwds: Optional[float]
    ) -> List[float]: ...
