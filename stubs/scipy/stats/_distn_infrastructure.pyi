from __future__ import annotations

from typing import Sequence

class rv_continuous:
    def __init__(self, a: float | None = None, name: str | None = None) -> None: ...
    def sf(
        self, x: float | Sequence[float], *args: int, **kwds: float | None
    ) -> list[float]: ...
