from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

__all__ = ["TokenizerProtocol"]


class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> list[int]:
        """Return the token IDs for ``text`` (``POST /tokenize``)."""
        ...

    def decode(self, tokens: Iterable[int]) -> str:
        """Return the text for ``tokens`` (``POST /detokenize``)."""
        ...

    def __call__(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...
