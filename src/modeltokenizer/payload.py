from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DetokenizePayload", "TokenizePayload"]


@dataclass(slots=True)
class TokenizePayload:
    """Body of a ``POST /tokenize`` request.

    Attributes:
        content: Text to tokenize (required by the server).
        model: Model name for routers/proxies (required); a plain llama.cpp
            server ignores it. Always a string; always included in the
            request body.
        add_special: Insert special tokens such as BOS. Server default is
            ``False``.
        parse_special: Tokenize special-token *text* as special tokens.
            Server default is ``True``.
        with_pieces: Return ``{"id", "piece"}`` objects instead of plain IDs.
            Server default is ``False``.
    """

    content: str
    model: str
    add_special: bool = False
    parse_special: bool = True
    with_pieces: bool = False


@dataclass(slots=True)
class DetokenizePayload:
    """Body of a ``POST /detokenize`` request.

    Attributes:
        tokens: Token IDs to convert back to text.
        model: Model name for routers/proxies (required); a plain llama.cpp
            server ignores it. Always a string; always included in the
            request body.
    """

    tokens: list[int]
    model: str
