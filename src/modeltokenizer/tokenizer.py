from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .errors import TransportError
from .payload import DetokenizePayload, TokenizePayload
from .transport import (
    LlamaCppHttpTransport,
    TransportProtocol,
    text_from_body,
    tokens_from_body,
)

__all__ = [
    "LlamaCppHttpTransport",
    "ModelTokenizer",
    "TransportError",
    "TransportProtocol",
    "text_from_body",
    "tokens_from_body",
]


@dataclass(slots=True)
class ModelTokenizer:
    """Count tokens with a llama.cpp-compatible tokenizer.

    The splitter probes many candidate spans per chunk, so every ``size()``
    call is a blocking round trip and counts are not cached: each call sends a
    request through :attr:`transport`.

    Attributes:
        transport: Realization of the endpoint calls.
            :class:`LlamaCppHttpTransport` is the default; tests can inject an
            in-memory fake.
        add_special: Insert BOS/EOS. Keep ``False`` so the count matches the
            chunk's own content; add the model's special tokens to ``capacity``
            separately.
        parse_special: Tokenize special-token *text* as special tokens (server
            default) so counts match what the model actually sees at inference.
        with_pieces: Ask the server for ``{"id", "piece"}`` tokens instead of
            plain IDs. :meth:`encode` still returns IDs.
    """

    transport: TransportProtocol
    add_special: bool = False
    parse_special: bool = True
    with_pieces: bool = False

    @property
    def model_name(self) -> str:
        """Model name carried by :attr:`transport`."""
        return self.transport.model_name

    def __repr__(self) -> str:
        return f"ModelTokenizer(transport={self.transport!r})"

    # ------------------------------------------------------------------ API

    def __call__(self, text: str) -> int:
        """Number of tokens in ``text`` — the splitter's ``size()`` callback."""
        if not text:
            return 0
        return len(self.encode(text))

    def encode(self, text: str) -> list[int]:
        """Token IDs for ``text`` (``POST /tokenize``, no caching)."""
        if not text:
            return []
        payload = self.payload(text)
        return self.transport.tokenize(payload)

    def decode(self, tokens: Iterable[int]) -> str:
        """Text for ``tokens`` via ``POST /detokenize``.

        The inverse of :meth:`encode`; the server calls this operation
        ``detokenize``.
        """
        ids = list(tokens)
        if not ids:
            return ""
        payload = DetokenizePayload(
            tokens=ids,
            model=self.model_name,
        )
        return self.transport.detokenize(payload)

    def payload(self, text: str) -> TokenizePayload:
        """The :class:`TokenizePayload` this instance sends for ``text``."""
        return TokenizePayload(
            content=text,
            model=self.model_name,
            add_special=self.add_special,
            parse_special=self.parse_special,
            with_pieces=self.with_pieces,
        )
