from __future__ import annotations

import json
import platform
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any, Protocol

from .errors import TransportError
from .payload import DetokenizePayload, TokenizePayload

__all__ = [
    "LlamaCppHttpTransport",
    "TransportError",
    "TransportProtocol",
    "VllmHttpTransport",
    "text_from_body",
    "tokens_from_body",
]


def user_agent() -> str:
    """``User-Agent`` sent with every request, e.g.
    ``model-tokenizer/1.2.0 (CPython/3.12.4)``."""
    package_ver = version("model-tokenizer")
    comment = platform.python_implementation() + "/" + platform.python_version()
    return f"model-tokenizer/{package_ver} ({comment})"


class TransportProtocol(Protocol):
    """The effectful seam between the tokenizer and a remote server.

    Implementations turn the two logical operations into requests. The
    protocol says nothing about HTTP: an in-memory fake can answer directly,
    and a different backend can speak a different wire format.
    """

    @property
    def model_name(self) -> str:
        """Model the server should use for this transport."""
        ...

    def tokenize(self, payload: TokenizePayload) -> list[int]:
        """Return the token IDs for ``payload.content``."""
        ...

    def detokenize(self, payload: DetokenizePayload) -> str:
        """Return the text for ``payload.tokens``."""
        ...


@dataclass(slots=True)
class LlamaCppHttpTransport:
    """Perform the token endpoints over HTTP with :mod:`urllib`.

    Attributes:
        base_url: Server origin, e.g. ``http://127.0.0.1:9931`` (a trailing
            ``/tokenize`` is tolerated).
        model_name: Model name sent with each request; llama.cpp ignores it,
            routers/proxies in front of it (llama-swap, LiteLLM) use it to pick
            a backend. Also used in error messages.
        api_key: Sent as ``Authorization: Bearer <key>`` when set.
        timeout: Per-request socket timeout in seconds.
    """

    base_url: str
    model_name: str
    api_key: str | None = None
    timeout: float = 7200

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url is required")
        self.base_url = self.base_url.rstrip("/").removesuffix("/tokenize")

    def __repr__(self) -> str:
        key = "***" if self.api_key else None
        return (
            f"LlamaCppHttpTransport(base_url={self.base_url!r}, "
            f"model_name={self.model_name!r}, api_key={key!r}, "
            f"timeout={self.timeout})"
        )

    def tokenize(self, payload: TokenizePayload) -> list[int]:
        """POST ``payload.content`` to ``/tokenize`` and return token IDs."""
        body = self._post("/tokenize", payload)
        return tokens_from_body(body, self.model_name)

    def detokenize(self, payload: DetokenizePayload) -> str:
        """POST ``payload.tokens`` to ``/detokenize`` and return text."""
        body = self._post("/detokenize", payload)
        return text_from_body(body, self.model_name)

    # -------------------------------------------------------------- transport

    def _url(self, path: str, payload: TokenizePayload | DetokenizePayload) -> str:
        url = f"{self.base_url}{path}"
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{urllib.parse.urlencode({'model': payload.model})}"
        return url

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent(),
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(
        self, path: str, payload: TokenizePayload | DetokenizePayload
    ) -> dict[str, Any]:
        body = asdict(payload)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        url = self._url(path, payload)
        who = f"model {self.model_name!r}"
        return _post_json(url, data, self._headers(), self.timeout, who)


def _post_json(
    url: str,
    data: bytes,
    headers: dict[str, str],
    timeout: float,
    who: str,
) -> dict[str, Any]:
    """POST ``data`` and parse a JSON response, mapping failures to TransportError."""
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = f"HTTP {exc.code}: {_read_error_body(exc)}"
        raise TransportError(f"POST {url} for {who} -> {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransportError(f"POST {url} for {who} failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TransportError(
            f"POST {url} for {who} returned invalid JSON: {exc}"
        ) from exc


def tokens_from_body(body: dict[str, Any], model_name: str) -> list[int]:
    """Extract token IDs from a ``/tokenize`` response body.

    Accepts plain IDs and ``{"id", "piece"}`` objects (``with_pieces``).
    """
    tokens = body.get("tokens")
    if not isinstance(tokens, list):
        raise TransportError(
            f"unexpected /tokenize response from {model_name!r}: {body!r}"
        )
    ids: list[int] = []
    for token in tokens:
        if isinstance(token, dict):
            if "id" not in token:
                raise TransportError(
                    f"unexpected /tokenize token from {model_name!r}: {token!r}"
                )
            ids.append(int(token["id"]))
        else:
            ids.append(int(token))
    return ids


def text_from_body(body: dict[str, Any], model_name: str) -> str:
    """Extract decoded text from a ``/detokenize`` response body."""
    for key in ("content", "text"):
        value = body.get(key)
        if isinstance(value, str):
            return value
    raise TransportError(
        f"unexpected /detokenize response from {model_name!r}: {body!r}"
    )


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:500]
    except Exception:  # noqa: BLE001 - best effort diagnostics only
        return exc.reason or ""
    finally:
        exc.close()


@dataclass(slots=True)
class VllmHttpTransport:
    """Perform vLLM's token endpoints over HTTP with :mod:`urllib`.

    vLLM speaks a slightly different wire format than llama.cpp: the text
    field is ``prompt`` instead of ``content``, special-token handling is
    ``add_special_tokens``, and detokenization returns ``text``. vLLM requires
    ``model`` in the request body, so :attr:`model_name` is always sent.

    Endpoints: ``POST {base_url}/tokenize`` and ``POST {base_url}/detokenize``.

    Attributes:
        base_url: Server origin, e.g. ``http://127.0.0.1:8000`` or
            ``http://127.0.0.1:8000/v1`` (a trailing ``/tokenize`` is
            tolerated).
        model_name: Model served by vLLM; sent as ``model`` in each request
            body. Also used in error messages.
        api_key: Sent as ``Authorization: Bearer <key>`` when set.
        timeout: Per-request socket timeout in seconds.
    """

    base_url: str
    model_name: str
    api_key: str | None = None
    timeout: float = 7200

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url is required")
        self.base_url = self.base_url.rstrip("/").removesuffix("/tokenize")

    def __repr__(self) -> str:
        key = "***" if self.api_key else None
        return (
            f"VllmHttpTransport(base_url={self.base_url!r}, "
            f"model_name={self.model_name!r}, api_key={key!r}, "
            f"timeout={self.timeout})"
        )

    def tokenize(self, payload: TokenizePayload) -> list[int]:
        """POST ``payload.content`` to ``/tokenize`` and return token IDs."""
        who = f"model {self.model_name!r}"
        body = _post_json(
            f"{self.base_url}/tokenize",
            json.dumps(
                {
                    "model": self.model_name,
                    "prompt": payload.content,
                    "add_special_tokens": payload.add_special,
                },
                ensure_ascii=False,
            ).encode(),
            self._headers(),
            self.timeout,
            who,
        )
        return tokens_from_body(body, self.model_name)

    def detokenize(self, payload: DetokenizePayload) -> str:
        """POST ``payload.tokens`` to ``/detokenize`` and return text."""
        who = f"model {self.model_name!r}"
        body = _post_json(
            f"{self.base_url}/detokenize",
            json.dumps(
                {"model": self.model_name, "tokens": list(payload.tokens)}
            ).encode(),
            self._headers(),
            self.timeout,
            who,
        )
        return text_from_body(body, self.model_name)

    # -------------------------------------------------------------- transport

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent(),
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
