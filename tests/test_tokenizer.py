"""Tests for the llama.cpp client, using an in-memory dummy transport."""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from dataclasses import asdict
from typing import Any, cast
from unittest import mock

from modeltokenizer import (
    DetokenizePayload,
    LlamaCppHttpTransport,
    ModelTokenizer,
    TokenizePayload,
    TransportError,
    VllmHttpTransport,
)


class DummyTransport:
    """In-memory remote tokenizer: token ID == code point, detokenize == chr()."""

    def __init__(
        self, model_name: str = "fake", error: Exception | None = None
    ) -> None:
        self.model_name = model_name
        self.error = error
        self.calls: list[tuple[str, TokenizePayload | DetokenizePayload]] = []

    def tokenize(self, payload: TokenizePayload) -> list[int]:
        self.calls.append(("tokenize", payload))
        if self.error is not None:
            raise self.error
        return [ord(c) for c in payload.content]

    def detokenize(self, payload: DetokenizePayload) -> str:
        self.calls.append(("detokenize", payload))
        if self.error is not None:
            raise self.error
        return "".join(chr(i) for i in payload.tokens)


class FakeResponse:
    """Minimal :func:`urllib.request.urlopen` context manager."""

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:  # noqa: PYI034
        return self

    def __exit__(self, *_args: object) -> None:
        pass


class TokenizerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = DummyTransport()
        self.tokenizer = ModelTokenizer(transport=self.transport)


class TestPayloads(unittest.TestCase):
    def test_tokenize_payload_omits_empty_model(self) -> None:
        body = TokenizePayload(content="hi", model="", with_pieces=True)
        self.assertEqual(
            asdict(body),
            {
                "content": "hi",
                "model": "",
                "add_special": False,
                "parse_special": True,
                "with_pieces": True,
            },
        )

    def test_detokenize_payload(self) -> None:
        payload = DetokenizePayload(tokens=[1, 2], model="m")
        self.assertEqual(asdict(payload), {"tokens": [1, 2], "model": "m"})


class TestTransport(unittest.TestCase):
    def test_transport_normalizes_base_url(self) -> None:
        transport = LlamaCppHttpTransport(
            base_url="http://127.0.0.1:9931/tokenize/", model_name="fake"
        )
        self.assertEqual(transport.base_url, "http://127.0.0.1:9931")

    def test_transport_rejects_empty_base_url(self) -> None:
        with self.assertRaises(ValueError):
            LlamaCppHttpTransport(base_url="", model_name="fake")

    def test_repr_masks_api_key(self) -> None:
        transport = LlamaCppHttpTransport(
            base_url="http://127.0.0.1:9931", model_name="m", api_key="secret"
        )
        self.assertNotIn("secret", repr(transport))

    def test_tokenize_posts_model_in_query_and_bearer_token(self) -> None:
        transport = LlamaCppHttpTransport(
            base_url="http://127.0.0.1:9931",
            model_name="fake",
            api_key="secret",
        )
        response = FakeResponse({"tokens": [97]})

        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(
                transport.tokenize(TokenizePayload(content="a", model="fake")),
                [97],
            )

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:9931/tokenize?model=fake")
        self.assertEqual(dict(request.header_items())["Authorization"], "Bearer secret")
        self.assertEqual(
            json.loads(request.data),
            {
                "content": "a",
                "model": "fake",
                "add_special": False,
                "parse_special": True,
                "with_pieces": False,
            },
        )

    def test_detokenize_posts_to_detokenize(self) -> None:
        transport = LlamaCppHttpTransport(
            base_url="http://127.0.0.1:9931", model_name="fake"
        )
        response = FakeResponse({"content": "a"})

        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(
                transport.detokenize(DetokenizePayload(tokens=[97], model="fake")),
                "a",
            )

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url, "http://127.0.0.1:9931/detokenize?model=fake"
        )
        self.assertEqual(json.loads(request.data), {"tokens": [97], "model": "fake"})

    def test_http_error_becomes_transport_error(self) -> None:
        transport = LlamaCppHttpTransport(
            base_url="http://127.0.0.1:9931", model_name="fake"
        )
        error = urllib.error.HTTPError(
            url="http://127.0.0.1:9931/tokenize",
            code=503,
            msg="Service Unavailable",
            hdrs=cast(Any, {}),
            fp=io.BytesIO(b'{"error": "down"}'),
        )

        with (
            mock.patch("urllib.request.urlopen", side_effect=error),
            self.assertRaisesRegex(TransportError, "HTTP 503"),
        ):
            transport.tokenize(TokenizePayload(content="a", model="fake"))


class TestVllmTransport(unittest.TestCase):
    def test_normalize_base_url_and_repr(self) -> None:
        transport = VllmHttpTransport(
            base_url="http://127.0.0.1:8000/v1/tokenize/",
            model_name="m",
            api_key="secret",
        )
        self.assertEqual(transport.base_url, "http://127.0.0.1:8000/v1")
        self.assertNotIn("secret", repr(transport))
        with self.assertRaises(ValueError):
            VllmHttpTransport(base_url="", model_name="m")

    def test_tokenize_posts_prompt_and_add_special_tokens(self) -> None:
        transport = VllmHttpTransport(
            base_url="http://127.0.0.1:8000", model_name="fake", api_key="secret"
        )
        response = FakeResponse({"tokens": [97]})

        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(
                transport.tokenize(TokenizePayload(content="a", model="fake")),
                [97],
            )

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/tokenize")
        self.assertEqual(dict(request.header_items())["Authorization"], "Bearer secret")
        self.assertEqual(
            json.loads(request.data),
            {"model": "fake", "prompt": "a", "add_special_tokens": False},
        )

    def test_detokenize_posts_tokens_and_reads_text(self) -> None:
        transport = VllmHttpTransport(
            base_url="http://127.0.0.1:8000", model_name="fake"
        )
        response = FakeResponse({"text": "a"})

        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(
                transport.detokenize(DetokenizePayload(tokens=[97], model="fake")),
                "a",
            )

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/detokenize")
        self.assertEqual(json.loads(request.data), {"model": "fake", "tokens": [97]})

    def test_http_error_becomes_transport_error(self) -> None:
        transport = VllmHttpTransport(
            base_url="http://127.0.0.1:8000", model_name="fake"
        )
        error = urllib.error.HTTPError(
            url="http://127.0.0.1:8000/tokenize",
            code=404,
            msg="Not Found",
            hdrs=cast(Any, {}),
            fp=io.BytesIO(b'{"detail": "not found"}'),
        )

        with (
            mock.patch("urllib.request.urlopen", side_effect=error),
            self.assertRaisesRegex(TransportError, "HTTP 404"),
        ):
            transport.tokenize(TokenizePayload(content="a", model="fake"))


class TestTokenizer(TokenizerTestCase):
    def test_encode_and_decode_follow_huggingface_names(self) -> None:
        self.assertEqual(self.tokenizer.encode("abc"), [97, 98, 99])
        self.assertEqual(self.tokenizer.decode([97, 98, 99]), "abc")
        self.assertEqual(self.tokenizer.decode([104, 105]), "hi")
        self.assertEqual(self.tokenizer.decode([]), "")
        calls_before = len(self.transport.calls)
        self.assertEqual(self.tokenizer("abc"), 3)
        self.assertEqual(self.tokenizer("abc"), 3)
        # Counts are not cached: both calls hit the transport.
        self.assertEqual(len(self.transport.calls), calls_before + 2)

    def test_empty_text_never_reaches_the_transport(self) -> None:
        self.assertEqual(self.tokenizer(""), 0)
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")
        self.assertEqual(self.transport.calls, [])

    def test_model_is_sent_to_router(self) -> None:
        self.tokenizer.decode([65])
        operation, payload = self.transport.calls[-1]
        self.assertEqual(operation, "detokenize")
        self.assertIsInstance(payload, DetokenizePayload)
        self.assertEqual(payload.model, "fake")
        self.assertEqual(self.tokenizer.payload("x").model, "fake")

    def test_transport_error_is_not_retried(self) -> None:
        error = TransportError("boom")
        transport = DummyTransport(error=error)
        tokenizer = ModelTokenizer(transport=transport)

        with self.assertRaises(TransportError):
            tokenizer("abc")

        self.assertEqual(len(transport.calls), 1)

    def test_tokenizer_delegates_to_injected_transport(self) -> None:
        self.assertEqual(self.tokenizer("abc"), 3)
        self.assertEqual(self.tokenizer.decode([1, 2, 3]), "\x01\x02\x03")
        self.assertEqual(
            [(operation, type(payload)) for operation, payload in self.transport.calls],
            [("tokenize", TokenizePayload), ("detokenize", DetokenizePayload)],
        )


class TestSplitterInterop(TokenizerTestCase):
    def test_semantic_text_splitter_uses_tokenizer_as_callback(self) -> None:
        try:
            from semantic_text_splitter import TextSplitter  # type: ignore[import-not-found, ty:import-not-found]
        except ImportError:
            self.skipTest("semantic-text-splitter is not installed")

        splitter = TextSplitter.from_callback(self.tokenizer, 10)

        chunks = splitter.chunks("The quick brown fox jumps over the lazy dog. " * 2)

        self.assertTrue(chunks)
        self.assertTrue(all(self.tokenizer(chunk) <= 10 for chunk in chunks))

    def test_markdown_splitter_uses_tokenizer_as_callback(self) -> None:
        try:
            from semantic_text_splitter import MarkdownSplitter  # type: ignore[import-not-found, ty:import-not-found]
        except ImportError:
            self.skipTest("semantic-text-splitter is not installed")

        document = (
            "# Retrieval notes\n\n"
            "Chunk sizing walks semantic levels from coarse to fine.\n\n"
            "## Why tokens\n\n"
            "A single character budget is wrong for mixed content.\n"
        )
        splitter = MarkdownSplitter.from_callback(self.tokenizer, (20, 40))

        chunks = splitter.chunks(document)

        self.assertTrue(chunks)
        self.assertTrue(all(self.tokenizer(chunk) <= 40 for chunk in chunks))

        lossless = MarkdownSplitter.from_callback(self.tokenizer, (20, 40), trim=False)
        self.assertEqual("".join(lossless.chunks(document)), document)

    def test_code_splitter_uses_tokenizer_as_callback(self) -> None:
        try:
            from semantic_text_splitter import CodeSplitter  # type: ignore[import-not-found, ty:import-not-found]
        except ImportError:
            self.skipTest("semantic-text-splitter is not installed")
        try:
            import tree_sitter_python  # type: ignore[import-not-found, ty:import-not-found]
        except ImportError:
            self.skipTest("tree-sitter-python is not installed")

        source = (
            "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
        )
        splitter = CodeSplitter.from_callback(
            tree_sitter_python.language(), self.tokenizer, (20, 40)
        )

        chunks = splitter.chunks(source)

        self.assertTrue(chunks)
        self.assertTrue(all(self.tokenizer(chunk) <= 40 for chunk in chunks))

    def test_langchain_split_text_on_tokens_uses_encode_and_decode(self) -> None:
        try:
            from langchain_text_splitters import Tokenizer, split_text_on_tokens  # type: ignore[import-not-found, ty:import-not-found]
        except ImportError:
            self.skipTest("langchain-text-splitters is not installed")

        langchain_tokenizer = Tokenizer(
            chunk_overlap=2,
            tokens_per_chunk=10,
            encode=self.tokenizer.encode,
            decode=self.tokenizer.decode,
        )
        chunks = split_text_on_tokens(
            text="The quick brown fox jumps over the lazy dog. " * 2,
            tokenizer=langchain_tokenizer,
        )

        self.assertTrue(chunks)
        self.assertTrue(all(self.tokenizer(chunk) <= 10 for chunk in chunks))

    def test_recursive_splitter_uses_token_length_function(self) -> None:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore[import-not-found, ty:import-not-found]
        except ImportError:
            self.skipTest("langchain-text-splitters is not installed")

        splitter = RecursiveCharacterTextSplitter(
            length_function=self.tokenizer, chunk_size=40, chunk_overlap=5
        )
        chunks = splitter.split_text(
            "The quick brown fox jumps over the lazy dog. " * 4
        )
        self.assertTrue(chunks)
        self.assertTrue(all(self.tokenizer(chunk) <= 40 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
