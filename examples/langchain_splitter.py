"""LangChain text splitting sized by a remote llama.cpp / llama-swap server.

Shows both integration styles:

1. ``RecursiveCharacterTextSplitter(length_function=tokenizer)`` — character
   separators with token-accurate measurements.
2. ``split_text_on_tokens`` — fixed token windows, using the server's
   tokenizer and detokenizer.

    pip install model-tokenizer[langchain-text-splitters]
    export BASE_URL=http://10.0.0.10:9931
    export LLAMA_CPP_API_KEY=<token>
    export MODEL_NAME=<model>
    python langchain_splitter.py
"""

from __future__ import annotations

import os
import time

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Tokenizer,
    split_text_on_tokens,
)

from modeltokenizer import LlamaCppHttpTransport, ModelTokenizer

DOCUMENT = """\
# Retrieval notes

## Chunk sizing

The splitter walks semantic levels from coarse to fine: headings, then
paragraphs, then list items, then sentences, then words. It keeps the largest
unit that still fits, so a chunk boundary usually lands where a reader would
have put one. Character counts approximate the model's budget; token counts
mean the budget exactly.

## Why tokens

A single character budget is wrong for mixed content. English prose runs at
roughly four characters per token, source code and JSON at two or three, and
Cyrillic or CJK at one or two. Measuring in tokens removes the guesswork and
keeps every chunk inside the context window.

- Bullet one: uniform information per chunk.
- Bullet two: no silent truncation at embed time.
- Bullet three: chunk counts become comparable across languages.

## Operational cost

Every size probe is one HTTP round trip, so counting is the slow part. Counts
are not cached, so each probe costs a request even when the substring repeats.

## Choosing the capacity range

Pass a range rather than a hard maximum when you can afford the headroom. The
lower bound tells the splitter when a chunk is already full enough, so it stops
reaching for the next paragraph; the upper bound is the real limit that must
never be crossed.

```python
tokenizer = ModelTokenizer(
    transport=LlamaCppHttpTransport(
        base_url=BASE_HTTP,
        model_name=MODEL_NAME,
        api_key=os.environ["LLAMA_CPP_API_KEY"],
    ),
)
```
"""


def report(
    name: str, chunks: list[str], tokenizer: ModelTokenizer, elapsed: float
) -> None:
    print(f"\n{name}: {len(chunks)} chunks in {elapsed:.2f}s")
    for index, chunk in enumerate(chunks, 1):
        heading = chunk.lstrip("#").splitlines()[0][:38]
        print(f"  {tokenizer(chunk):4d} tok  {len(chunk):4d} ch  #{index} {heading}")


def main() -> None:
    BASE_HTTP = os.environ.get("BASE_URL", "http://10.0.0.10:9931")
    MODEL_NAME = os.environ.get("MODEL_NAME", "local-model")
    API_KEY = os.environ.get("LLAMA_CPP_API_KEY")
    CHUNK_SIZE = 128
    CHUNK_OVERLAP = 16

    tokenizer = ModelTokenizer(
        transport=LlamaCppHttpTransport(
            base_url=BASE_HTTP,
            model_name=MODEL_NAME,
            api_key=API_KEY,
        ),
    )
    print(f"tokenizer: {tokenizer}")

    # 1. Character-recursive splitter, measured in model tokens.
    recursive = RecursiveCharacterTextSplitter(
        length_function=tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    started = time.perf_counter()
    recursive_chunks = recursive.split_text(DOCUMENT)
    report(
        "RecursiveCharacterTextSplitter",
        recursive_chunks,
        tokenizer,
        time.perf_counter() - started,
    )

    # 2. Fixed-size token windows, decoded by the server (/detokenize).
    chunk_tokenizer = Tokenizer(
        chunk_overlap=CHUNK_OVERLAP,
        tokens_per_chunk=CHUNK_SIZE,
        encode=tokenizer.encode,
        decode=tokenizer.decode,
    )
    started = time.perf_counter()
    token_chunks = split_text_on_tokens(text=DOCUMENT, tokenizer=chunk_tokenizer)
    report(
        "split_text_on_tokens",
        token_chunks,
        tokenizer,
        time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
