"""Chunk Markdown with semantic-text-splitter sized by remote llama.cpp tokens.

``ModelTokenizer`` implements the ``Callable[[str], int]`` contract, so it
can be passed straight to ``MarkdownSplitter.from_callback``. Every size probe
is one ``POST /tokenize`` request.

    pip install model-tokenizer[semantic-text-splitter]
    export BASE_URL=http://10.0.0.10:9931
    export LLAMA_CPP_API_KEY=<token>
    export MODEL_NAME=<model>
    python semantic_splitter.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

from semantic_text_splitter import MarkdownSplitter

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

With `from_callback`, every size probe is one HTTP round trip, so counting is
the slow part and counts are not cached. The splitter still walks semantic
levels locally; only the size measurements come from the server.

## Choosing the capacity range

Pass a range rather than a hard maximum when you can afford the headroom. The
lower bound tells the splitter when a chunk is already full enough, so it stops
reaching for the next paragraph; the upper bound is the real limit that must
never be crossed. With a single number the splitter must squeeze every chunk to
the ceiling, which usually means cutting at a sentence or a word instead of at
the paragraph it would have preferred.

## Overlap

Overlap repeats the tail of one chunk at the head of the next. It costs tokens,
so it is only worth it when answers frequently straddle a boundary, for example
when a definition at the end of a section is only meaningful together with the
introduction that follows it. Zero overlap keeps storage minimal and makes the
offsets easy to reason about.

## Picking the model name

A server that hosts one model ignores the model field entirely. A router that
fronts several backends uses it to choose which one answers, so the name must
match an entry in the model list exactly. Asking a router for an unknown name
either fails with a routing error or silently lands on whichever backend happens
to be resident.

```python
tokenizer = ModelTokenizer(
    transport=LlamaCppHttpTransport(
        base_url=BASE_HTTP,
        model_name=MODEL_NAME,
        api_key=os.environ["LLAMA_CPP_API_KEY"],
    ),
)
```

## Measuring before committing

Count first on a representative sample, then decide. One or two requests per
kilobyte is a useful rule of thumb, depending on how often the splitter
re-measures trimmed and overlapping spans. If the number of requests is
unacceptable, raise the lower bound of the capacity range so fewer candidate
spans need probing.
"""


def main() -> None:

    BASE_HTTP = os.environ.get("BASE_URL", "http://10.0.0.10:9931")
    MODEL_NAME = os.environ.get("MODEL_NAME", "local-model")
    API_KEY = os.environ.get("LLAMA_CPP_API_KEY")
    CAPACITY = (128, 256)  # (desired, max) in that model tokens

    tokenizer = ModelTokenizer(
        transport=LlamaCppHttpTransport(
            base_url=BASE_HTTP,
            model_name=MODEL_NAME,
            api_key=API_KEY,
        ),
    )
    print(f"tokenizer: {tokenizer}")

    # Raw call, exactly what the splitter does behind the scenes.
    request = tokenizer.payload("Hello world!")
    print("\nPOST /tokenize payload:")
    print(f"  {json.dumps(asdict(request))}")
    print(f"  -> {tokenizer.encode('Hello world!')}")

    splitter = MarkdownSplitter.from_callback(tokenizer, CAPACITY)
    started = time.perf_counter()
    chunks = splitter.chunks(DOCUMENT)
    elapsed = time.perf_counter() - started

    print(f"\n{len(chunks)} chunks in {elapsed:.2f}s with remote token counts")
    for index, chunk in enumerate(chunks, 1):
        tokens = tokenizer(chunk)
        assert tokens <= CAPACITY[1], (index, tokens)
        heading = chunk.lstrip("#").splitlines()[0][:38]
        print(f"  {tokens:4d} tok  {len(chunk):4d} ch  #{index} {heading}")

    # Lossless variant: trim=False makes "".join(chunks) == DOCUMENT.
    lossless = MarkdownSplitter.from_callback(tokenizer, CAPACITY, trim=False)
    joined = "".join(lossless.chunks(DOCUMENT))
    print("\ntrim=False roundtrip lossless:", joined == DOCUMENT)


if __name__ == "__main__":
    main()
