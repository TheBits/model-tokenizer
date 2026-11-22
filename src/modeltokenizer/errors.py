__all__ = ["TransportError"]


class TransportError(RuntimeError):
    """Raised when a remote tokenizer server cannot produce a tokenization."""
