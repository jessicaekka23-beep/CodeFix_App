"""Embedding back-end selection for the RAG index.

The index normally embeds with Ollama (``nomic-embed-text``), which keeps every
byte on the machine. When Ollama is not installed - a hosted Streamlit app, for
example - the app falls back to a dependency-free hashing embedder so retrieval
still works with no extra services and no model downloads.

Force a back-end with the ``CODEPILOT_EMBED_BACKEND`` environment variable
(``ollama`` or ``hash``). Vectors from different back-ends are not comparable, so
re-index after switching (the UIs clear the collection before indexing).
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time

logger = logging.getLogger(__name__)

HASH_DIM = 512

# check_ollama_available() pings localhost with a short timeout; cache the answer
# so a hosted app (where nothing listens on 11434) does not pay for it on every rerun.
_OLLAMA_CACHE = {"ok": None, "at": 0.0}
_OLLAMA_TTL = 10.0

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]+")


class EmbeddingError(RuntimeError):
    """Raised when text cannot be embedded, so indexing can report the failure."""


def _ollama_available() -> bool:
    now = time.time()
    cached = _OLLAMA_CACHE["ok"]
    if cached is not None and now - _OLLAMA_CACHE["at"] < _OLLAMA_TTL:
        return cached
    try:
        from config import check_ollama_available
        ok = bool(check_ollama_available())
    except Exception:
        ok = False
    _OLLAMA_CACHE.update(ok=ok, at=now)
    return ok


def active_backend() -> str:
    """Return ``"ollama"`` when the local model is reachable, otherwise ``"hash"``."""
    forced = os.environ.get("CODEPILOT_EMBED_BACKEND", "").strip().lower()
    if forced in ("hash", "hashing"):
        return "hash"
    if forced == "ollama":
        return "ollama"
    return "ollama" if _ollama_available() else "hash"


def backend_description() -> str:
    """Human-readable name of the active back-end, for captions and tooltips."""
    if active_backend() == "ollama":
        from config import EMBEDDING_MODEL
        return f"Ollama · {EMBEDDING_MODEL} (local)"
    return "built-in hashing embedder (Ollama not detected)"


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. Always available, no downloads.

    Each token is hashed into one of ``dim`` buckets with a sign, then the vector
    is L2-normalised. It is not as sharp as a trained model, but it needs nothing
    installed and gives useful keyword-overlap retrieval out of the box.
    """

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim

    def name(self) -> str:
        return f"codepilot-hash-{self.dim}"

    def embed(self, text: str) -> list[float]:
        return self._embed_one(text)

    def embed_many(self, texts) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text if isinstance(text, str) else str(text)):
            digest = hashlib.md5(token.lower().encode("utf-8")).digest()
            value = int.from_bytes(digest[:8], "big")
            vector[value % self.dim] += 1.0 if (value >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


def embed_texts(texts) -> list[list[float]]:
    """Embed many chunks with the active back-end. Raises EmbeddingError on failure."""
    if texts is None:
        return []
    texts = [t if isinstance(t, str) else str(t) for t in texts]
    if not texts:
        return []
    if active_backend() == "ollama":
        from rag import generate_embeddings
        vectors = generate_embeddings(texts)
        if not vectors or len(vectors) != len(texts) or any(v is None for v in vectors):
            raise EmbeddingError(
                "Ollama could not embed the code. Is 'ollama serve' running with "
                "`nomic-embed-text` pulled? (Set CODEPILOT_EMBED_BACKEND=hash to use the "
                "built-in embedder instead.)")
        return [list(map(float, v)) for v in vectors]
    return HashEmbedder().embed_many(texts)


def embed_one(text: str) -> list[float]:
    """Embed a single string with the active back-end. Raises EmbeddingError on failure."""
    return embed_texts([text])[0]


def embed_query(text: str):
    """Best-effort single embedding for the demo views; returns None instead of raising."""
    try:
        return embed_one(text)
    except EmbeddingError as exc:
        logger.warning("Query embedding failed: %s", exc)
        return None
