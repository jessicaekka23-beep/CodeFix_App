"""Embedding back-end selection: the cloud-safe hashing embedder and env overrides.

No Ollama, no ChromaDB, no downloads: these exercise the fallback path that keeps the
hosted demo working when nothing listens on localhost:11434.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import embeddings
import rag


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_hash_embedder_is_deterministic_and_normalised():
    emb = embeddings.HashEmbedder(dim=64)
    text = "def calculate_average(numbers):\n    return sum(numbers)"
    first, second = emb.embed(text), emb.embed(text)
    assert first == second                                  # deterministic
    assert len(first) == 64
    assert abs(sum(v * v for v in first) - 1.0) < 1e-9      # L2-normalised


def test_hash_embedder_is_case_insensitive_and_handles_empty():
    emb = embeddings.HashEmbedder(dim=64)
    assert emb.embed("Hello World") == emb.embed("hello world")
    assert emb.embed("") == [0.0] * 64


def test_hash_embedder_name_reflects_dimension():
    assert embeddings.HashEmbedder(512).name() == "codepilot-hash-512"


def test_related_text_is_closer_than_unrelated_text():
    emb = embeddings.HashEmbedder()
    base = emb.embed("the average function divides by len plus one")
    near = emb.embed("the average function divides by len plus one bug")
    far = emb.embed("unrelated text about network protocols and sockets")
    assert _cos(base, near) > _cos(base, far)


def test_env_forces_hash_backend(monkeypatch):
    monkeypatch.setenv("CODEPILOT_EMBED_BACKEND", "hash")
    assert embeddings.active_backend() == "hash"
    assert "hash" in embeddings.backend_description()


def test_embed_texts_uses_hash_backend(monkeypatch):
    monkeypatch.setenv("CODEPILOT_EMBED_BACKEND", "hash")
    vectors = embeddings.embed_texts(["alpha", "beta"])
    assert len(vectors) == 2
    assert all(len(v) == embeddings.HASH_DIM for v in vectors)
    assert embeddings.embed_one("alpha") == vectors[0]
    assert embeddings.embed_texts([]) == []


def test_embed_query_swallows_an_ollama_failure(monkeypatch):
    """With the Ollama back-end forced and the batch call failing, embed_query must
    return None (for the demo map) while embed_one raises a reportable EmbeddingError."""
    monkeypatch.setenv("CODEPILOT_EMBED_BACKEND", "ollama")
    monkeypatch.setattr(rag, "generate_embeddings", lambda texts, model=None: None)

    assert embeddings.embed_query("anything") is None
    with pytest.raises(embeddings.EmbeddingError):
        embeddings.embed_one("anything")
