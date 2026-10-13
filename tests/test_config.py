"""Tests for config module."""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    AGENT_MAX_ITERATIONS,
    AVAILABLE_MODELS,
    CLOUD_GROQ_MODELS,
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    EXCLUDED_DIRS,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    SUPPORTED_EXTENSIONS,
    check_ollama_available,
    get_available_ollama_models,
    validate_workspace_path,
)


def test_available_models():
    """AVAILABLE_MODELS is a dict with expected keys."""
    assert isinstance(AVAILABLE_MODELS, dict)
    assert len(AVAILABLE_MODELS) > 0
    assert "qwen2.5-coder:7b (Recommended)" in AVAILABLE_MODELS
    assert AVAILABLE_MODELS["qwen2.5-coder:7b (Recommended)"] == "qwen2.5-coder:7b"


def test_default_model():
    """DEFAULT_MODEL is a string."""
    assert isinstance(DEFAULT_MODEL, str)
    assert len(DEFAULT_MODEL) > 0
    assert DEFAULT_MODEL == "qwen2.5-coder:7b"


def test_embedding_model():
    """EMBEDDING_MODEL is nomic-embed-text."""
    assert EMBEDDING_MODEL == "nomic-embed-text"


def test_agent_max_iterations():
    """AGENT_MAX_ITERATIONS > 0."""
    assert isinstance(AGENT_MAX_ITERATIONS, int)
    assert AGENT_MAX_ITERATIONS > 0


def test_rag_chunk_settings():
    """RAG_CHUNK_SIZE > 0 and RAG_CHUNK_OVERLAP < RAG_CHUNK_SIZE."""
    assert isinstance(RAG_CHUNK_SIZE, int)
    assert isinstance(RAG_CHUNK_OVERLAP, int)
    assert RAG_CHUNK_SIZE > 0
    assert RAG_CHUNK_OVERLAP < RAG_CHUNK_SIZE


def test_supported_extensions():
    """SUPPORTED_EXTENSIONS contains .py and .md."""
    assert ".py" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS


def test_excluded_dirs():
    """EXCLUDED_DIRS contains __pycache__ and .git."""
    assert "__pycache__" in EXCLUDED_DIRS
    assert ".git" in EXCLUDED_DIRS


def test_validate_workspace_path_valid(tmp_path):
    """validate_workspace_path with valid directory returns (True, ...)."""
    valid_dir = tmp_path / "valid_workspace"
    valid_dir.mkdir()
    is_valid, msg = validate_workspace_path(str(valid_dir))
    assert is_valid is True
    assert "valid" in msg.lower()


def test_validate_workspace_path_nonexistent(tmp_path):
    """validate_workspace_path with nonexistent path returns (False, ...)."""
    nonexistent = tmp_path / "does_not_exist"
    is_valid, msg = validate_workspace_path(str(nonexistent))
    assert is_valid is False
    assert "does not exist" in msg.lower()


def test_validate_workspace_path_file(tmp_path):
    """validate_workspace_path with a file (not directory) returns (False, ...)."""
    file_path = tmp_path / "regular_file.txt"
    file_path.write_text("hello", encoding="utf-8")
    is_valid, msg = validate_workspace_path(str(file_path))
    assert is_valid is False
    assert "not a directory" in msg.lower()


def test_validate_workspace_path_empty():
    """validate_workspace_path with empty string returns (False, ...)."""
    is_valid, msg = validate_workspace_path("")
    assert is_valid is False
    assert "cannot be empty" in msg.lower()


def test_check_ollama_available_returns_bool():
    """check_ollama_available returns bool."""
    result = check_ollama_available()
    assert isinstance(result, bool)


@pytest.mark.integration
def test_ollama_live_service():
    """Integration test checking Ollama server and model retrieval."""
    is_up = check_ollama_available()
    if is_up:
        models = get_available_ollama_models()
        assert isinstance(models, list)
    else:
        pytest.skip("Ollama service is not reachable.")


def test_groq_model_catalog_uses_current_gpt_oss_ids():
    assert CLOUD_GROQ_MODELS["GPT-OSS 20B (Recommended — faster)"] == "openai/gpt-oss-20b"
    assert CLOUD_GROQ_MODELS["GPT-OSS 120B (Advanced reasoning)"] == "openai/gpt-oss-120b"
    assert "llama-3.3-70b-versatile" not in CLOUD_GROQ_MODELS.values()
    assert "llama-3.1-8b-instant" not in CLOUD_GROQ_MODELS.values()
