"""Tests for RAG module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag import (
    build_rag_context,
    chunk_all_documents,
    chunk_document,
    generate_embedding,
    load_documents,
    retrieve_relevant_chunks,
)


@pytest.fixture
def rag_workspace(tmp_path):
    """Fixture providing workspace with python files, markdown, pycache, and large file."""
    ws = tmp_path / "rag_workspace"
    ws.mkdir()

    # Valid Python file
    py_file = ws / "calc.py"
    py_file.write_text(
        "def add(x: int, y: int) -> int:\n"
        "    return x + y\n\n"
        "def sub(x: int, y: int) -> int:\n"
        "    return x - y\n",
        encoding="utf-8"
    )

    # Valid Markdown file
    md_file = ws / "README.md"
    md_file.write_text("# Calc Library\nDocumentation for math helpers.\n", encoding="utf-8")

    # __pycache__ directory that should be excluded
    pycache = ws / "__pycache__"
    pycache.mkdir()
    (pycache / "calc.cpython-314.pyc").write_bytes(b"\x00\x00\x00\x00")

    # Large file (> 1MB) that should be skipped
    large_file = ws / "huge.py"
    large_file.write_bytes(b"#" * (1024 * 1024 + 500))

    return ws


# ── 1. load_documents ────────────────────────────────────

def test_load_documents_success(rag_workspace):
    """Test loading documents from temp directory with .py and .md files."""
    docs = load_documents(str(rag_workspace))
    file_names = [Path(d["path"]).name for d in docs]

    assert "calc.py" in file_names
    assert "README.md" in file_names


def test_load_documents_excludes_pycache(rag_workspace):
    """Test that files in __pycache__ are excluded."""
    docs = load_documents(str(rag_workspace))
    file_names = [Path(d["path"]).name for d in docs]

    assert not any("pyc" in name for name in file_names)
    assert not any("__pycache__" in d["path"] for d in docs)


def test_load_documents_skips_large_files(rag_workspace):
    """Test that files exceeding 1MB are skipped."""
    docs = load_documents(str(rag_workspace))
    file_names = [Path(d["path"]).name for d in docs]

    assert "huge.py" not in file_names


def test_load_documents_nonexistent_directory(tmp_path):
    """Test load_documents handles nonexistent paths safely."""
    docs = load_documents(str(tmp_path / "missing_dir"))
    assert docs == []


# ── 2. chunk_document ────────────────────────────────────

def test_chunk_document_basic():
    """Test chunking a document dict preserves metadata and segments text."""
    doc = {
        "path": "/workspace/main.py",
        "content": (
            "line 1: import os\n"
            "line 2: import sys\n"
            "line 3: def hello():\n"
            "line 4:     return 'hello'\n"
            "line 5: def bye():\n"
            "line 6:     return 'bye'\n"
        ),
        "extension": ".py"
    }

    chunks = chunk_document(doc, chunk_size=50, chunk_overlap=20)
    assert len(chunks) > 1

    first = chunks[0]
    assert "text" in first
    assert "metadata" in first
    assert first["metadata"]["source"] == "/workspace/main.py"
    assert first["metadata"]["extension"] == ".py"
    assert first["metadata"]["start_line"] == 1
    assert first["metadata"]["end_line"] >= 1
    assert "chunk_0" in first["metadata"]["chunk_id"]


def test_chunk_document_overlap_behavior():
    """Test that consecutive chunks share overlapping content when overlap > 0."""
    lines = [f"This is test sentence number {i} for verifying chunk overlap.\n" for i in range(20)]
    doc = {
        "path": "overlap_test.py",
        "content": "".join(lines),
        "extension": ".py"
    }

    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=60)
    assert len(chunks) >= 2

    # Check that chunk 1 starts before or at chunk 0 end
    assert chunks[1]["metadata"]["start_line"] <= chunks[0]["metadata"]["end_line"]


def test_chunk_document_empty():
    """Test chunking an empty document returns empty list."""
    empty_doc = {"path": "empty.py", "content": "", "extension": ".py"}
    chunks = chunk_document(empty_doc)
    assert chunks == []


# ── 3. chunk_all_documents ───────────────────────────────

def test_chunk_all_documents():
    """Test chunking multiple documents in a collection."""
    docs = [
        {"path": "a.py", "content": "print('file a')\n", "extension": ".py"},
        {"path": "b.py", "content": "print('file b')\n", "extension": ".py"}
    ]
    chunks = chunk_all_documents(docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) == 2
    sources = [c["metadata"]["source"] for c in chunks]
    assert "a.py" in sources
    assert "b.py" in sources


# ── 4. build_rag_context ─────────────────────────────────

@pytest.mark.integration
def test_build_rag_context_mock():
    """Test build_rag_context returns formatted citations and text with mocked retriever."""
    mock_chunks = [
        {
            "text": "def calculate(a, b):\n    return a + b",
            "metadata": {
                "source": "math_utils.py",
                "start_line": 10,
                "end_line": 12,
            },
            "distance": 0.15,
        }
    ]

    with patch("rag.retrieve_relevant_chunks", return_value=mock_chunks):
        mock_collection = MagicMock()
        context, sources = build_rag_context("calculate sum", mock_collection)

        assert "--- Source: math_utils.py (lines 10-12) ---" in context
        assert "def calculate(a, b):" in context
        assert len(sources) == 1
        assert sources[0]["source"] == "math_utils.py"


@pytest.mark.integration
def test_build_rag_context_empty():
    """Test build_rag_context returns empty context and sources when no chunks match."""
    with patch("rag.retrieve_relevant_chunks", return_value=[]):
        mock_collection = MagicMock()
        context, sources = build_rag_context("unknown query", mock_collection)
        assert context == ""
        assert sources == []


@pytest.mark.integration
def test_generate_embedding_live():
    """Live integration test with Ollama nomic-embed-text if service is up."""
    from config import check_ollama_available
    if not check_ollama_available():
        pytest.skip("Ollama is not running.")

    emb = generate_embedding("def test(): pass")
    if emb is None:
        pytest.skip("Embedding model may not be pulled.")
    assert isinstance(emb, list)
    assert len(emb) > 0
    assert isinstance(emb[0], float)
