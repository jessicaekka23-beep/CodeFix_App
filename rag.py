"""
RAG (Retrieval-Augmented Generation) pipeline for CodePilot Local.

Handles document loading, chunking, embedding generation with Ollama,
and vector similarity retrieval using ChromaDB.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests

try:
    from config import (
        EMBEDDING_MODEL,
        EXCLUDED_DIRS,
        OLLAMA_EMBED_URL,
        RAG_CHUNK_OVERLAP,
        RAG_CHUNK_SIZE,
        RAG_TOP_K,
        SUPPORTED_EXTENSIONS,
    )
except ImportError:
    OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
    EMBEDDING_MODEL = "nomic-embed-text"
    RAG_CHUNK_SIZE = 800
    RAG_CHUNK_OVERLAP = 200
    RAG_TOP_K = 5
    SUPPORTED_EXTENSIONS = {
        ".py",
        ".md",
        ".txt",
        ".rst",
        ".cfg",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".ini",
    }
    EXCLUDED_DIRS = {
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".env",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        "chroma_db",
        ".idea",
        ".vscode",
    }

logger = logging.getLogger(__name__)


def _should_skip_directory(dir_name: str, excluded_dirs: set[str]) -> bool:
    """Check if directory name matches excluded list or patterns."""
    if dir_name in excluded_dirs:
        return True
    if dir_name.endswith(".egg-info"):
        return True
    return False


def load_documents(workspace_path: str) -> list[dict[str, str]]:
    """
    Recursively scan workspace directory and load supported text/code files.

    Args:
        workspace_path: Root directory path of workspace to scan.

    Returns:
        List of dicts: {"path": str, "content": str, "extension": str}
    """
    workspace = Path(workspace_path)
    if not workspace.exists() or not workspace.is_dir():
        logger.warning(f"Workspace path does not exist or is not a directory: {workspace_path}")
        return []

    documents: list[dict[str, str]] = []
    excluded = set(EXCLUDED_DIRS) if isinstance(EXCLUDED_DIRS, (set, list, tuple)) else set()
    supported = set(SUPPORTED_EXTENSIONS) if isinstance(SUPPORTED_EXTENSIONS, (set, list, tuple)) else set()
    max_file_size = 1024 * 1024  # 1MB limit

    for root, dirs, files in os.walk(workspace):
        # Filter directories in-place to avoid descending into skipped paths
        dirs[:] = [d for d in dirs if not _should_skip_directory(d, excluded)]

        for file_name in files:
            file_path = Path(root) / file_name
            ext = file_path.suffix.lower()

            if ext not in supported:
                continue

            try:
                stat = file_path.stat()
                if stat.st_size > max_file_size:
                    logger.debug(f"Skipping file larger than 1MB: {file_path}")
                    continue
            except OSError as e:
                logger.warning(f"Failed to check stats for {file_path}: {e}")
                continue

            content: str | None = None
            for encoding in ("utf-8", "latin-1"):
                try:
                    content = file_path.read_text(encoding=encoding)
                    break
                except (UnicodeDecodeError, OSError):
                    continue

            if content is None:
                logger.warning(f"Could not read file {file_path} with supported encodings (utf-8, latin-1).")
                continue

            documents.append({
                "path": str(file_path.resolve()),
                "content": content,
                "extension": ext,
            })

    return documents


def chunk_document(
    doc: dict[str, Any],
    chunk_size: int = RAG_CHUNK_SIZE,
    chunk_overlap: int = RAG_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split document content into line-based overlapping chunks to preserve code structure.

    Args:
        doc: Document dictionary containing 'path', 'content', and 'extension'.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Target character overlap between consecutive chunks.

    Returns:
        List of chunk dicts:
            {
                "text": str,
                "metadata": {
                    "source": str,
                    "chunk_id": str,
                    "start_line": int,
                    "end_line": int,
                    "extension": str
                }
            }
    """
    content = doc.get("content", "")
    file_path = doc.get("path", "")
    extension = doc.get("extension", "")

    if not content:
        return []

    lines = content.splitlines(keepends=True)
    if not lines:
        return []

    chunks: list[dict[str, Any]] = []
    total_lines = len(lines)
    start_idx = 0
    chunk_idx = 0

    while start_idx < total_lines:
        curr_chars = 0
        end_idx = start_idx

        while end_idx < total_lines:
            line_len = len(lines[end_idx])
            if curr_chars + line_len > chunk_size and end_idx > start_idx:
                break
            curr_chars += line_len
            end_idx += 1

        chunk_text = "".join(lines[start_idx:end_idx])
        start_line = start_idx + 1
        end_line = end_idx

        chunk_id = f"{file_path}::chunk_{chunk_idx}"
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": file_path,
                "chunk_id": chunk_id,
                "start_line": start_line,
                "end_line": end_line,
                "extension": extension,
            },
        })
        chunk_idx += 1

        if end_idx >= total_lines:
            break

        # Calculate next start index based on chunk_overlap
        next_start = end_idx
        overlap_chars = 0
        while next_start > start_idx + 1:
            prev_line_len = len(lines[next_start - 1])
            if overlap_chars + prev_line_len > chunk_overlap:
                break
            overlap_chars += prev_line_len
            next_start -= 1

        # Guarantee progress
        if next_start <= start_idx:
            start_idx = end_idx
        else:
            start_idx = next_start

    return chunks


def chunk_all_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = RAG_CHUNK_SIZE,
    chunk_overlap: int = RAG_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Apply chunk_document across a collection of documents and return a flattened list.

    Args:
        documents: List of document dicts.
        chunk_size: Character length limit per chunk.
        chunk_overlap: Overlapping character count between chunks.

    Returns:
        Flattened list of all chunk dicts.
    """
    all_chunks: list[dict[str, Any]] = []
    for doc in documents:
        doc_chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(doc_chunks)
    return all_chunks


def generate_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float] | None:
    """
    Generate vector embedding using Ollama API.

    Args:
        text: Input string to embed.
        model: Embedding model name.

    Returns:
        Embedding vector as list of floats, or None on failure.
    """
    if not text:
        return None

    payload = {
        "model": model,
        "input": text,
    }

    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # Support both Ollama /api/embed (list of embeddings) and legacy /api/embeddings
        if "embeddings" in data and isinstance(data["embeddings"], list) and len(data["embeddings"]) > 0:
            return data["embeddings"][0]
        if "embedding" in data and isinstance(data["embedding"], list):
            return data["embedding"]

        logger.warning(f"Unexpected response format from Ollama embedding API: {data}")
        return None
    except requests.RequestException as e:
        logger.warning(f"Failed to generate embedding for model '{model}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating embedding: {e}")
        return None


def generate_embeddings(texts, model: str = EMBEDDING_MODEL) -> list[list[float]] | None:
    """Embed a batch of texts with Ollama in a single request.

    Ollama's ``/api/embed`` accepts a list of inputs and returns one vector per
    input, which is much faster than one request per chunk. Returns ``None`` on any
    failure (or an unexpected shape) so callers can fall back or report the error.

    Args:
        texts: Iterable of strings to embed.
        model: Ollama embedding model name.

    Returns:
        A list with one embedding vector per input, or ``None`` on failure.
    """
    inputs = [t if isinstance(t, str) else str(t) for t in (texts or [])]
    if not inputs:
        return []

    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": model, "input": inputs},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and len(embeddings) == len(inputs):
            return embeddings

        # Legacy /api/embeddings endpoint returns a single vector.
        if isinstance(data.get("embedding"), list) and len(inputs) == 1:
            return [data["embedding"]]

        logger.warning(f"Unexpected response format from Ollama embedding API: {data}")
        return None
    except requests.RequestException as e:
        logger.warning(f"Failed to generate embeddings for model '{model}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating embeddings: {e}")
        return None


def retrieve_relevant_chunks(query: str, collection: Any, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
    """
    Query ChromaDB collection using embedding of query text.

    Args:
        query: Search query string.
        collection: ChromaDB collection instance.
        top_k: Number of most relevant results to retrieve.

    Returns:
        List of dicts: {"text": str, "metadata": dict, "distance": float} sorted by relevance.
    """
    if not query or collection is None:
        return []

    try:
        # Check if collection has documents
        if hasattr(collection, "count") and collection.count() == 0:
            return []
    except Exception as e:
        logger.debug(f"Could not count collection items: {e}")

    from embeddings import embed_query
    query_embedding = embed_query(query)
    if query_embedding is None:
        logger.warning("Could not generate query embedding for retrieval.")
        return []

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
    except Exception as e:
        logger.error(f"Error querying ChromaDB collection: {e}")
        return []

    if not results:
        return []

    documents_list = results.get("documents", [[]])
    metadatas_list = results.get("metadatas", [[]])
    distances_list = results.get("distances", [[]])

    docs = documents_list[0] if documents_list else []
    metas = metadatas_list[0] if metadatas_list else []
    dists = distances_list[0] if distances_list else []

    retrieved: list[dict[str, Any]] = []
    for i in range(len(docs)):
        retrieved.append({
            "text": docs[i] if i < len(docs) else "",
            "metadata": metas[i] if i < len(metas) and metas[i] is not None else {},
            "distance": float(dists[i]) if i < len(dists) and dists[i] is not None else 0.0,
        })

    # Sort by distance (lower distance = more relevant in ChromaDB metric spaces)
    retrieved.sort(key=lambda x: x["distance"])
    return retrieved


def build_rag_context(query: str, collection: Any, top_k: int = RAG_TOP_K) -> tuple[str, list[dict[str, Any]]]:
    """
    Retrieve relevant chunks and construct formatted context string with citations.

    Args:
        query: Search query string.
        collection: ChromaDB collection instance.
        top_k: Number of chunks to retrieve.

    Returns:
        Tuple of (context_string, sources_list).
    """
    chunks = retrieve_relevant_chunks(query, collection, top_k=top_k)
    if not chunks:
        return "", []

    context_blocks: list[str] = []
    sources: list[dict[str, Any]] = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        text = chunk.get("text", "")
        source_path = metadata.get("source", "unknown")
        start_line = metadata.get("start_line", 1)
        end_line = metadata.get("end_line", 1)

        header = f"--- Source: {source_path} (lines {start_line}-{end_line}) ---"
        context_blocks.append(f"{header}\n{text}\n")
        sources.append(metadata)

    full_context = "\n".join(context_blocks)
    return full_context, sources
