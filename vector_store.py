"""
ChromaDB vector store management for CodePilot Local.

Handles persistence, workspace code indexing, reference document indexing,
similarity search, and debugging notes. Vectors come from the selected embedding
back-end (Ollama when reachable, otherwise the built-in hashing embedder) and are
always passed to Chroma explicitly, so the store never depends on a registered
embedding function and never triggers a model download.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any

import chromadb

from config import CHROMA_PERSIST_DIR
import embeddings
import rag

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages ChromaDB collections for CodePilot Local."""

    WORKSPACE_COLLECTION = "workspace_code"
    REFERENCE_COLLECTION = "reference_docs"

    def __init__(self, persist_dir: str | None = None) -> None:
        """
        Initialize ChromaDB PersistentClient and vector store logger.

        Args:
            persist_dir: Optional path for ChromaDB persistent storage.
                         Defaults to CHROMA_PERSIST_DIR from config.
        """
        self.persist_dir = str(persist_dir) if persist_dir is not None else str(CHROMA_PERSIST_DIR)
        self.logger = logging.getLogger(__name__)

        try:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.logger.info(f"Initialized ChromaDB PersistentClient at {self.persist_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB PersistentClient at {self.persist_dir}: {e}")
            raise

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """
        Get or create a ChromaDB collection.

        Embedding is handled explicitly at every add()/query() call, so no embedding
        function is registered. Passing ``embedding_function=None`` stops Chroma from
        attaching its default ONNX model (which would otherwise try to download on a
        machine with no Ollama/models).

        Args:
            name: Name of the collection.

        Returns:
            ChromaDB Collection instance.
        """
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=None,
            )
            return collection
        except Exception as e:
            self.logger.error(f"Failed to get or create collection '{name}': {e}")
            raise

    def index_workspace(self, workspace_path: str, collection_name: str | None = None) -> dict[str, int]:
        """
        Load, chunk, and index all supported files in a workspace directory into ChromaDB.

        Args:
            workspace_path: Root directory path of workspace to index.
            collection_name: Target collection name (defaults to WORKSPACE_COLLECTION).

        Returns:
            Dictionary with statistics:
            {"total_documents": int, "total_chunks": int, "indexed": int, "skipped_duplicates": int, "errors": int}
        """
        target_collection = collection_name or self.WORKSPACE_COLLECTION
        stats: dict[str, int] = {
            "total_documents": 0,
            "total_chunks": 0,
            "indexed": 0,
            "skipped_duplicates": 0,
            "errors": 0,
        }

        try:
            collection = self.get_or_create_collection(target_collection)
        except Exception as e:
            self.logger.error(f"Could not access collection '{target_collection}' for indexing: {e}")
            stats["errors"] += 1
            return stats

        self.logger.info(f"Loading workspace documents from: {workspace_path}")
        documents = rag.load_documents(workspace_path)
        stats["total_documents"] = len(documents)

        if not documents:
            self.logger.warning(f"No documents found to index in workspace: {workspace_path}")
            return stats

        chunks = rag.chunk_all_documents(documents)
        stats["total_chunks"] = len(chunks)

        if not chunks:
            self.logger.warning(f"No chunks produced from documents in workspace: {workspace_path}")
            return stats

        batch_size = 50
        seen_in_run: set[str] = set()

        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start:batch_start + batch_size]
            batch_ids: list[str] = []
            batch_documents: list[str] = []
            batch_metadatas: list[dict[str, Any]] = []

            for chunk in batch_chunks:
                text = chunk.get("text", "")
                raw_metadata = chunk.get("metadata", {})
                source = str(raw_metadata.get("source", ""))

                # Compute content hash (md5 of text + source)
                content_key = f"{source}:{text}"
                doc_id = hashlib.md5(content_key.encode("utf-8", errors="ignore")).hexdigest()

                if doc_id in seen_in_run:
                    stats["skipped_duplicates"] += 1
                    continue
                seen_in_run.add(doc_id)

                # Ensure metadata values are primitive types for ChromaDB compatibility
                clean_metadata: dict[str, str | int | float | bool] = {}
                for k, v in raw_metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_metadata[k] = v
                    else:
                        clean_metadata[k] = str(v)

                batch_ids.append(doc_id)
                batch_documents.append(text)
                batch_metadatas.append(clean_metadata)

            if not batch_ids:
                continue

            # Check ChromaDB for existing duplicates to skip re-indexing
            try:
                existing = collection.get(ids=batch_ids)
                existing_ids = set(existing.get("ids", [])) if existing else set()
            except Exception as e:
                self.logger.warning(f"Failed to check existing IDs in collection: {e}")
                existing_ids = set()

            new_ids: list[str] = []
            new_docs: list[str] = []
            new_metas: list[dict[str, Any]] = []

            for doc_id, text, meta in zip(batch_ids, batch_documents, batch_metadatas):
                if doc_id in existing_ids:
                    stats["skipped_duplicates"] += 1
                else:
                    new_ids.append(doc_id)
                    new_docs.append(text)
                    new_metas.append(meta)

            if not new_ids:
                continue

            # Embed the new chunks with the active back-end (Ollama or hashing).
            # A failure here is reported as an error instead of silently storing
            # zero vectors, which used to make RAG return meaningless matches.
            try:
                vectors = embeddings.embed_texts(new_docs)
            except embeddings.EmbeddingError as e:
                self.logger.error(f"Could not embed {len(new_ids)} chunk(s): {e}")
                stats["errors"] += len(new_ids)
                continue

            # Add new chunks in batch
            try:
                collection.add(
                    ids=new_ids,
                    documents=new_docs,
                    embeddings=vectors,
                    metadatas=new_metas,
                )
                stats["indexed"] += len(new_ids)
            except Exception as e:
                self.logger.error(f"Error adding batch of {len(new_ids)} chunks to collection: {e}")
                stats["errors"] += len(new_ids)

        self.logger.info(
            f"Workspace indexing complete. Indexed: {stats['indexed']}, "
            f"Duplicates skipped: {stats['skipped_duplicates']}, Errors: {stats['errors']}"
        )
        return stats

    def index_reference_doc(self, file_path: str, collection_name: str | None = None) -> dict[str, int]:
        """
        Index a single reference document into the reference collection.

        Args:
            file_path: Path to the reference document file.
            collection_name: Target collection name (defaults to REFERENCE_COLLECTION).

        Returns:
            Dictionary with statistics:
            {"total_documents": int, "total_chunks": int, "indexed": int, "skipped_duplicates": int, "errors": int}
        """
        target_collection = collection_name or self.REFERENCE_COLLECTION
        stats: dict[str, int] = {
            "total_documents": 0,
            "total_chunks": 0,
            "indexed": 0,
            "skipped_duplicates": 0,
            "errors": 0,
        }

        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            self.logger.error(f"Reference file not found: {file_path}")
            stats["errors"] += 1
            return stats

        content: str | None = None
        for encoding in ("utf-8", "latin-1"):
            try:
                content = path.read_text(encoding=encoding)
                break
            except (UnicodeDecodeError, OSError):
                continue

        if content is None:
            self.logger.error(f"Failed to read reference doc content: {file_path}")
            stats["errors"] += 1
            return stats

        doc = {
            "path": str(path),
            "content": content,
            "extension": path.suffix.lower(),
        }
        stats["total_documents"] = 1

        chunks = rag.chunk_document(doc)
        stats["total_chunks"] = len(chunks)

        if not chunks:
            return stats

        try:
            collection = self.get_or_create_collection(target_collection)
        except Exception as e:
            self.logger.error(f"Could not access collection '{target_collection}': {e}")
            stats["errors"] += len(chunks)
            return stats

        batch_ids: list[str] = []
        batch_docs: list[str] = []
        batch_metas: list[dict[str, Any]] = []

        for chunk in chunks:
            text = chunk.get("text", "")
            raw_metadata = chunk.get("metadata", {})
            source = str(raw_metadata.get("source", str(path)))

            content_key = f"{source}:{text}"
            doc_id = hashlib.md5(content_key.encode("utf-8", errors="ignore")).hexdigest()

            clean_metadata: dict[str, str | int | float | bool] = {}
            for k, v in raw_metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                else:
                    clean_metadata[k] = str(v)

            batch_ids.append(doc_id)
            batch_docs.append(text)
            batch_metas.append(clean_metadata)

        try:
            existing = collection.get(ids=batch_ids)
            existing_ids = set(existing.get("ids", [])) if existing else set()
        except Exception as e:
            self.logger.warning(f"Failed to check existing IDs: {e}")
            existing_ids = set()

        new_ids: list[str] = []
        new_docs: list[str] = []
        new_metas: list[dict[str, Any]] = []

        for doc_id, text, meta in zip(batch_ids, batch_docs, batch_metas):
            if doc_id in existing_ids:
                stats["skipped_duplicates"] += 1
            else:
                new_ids.append(doc_id)
                new_docs.append(text)
                new_metas.append(meta)

        if new_ids:
            try:
                vectors = embeddings.embed_texts(new_docs)
            except embeddings.EmbeddingError as e:
                self.logger.error(f"Could not embed {len(new_ids)} reference chunk(s): {e}")
                stats["errors"] += len(new_ids)
                return stats
            try:
                collection.add(
                    ids=new_ids,
                    documents=new_docs,
                    embeddings=vectors,
                    metadatas=new_metas,
                )
                stats["indexed"] += len(new_ids)
            except Exception as e:
                self.logger.error(f"Failed to add reference chunks to collection: {e}")
                stats["errors"] += len(new_ids)

        return stats

    def search(
        self,
        query: str,
        collection_name: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Generate embedding for query and retrieve top_k most similar chunks.

        Args:
            query: Natural language or code search query string.
            collection_name: Collection to search (defaults to WORKSPACE_COLLECTION).
            top_k: Maximum number of relevant chunks to return.

        Returns:
            List of dicts: [{"text": str, "metadata": dict, "distance": float}]
        """
        if not query or not query.strip():
            return []

        target_collection = collection_name or self.WORKSPACE_COLLECTION

        try:
            collection = self.get_or_create_collection(target_collection)
        except Exception as e:
            self.logger.error(f"Error accessing collection '{target_collection}' for search: {e}")
            return []

        try:
            count = collection.count()
            if count == 0:
                self.logger.debug(f"Collection '{target_collection}' is empty.")
                return []
        except Exception as e:
            self.logger.warning(f"Could not determine count for collection '{target_collection}': {e}")
            count = top_k

        query_embedding = embeddings.embed_query(query)
        if query_embedding is None:
            self.logger.warning(f"Failed to generate query embedding for: '{query}'")
            return []

        try:
            n_results = min(top_k, count) if count > 0 else top_k
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )
        except Exception as e:
            self.logger.error(f"Error querying collection '{target_collection}': {e}")
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

        retrieved.sort(key=lambda x: x["distance"])
        return retrieved

    def clear_collection(self, collection_name: str) -> bool:
        """
        Delete and recreate the specified collection.

        Args:
            collection_name: Name of the collection to clear.

        Returns:
            True if cleared and recreated successfully, False otherwise.
        """
        try:
            self.client.delete_collection(name=collection_name)
            self.get_or_create_collection(collection_name)
            self.logger.info(f"Cleared and recreated collection '{collection_name}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear collection '{collection_name}': {e}")
            return False

    def get_collection_stats(self, collection_name: str | None = None) -> dict[str, Any]:
        """
        Get metadata and count statistics for the specified collection.

        Args:
            collection_name: Target collection (defaults to WORKSPACE_COLLECTION).

        Returns:
            Dictionary: {"name": str, "count": int}
        """
        target_name = collection_name or self.WORKSPACE_COLLECTION
        try:
            collection = self.get_or_create_collection(target_name)
            count = collection.count()
            return {"name": target_name, "count": count}
        except Exception as e:
            self.logger.warning(f"Could not retrieve stats for collection '{target_name}': {e}")
            return {"name": target_name, "count": 0}

    def save_debugging_note(self, note: str, context: dict[str, Any] | None = None) -> bool:
        """
        Save a debugging note or fix record into the reference collection.

        Args:
            note: Debugging note or explanation text.
            context: Optional dictionary with additional metadata (e.g. error, file, model).

        Returns:
            True if saved successfully, False otherwise.
        """
        if not note or not note.strip():
            return False

        try:
            collection = self.get_or_create_collection(self.REFERENCE_COLLECTION)
            timestamp = time.time()
            content_key = f"debugging_note:{timestamp}:{note}"
            doc_id = hashlib.md5(content_key.encode("utf-8", errors="ignore")).hexdigest()

            metadata: dict[str, str | int | float | bool] = {
                "type": "debugging_note",
                "timestamp": timestamp,
            }

            if context:
                for k, v in context.items():
                    if isinstance(v, (str, int, float, bool)):
                        metadata[k] = v
                    else:
                        try:
                            metadata[k] = json.dumps(v)
                        except Exception:
                            metadata[k] = str(v)

            try:
                vector = embeddings.embed_one(note)
            except embeddings.EmbeddingError as e:
                self.logger.error(f"Failed to embed debugging note: {e}")
                return False

            collection.add(
                ids=[doc_id],
                documents=[note],
                embeddings=[vector],
                metadatas=[metadata],
            )
            self.logger.info(f"Saved debugging note with ID {doc_id} to {self.REFERENCE_COLLECTION}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save debugging note: {e}")
            return False
