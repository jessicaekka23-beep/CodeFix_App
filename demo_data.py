"""Instrumented RAG indexing for the demo UI. Pure Python, no Streamlit.

IndexRun walks the same steps VectorStore.index_workspace does, but records what happens at each one so the UI
can show it: files loaded -> chunks cut -> a sample embedding -> chunks stored -> a 2-D map of the stored vectors.
The actual write to ChromaDB is still done by VectorStore.index_workspace, so what you see is what the agent
later searches. Every collaborator is injectable, so tests run without Ollama or ChromaDB.
"""
import threading
import time
from pathlib import Path

COLLECTION = "workspace_code"
STAGES = ("load", "chunk", "embed", "store", "map")
STAGE_LABELS = {"load": "Load files", "chunk": "Cut chunks", "embed": "Embed", "store": "Store", "map": "Map"}
MAX_MAP_POINTS = 1500


# ── 2-D projection of embedding vectors (PCA via SVD) ──────────
def fit_projection(vectors):
    """Return {"mean", "comps", "lo", "hi", "points"} or None (needs numpy and at least 2 vectors)."""
    try:
        import numpy as np
    except ImportError:
        return None
    if vectors is None or len(vectors) < 2:
        return None
    x = np.asarray(vectors, dtype=float)
    mean = x.mean(axis=0)
    centred = x - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    comps = vt[:2]
    if comps.shape[0] < 2:                                   # a 1-D embedding space: pad the second axis
        comps = np.vstack([comps, np.zeros_like(comps[0])])
    xy = centred @ comps.T
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    pts = (xy - lo) / span
    return {"mean": mean.tolist(), "comps": comps.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
            "points": [(float(a), float(b)) for a, b in pts]}


def project_point(model, vector):
    """Place one more vector (e.g. the query) on an existing map. Clamped to the 0..1 square."""
    if not model or vector is None:
        return None
    import numpy as np
    comps = np.asarray(model["comps"])
    xy = (np.asarray(vector, dtype=float) - np.asarray(model["mean"])) @ comps.T
    lo, hi = np.asarray(model["lo"]), np.asarray(model["hi"])
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    out = (xy - lo) / span
    return float(min(max(out[0], -0.05), 1.05)), float(min(max(out[1], -0.05), 1.05))


# ── default collaborators (imported lazily: chromadb is heavy and absent in unit tests) ──
def _default_loader(ws):
    import rag
    return rag.load_documents(ws)


def _default_chunker(docs):
    import rag
    return rag.chunk_all_documents(docs)


def _default_embedder(text):
    import embeddings
    try:
        return embeddings.embed_one(text)
    except embeddings.EmbeddingError:
        return None


def _default_store():
    from vector_store import VectorStore
    return VectorStore()


def _default_fetch_vectors(store):
    """(vectors, metadatas) for everything in the project collection."""
    col = store.get_or_create_collection(COLLECTION)
    got = col.get(include=["embeddings", "metadatas"], limit=MAX_MAP_POINTS)
    vecs = got.get("embeddings")
    return (list(vecs) if vecs is not None else []), (got.get("metadatas") or [])


class IndexRun:
    """One indexing pass, run in a background thread. The UI polls the plain attributes below."""

    def __init__(self, workspace, loader=None, chunker=None, embedder=None, store_factory=None, fetch_vectors=None):
        self.workspace = str(workspace)
        self._loader = loader or _default_loader
        self._chunker = chunker or _default_chunker
        self._embedder = embedder or _default_embedder
        self._store_factory = store_factory or _default_store
        self._fetch = fetch_vectors or _default_fetch_vectors
        self.stage = None                     # stage being worked on right now
        self.done_stages = []                 # finished stages, in order
        self.timings = {}                     # stage -> seconds
        self.docs, self.chunks = [], []       # what was loaded / cut
        self.embedding = None                 # {"dim", "values", "seconds", "source", "lines", "text"}
        self.stats, self.count = {}, None     # from index_workspace / collection.count()
        self.map = None                       # {"mean","comps","lo","hi","points"}
        self.map_meta = []                    # per point: {"file", "chunk_id", "start", "end"}
        self.error = None
        self.ended = False
        self.started = time.time()
        self.finished = None

    # control
    def start(self):
        threading.Thread(target=self._work, daemon=True, name="codepilot-index").start()
        return self

    def alive(self):
        return not self.ended

    def ok(self):
        return self.ended and self.error is None

    def elapsed(self):
        return (self.finished or time.time()) - self.started

    def _rel(self, path):
        try:
            rel = Path(path).resolve().relative_to(Path(self.workspace).resolve())
            # Normalise to forward slashes so the UI (and tests) read the same on every OS.
            return str(rel).replace("\\", "/")
        except Exception:
            return Path(path).name

    def _run_stage(self, name, fn):
        self.stage = name
        t0 = time.time()
        out = fn()
        self.timings[name] = time.time() - t0
        self.done_stages.append(name)
        return out

    def _work(self):
        try:
            self._run_stage("load", self._load)
            self._run_stage("chunk", self._chunk)
            self._run_stage("embed", self._embed)
            store = self._run_stage("store", self._store)
            try:
                self._run_stage("map", lambda: self._map(store))
            except Exception:                                  # the map is a bonus; never fail the index over it
                self.map = None
                if "map" not in self.done_stages:
                    self.done_stages.append("map")
        except Exception as exc:
            self.error = self.error or f"{type(exc).__name__}: {exc}"
        finally:
            self.stage = None
            self.finished = time.time()
            self.ended = True

    # stages
    def _load(self):
        docs = self._loader(self.workspace) or []
        self.docs = [{"file": self._rel(d.get("path", "")), "lines": len((d.get("content") or "").splitlines()),
                      "chars": len(d.get("content") or ""), "ext": d.get("extension", "")} for d in docs]
        self._raw_docs = docs
        if not docs:
            self.error = "No supported source files found in this folder."
            raise RuntimeError(self.error)

    def _chunk(self):
        chunks = self._chunker(self._raw_docs) or []
        self.chunks = []
        for c in chunks:
            m = c.get("metadata", {})
            self.chunks.append({"file": self._rel(m.get("source", "")), "id": m.get("chunk_id", ""), "start": m.get("start_line", 0),
                                "end": m.get("end_line", 0), "chars": len(c.get("text", "")), "text": c.get("text", "")})
        if not self.chunks:
            self.error = "The files produced no chunks."
            raise RuntimeError(self.error)

    def _embed(self):
        """Embed ONE chunk up front, only to show what an embedding is (and to fail early if Ollama is not ready)."""
        sample = max(self.chunks[: 20], key=lambda c: c["chars"])
        t0 = time.time()
        vec = self._embedder(sample["text"])
        if not vec:
            self.error = "Could not create an embedding. Is Ollama running with `nomic-embed-text` pulled?"
            raise RuntimeError(self.error)
        self.embedding = {"dim": len(vec), "values": [float(v) for v in vec[:96]], "seconds": time.time() - t0,
                          "source": sample["file"], "lines": f"{sample['start']}-{sample['end']}", "text": sample["text"][:400]}

    def _store(self):
        store = self._store_factory()
        store.clear_collection(COLLECTION)                     # same as the main app: no stale chunks from another project
        self.stats = dict(store.index_workspace(self.workspace) or {})
        try:
            self.count = store.get_collection_stats(COLLECTION).get("count")
        except Exception:
            self.count = self.stats.get("indexed")
        if self.stats.get("errors"):
            self.error = f"{self.stats['errors']} chunk(s) failed to embed or store."
        return store

    def _map(self, store):
        vectors, metas = self._fetch(store)
        if vectors is None or len(vectors) < 2:
            return
        self.map = fit_projection(vectors)
        self.map_meta = [{"file": self._rel((m or {}).get("source", "")), "chunk_id": (m or {}).get("chunk_id", ""),
                          "start": (m or {}).get("start_line", 0), "end": (m or {}).get("end_line", 0)} for m in metas]
