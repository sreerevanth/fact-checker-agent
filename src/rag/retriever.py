"""
RAG Retrieval Engine
====================
Manages the FAISS vector index and provides semantic search over
ingested news articles and fact-check documents.
"""
from __future__ import annotations

import asyncio
import json
import pickle
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.logging import get_logger
from src.utils.models import Evidence, SourceCategory, SourceInfo
from src.utils.settings import get_settings

log = get_logger(__name__)

# Lazy import FAISS to avoid import error if not installed
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    log.warning("faiss not installed — vector search disabled")


class DocumentStore:
    """Simple metadata store for retrieved documents."""

    def __init__(self) -> None:
        self._docs: dict[int, dict[str, Any]] = {}
        self._counter: int = 0

    def add(self, doc: dict[str, Any]) -> int:
        idx = self._counter
        self._docs[idx] = doc
        self._counter += 1
        return idx

    def get(self, idx: int) -> dict[str, Any] | None:
        return self._docs.get(idx)

    def __len__(self) -> int:
        return self._counter


class FAISSRetriever:
    """
    FAISS-backed semantic retrieval engine.

    Embeddings: sentence-transformers (local, no API cost).
    Index type: IndexFlatIP (inner-product / cosine after normalization).
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: SentenceTransformer | None = None
        self._index: Any = None  # faiss.Index
        self._store = DocumentStore()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load embedding model and FAISS index (or create fresh)."""
        log.info("Initializing RAG retriever", model=self.settings.embedding_model)

        # Load embedding model in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None, lambda: SentenceTransformer(self.settings.embedding_model)
        )

        index_path = self.settings.faiss_index_path
        if FAISS_AVAILABLE and index_path.with_suffix(".index").exists():
            await self._load_index(index_path)
        elif FAISS_AVAILABLE:
            self._create_index()
        else:
            log.warning("Running without FAISS — retrieval will be keyword-only fallback")

        log.info("Retriever ready", docs=len(self._store))

    def _create_index(self) -> None:
        dim = self.settings.faiss_dimension
        self._index = faiss.IndexFlatIP(dim)
        log.info("Created fresh FAISS index", dimension=dim)

    async def _load_index(self, base_path: Path) -> None:
        loop = asyncio.get_event_loop()

        def _load() -> None:
            self._index = faiss.read_index(str(base_path.with_suffix(".index")))
            store_path = base_path.with_suffix(".store")
            if store_path.exists():
                with open(store_path, "rb") as f:
                    self._store = pickle.load(f)

        await loop.run_in_executor(None, _load)
        log.info("Loaded FAISS index", docs=len(self._store))

    async def save(self) -> None:
        if self._index is None:
            return
        loop = asyncio.get_event_loop()
        base_path = self.settings.faiss_index_path
        base_path.parent.mkdir(parents=True, exist_ok=True)

        def _save() -> None:
            faiss.write_index(self._index, str(base_path.with_suffix(".index")))
            with open(base_path.with_suffix(".store"), "wb") as f:
                pickle.dump(self._store, f)

        await loop.run_in_executor(None, _save)
        log.info("Saved FAISS index", docs=len(self._store))

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def add_documents(self, documents: list[dict[str, Any]]) -> int:
        """Embed and index a batch of documents. Returns count added."""
        if not documents or self._model is None:
            return 0

        texts = [d.get("text", "") for d in documents]
        loop = asyncio.get_event_loop()

        t0 = time.perf_counter()
        embeddings: np.ndarray = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        )
        elapsed = (time.perf_counter() - t0) * 1000
        log.debug("Embedded documents", count=len(texts), latency_ms=round(elapsed, 1))

        async with self._lock:
            if FAISS_AVAILABLE and self._index is not None:
                self._index.add(embeddings.astype("float32"))
            for doc in documents:
                self._store.add(doc)

        return len(documents)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        """Semantic search — returns top_k Evidence objects."""
        if self._model is None:
            return []

        k = top_k or self.settings.retrieval_top_k
        loop = asyncio.get_event_loop()

        # Embed query
        query_vec: np.ndarray = await loop.run_in_executor(
            None, lambda: self._model.encode([query], normalize_embeddings=True)
        )

        if not FAISS_AVAILABLE or self._index is None or self._index.ntotal == 0:
            return self._keyword_fallback(query, k)

        async with self._lock:
            scores, indices = self._index.search(query_vec.astype("float32"), k)

        results: list[Evidence] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self._store.get(int(idx))
            if doc is None:
                continue

            source = SourceInfo(
                name=doc.get("source_name", "Unknown"),
                url=doc.get("url"),
                credibility_score=doc.get("credibility_score", 0.5),
                category=SourceCategory(doc.get("category", "unknown")),
            )
            results.append(
                Evidence(
                    text=doc.get("text", ""),
                    source=source,
                    relevance_score=float(score),
                )
            )

        return results

    def _keyword_fallback(self, query: str, k: int) -> list[Evidence]:
        """Simple keyword overlap when FAISS is unavailable."""
        query_words = set(query.lower().split())
        scored: list[tuple[float, dict]] = []
        for i in range(len(self._store)):
            doc = self._store.get(i)
            if doc is None:
                continue
            text_words = set(doc.get("text", "").lower().split())
            overlap = len(query_words & text_words) / (len(query_words) + 1)
            if overlap > 0:
                scored.append((overlap, doc))
        scored.sort(reverse=True)
        results = []
        for score, doc in scored[:k]:
            source = SourceInfo(
                name=doc.get("source_name", "Unknown"),
                url=doc.get("url"),
                credibility_score=doc.get("credibility_score", 0.5),
            )
            results.append(Evidence(text=doc.get("text", ""), source=source, relevance_score=score))
        return results

    @property
    def index_size(self) -> int:
        return len(self._store)
