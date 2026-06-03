"""
Layer 2 — RAG retrieval against ChromaDB.

Queries two collections concurrently:
  - injection_examples  → top-K semantically similar known injection texts
  - owasp_reference     → top-1 most relevant OWASP LLM Top-10 technique

Public API: `get_rag_retriever()` returns a cached singleton; call
`await retriever.retrieve(text)` to get a `RetrievalContext`.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.core.chroma_client import get_chroma_client
from app.core.config import get_settings
from app.models.schemas import RetrievalContext

log = logging.getLogger(__name__)

_EXAMPLES_COLLECTION = "injection_examples"
_OWASP_COLLECTION = "owasp_reference"


class RagRetriever:
    """
    Wraps two ChromaDB collections and exposes a single async `retrieve` method.

    Lifetime: one instance per process (via `get_rag_retriever()`).
    The SentenceTransformer model is loaded in `__init__`; that cost is paid
    once at startup, not per request.
    """

    def __init__(self) -> None:
        settings = get_settings()
        client = get_chroma_client()

        self._top_k_examples: int = settings.examples_top_k
        self._top_k_owasp: int = settings.owasp_top_k

        # Single embedding function instance — used to embed the query text
        # once and then passed as a pre-computed vector to both collections.
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
            device="cpu",
        )

        # Get collections. If an ingest script hasn't been run yet, the
        # collection won't exist — log a warning and continue with None so the
        # detection pipeline degrades gracefully rather than crashing.
        self._examples_col = self._get_collection(client, _EXAMPLES_COLLECTION)
        self._owasp_col = self._get_collection(client, _OWASP_COLLECTION)

    # ── Initialisation helpers ────────────────────────────────────────────────

    def _get_collection(self, client, name: str):
        """Return the named ChromaDB collection, or None if it doesn't exist."""
        try:
            # Pass the embedding function so ChromaDB can accept query_texts.
            # We won't actually use query_texts (we pass query_embeddings
            # directly), but ChromaDB requires ef to be set for validation.
            return client.get_collection(name=name, embedding_function=self._ef)
        except Exception:
            log.warning(
                "ChromaDB collection '%s' not found — run the ingest script to populate it. "
                "Layer 2 will return empty context until then.",
                name,
            )
            return None

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """
        Compute a single embedding vector for `text`.

        SentenceTransformerEmbeddingFunction.__call__ accepts a list of
        documents and returns a list of embedding vectors. We extract index 0.
        """
        embeddings: list[list[float]] = self._ef([text])
        return embeddings[0]

    # ── Collection queries (sync — called via asyncio.to_thread) ─────────────

    def _query_examples(self, embedding: list[float]) -> list[dict]:
        """Query injection_examples with a pre-computed embedding vector."""
        if self._examples_col is None:
            return []

        count = self._examples_col.count()
        if count == 0:
            log.warning("Collection '%s' is empty — run ingest_datasets.py.", _EXAMPLES_COLLECTION)
            return []

        n = min(self._top_k_examples, count)
        try:
            results = self._examples_col.query(
                query_embeddings=[embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.error("ChromaDB query failed on '%s': %s", _EXAMPLES_COLLECTION, exc)
            return []

        return [
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "distance": round(dist, 4),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def _query_owasp(self, embedding: list[float]) -> list[dict]:
        """Query owasp_reference with a pre-computed embedding vector."""
        if self._owasp_col is None:
            return []

        count = self._owasp_col.count()
        if count == 0:
            log.warning("Collection '%s' is empty — run ingest_owasp.py.", _OWASP_COLLECTION)
            return []

        n = min(self._top_k_owasp, count)
        try:
            results = self._owasp_col.query(
                query_embeddings=[embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.error("ChromaDB query failed on '%s': %s", _OWASP_COLLECTION, exc)
            return []

        return [
            {
                "technique_id": meta.get("technique_id", ""),
                "technique_name": meta.get("technique_name", ""),
                "category": meta.get("category", ""),
                "description": doc,
                "distance": round(dist, 4),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    async def retrieve(self, text: str) -> RetrievalContext:
        """
        Embed `text` once, then query both collections concurrently.

        Steps:
          1. Embed in a thread (CPU-bound, ~50ms on CPU for a short text).
          2. Concurrently query injection_examples and owasp_reference, each
             in its own thread (disk I/O for HNSW traversal).

        Returns an empty RetrievalContext if ingest hasn't been run — the
        LLM judges still execute, they just lack retrieval context.
        """
        # Step 1 — single embed (avoids running the model twice).
        embedding: list[float] = await asyncio.to_thread(self._embed, text)

        # Step 2 — concurrent HNSW queries; genuine I/O parallelism via threads.
        similar_examples, owasp_techniques = await asyncio.gather(
            asyncio.to_thread(self._query_examples, embedding),
            asyncio.to_thread(self._query_owasp, embedding),
        )

        log.debug(
            "RAG retrieval | examples_returned=%d | owasp_returned=%d",
            len(similar_examples),
            len(owasp_techniques),
        )

        return RetrievalContext(
            similar_examples=similar_examples,
            owasp_techniques=owasp_techniques,
        )


@lru_cache(maxsize=1)
def get_rag_retriever() -> RagRetriever:
    """
    Return the process-wide singleton RagRetriever.

    The SentenceTransformer model load (~200ms) is paid on the first call.
    All subsequent calls return the cached instance instantly.
    """
    return RagRetriever()
