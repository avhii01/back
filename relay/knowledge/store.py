"""Optional persistent Chroma adapter.

The Docker Compose stack runs a Chroma server. If it is configured and reachable,
ingested chunks are mirrored there and queries use it. Any failure is logged once and
the caller falls back to the database vector index - the demo never claims Chroma ran
when it did not.

`chromadb` is an optional dependency (`pip install -e ".[rag]"`). It is imported lazily.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

log = logging.getLogger("relay.knowledge.chroma")


@dataclass
class ChromaHit:
    chunk_id: str
    score: float


class ChromaIndex:
    def __init__(self, url: str, collection: str):
        self.url = url
        self.collection_name = collection
        self._client = None
        self._collection = None
        self._warned = False

    def _connect(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            self._warn(f"chromadb is not installed ({type(e).__name__}); using database vectors")
            return None
        parsed = urlparse(self.url if "://" in self.url else f"http://{self.url}")
        try:
            self._client = chromadb.HttpClient(host=parsed.hostname or "localhost", port=parsed.port or 8000,
                                               ssl=parsed.scheme == "https")
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"})
            return self._collection
        except Exception as e:  # pragma: no cover - service availability
            self._warn(f"Chroma unavailable ({type(e).__name__}); using database vectors")
            return None

    def _warn(self, message: str) -> None:
        if not self._warned:
            log.warning(message)
            self._warned = True

    def upsert(self, rows: list[dict], embeddings: list[list[float]]) -> bool:  # pragma: no cover - service path
        collection = self._connect()
        if collection is None:
            return False
        if not rows:
            return True
        try:
            collection.upsert(
                ids=[r["chunk_id"] for r in rows],
                embeddings=embeddings,
                documents=[r["content"] for r in rows],
                metadatas=[{"document_id": r["document_id"], "title": r["title"], "version": r["version"],
                            "section": r["section"]} for r in rows],
            )
            return True
        except Exception as e:
            self._warn(f"Chroma upsert failed ({type(e).__name__}); using database vectors")
            return False

    def query(self, embedding: list[float], k: int) -> list[ChromaHit] | None:  # pragma: no cover - service path
        collection = self._connect()
        if collection is None:
            return None
        try:
            res = collection.query(query_embeddings=[embedding], n_results=k,
                                   include=["distances"])
        except Exception as e:
            self._warn(f"Chroma query failed ({type(e).__name__}); using database vectors")
            return None
        ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        return [ChromaHit(chunk_id=cid, score=max(0.0, 1.0 - float(dist)))
                for cid, dist in zip(ids, distances, strict=False)]
