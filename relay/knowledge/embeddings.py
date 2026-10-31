"""Embedding client for RAG.

Uses any OpenAI-compatible `/embeddings` endpoint. If no key is configured the caller
must fall back to lexical retrieval; this module never pretends an embedding was made.
"""
from __future__ import annotations

import time

import httpx

from relay.config import Settings


class EmbeddingError(Exception):
    pass


class EmbeddingClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.model = settings.embedding_model
        self._base = settings.embedding_url
        self._key = settings.embedding_key
        self._timeout = settings.embedding_timeout_s
        self._client = httpx.Client(transport=transport)

    @property
    def configured(self) -> bool:
        return bool(self._key and self.model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.configured:
            raise EmbeddingError("embedding provider is not configured")
        if not texts:
            return []
        out: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            body = {"model": self.model, "input": batch}
            last = "unknown error"
            for attempt in range(1, 3):
                try:
                    r = self._client.post(f"{self._base}/embeddings", json=body, headers=headers,
                                          timeout=self._timeout)
                except httpx.HTTPError as e:
                    last = f"transport error: {type(e).__name__}"
                else:
                    if r.status_code == 200:
                        try:
                            rows = r.json()["data"]
                            vectors = [row["embedding"] for row in sorted(rows, key=lambda x: x.get("index", 0))]
                            if len(vectors) != len(batch) or not all(isinstance(v, list) and v for v in vectors):
                                raise ValueError("malformed embedding response")
                        except (ValueError, KeyError, TypeError) as e:
                            raise EmbeddingError("malformed embedding response") from e
                        out.extend([[float(x) for x in v] for v in vectors])
                        break
                    last = f"provider returned HTTP {r.status_code}"
                    if r.status_code not in (408, 409, 429) and r.status_code < 500:
                        raise EmbeddingError(last)
                if attempt < 2:
                    time.sleep(0.2)
            else:
                raise EmbeddingError(last)
        return out
