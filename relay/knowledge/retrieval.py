"""Grounded policy retrieval with citations.

Two real retrieval paths:
  * vector  - embeddings + Chroma (when configured) or the persisted DB vectors
  * lexical - a clearly labelled deterministic fallback (no embeddings, no LLM)

The answer composer is extractive: it quotes retrieved passages instead of inventing
policy. A caller may optionally rephrase with an LLM, but must keep the retrieved
citation set and fall back here on any invalid output.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from sqlalchemy import select

from relay.context import AppContext
from relay.knowledge.embeddings import EmbeddingClient, EmbeddingError
from relay.knowledge.ingest import ensure_ingested
from relay.knowledge.store import ChromaIndex
from relay.models import KnowledgeChunk
from relay.untrusted import detect_injection

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "being", "but", "by", "can", "could", "did", "do", "does", "doing",
    "for", "from", "get", "got", "had", "has", "have", "he", "her", "here", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "just", "like", "long", "me", "more", "most", "my", "need",
    "no", "not", "of", "on", "or", "our", "out", "please", "should", "so", "some", "take", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "those", "time", "to", "up", "us",
    "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "yes",
    "you", "your",
}
_MODE_LABELS = {
    "vector_chroma": "Vector RAG (embeddings + Chroma)",
    "vector_database": "Vector RAG (embeddings + database vectors)",
    "lexical_fallback": "Lexical fallback (no embeddings, deterministic)",
}


@dataclass
class Passage:
    chunk_id: str
    document_id: str
    title: str
    version: str
    section: str
    content: str
    score: float


@dataclass
class RetrievalResult:
    query: str
    mode: str
    passages: list[Passage] = field(default_factory=list)
    insufficient: bool = False
    conflicting: bool = False
    unsafe_sources: bool = False
    reason: str | None = None
    embedding_model: str | None = None

    @property
    def mode_label(self) -> str:
        return _MODE_LABELS.get(self.mode, self.mode)

    def citations(self) -> list[dict]:
        out = []
        for p in self.passages:
            text = re.sub(r"\s+", " ", p.content).strip()
            excerpt = text if len(text) <= 320 else text[:317].rstrip() + "…"
            out.append({"chunk_id": p.chunk_id, "document_id": p.document_id, "title": p.title,
                        "version": p.version, "section": p.section, "excerpt": excerpt,
                        "score": round(p.score, 4), "retrieval": self.mode})
        return out


def _stem(word: str) -> str:
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 4 and word.endswith("es"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    return word


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return max(0.0, min(1.0, dot / (na * nb))) if na and nb else 0.0


def _lexical(app: AppContext, query: str, k: int, required_document_ids: list[str] | None) -> RetrievalResult:
    with app.session() as s:
        rows = list(s.scalars(select(KnowledgeChunk)))
    q = _tokens(query)
    if not rows or not q:
        return RetrievalResult(query=query, mode="lexical_fallback", insufficient=True,
                               reason="no policy text matched the question")
    token_rows = {r.chunk_id: set(_tokens(r.content)) for r in rows}
    df: dict[str, int] = {}
    for toks in token_rows.values():
        for token in toks & set(q):
            df[token] = df.get(token, 0) + 1
    # Unknown words (order numbers, typos, off-topic nouns) must not dilute a genuinely
    # relevant question. But a question mostly made of unknown words is not answerable.
    known = [t for t in q if df.get(t, 0) > 0]
    if not known or len(known) / len(q) < app.settings.rag_lexical_min_score:
        return RetrievalResult(query=query, mode="lexical_fallback", insufficient=True,
                               reason="the question does not match the policy knowledge base")
    n = len(rows)
    idf = {t: math.log((n + 1) / (df.get(t, 0) + 1)) + 1.0 for t in known}
    total = sum(idf[t] for t in known) or 1.0
    scored: list[tuple[float, KnowledgeChunk]] = []
    for row in rows:
        matched = set(known) & token_rows[row.chunk_id]
        if not matched:
            continue
        coverage = sum(idf[t] for t in matched) / total
        phrase = 0.08 if query.lower().strip() in row.content.lower() else 0.0
        boost = 0.12 if required_document_ids and row.document_id in required_document_ids else 0.0
        scored.append((min(1.0, coverage + phrase + boost), row))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = scored[:k]
    min_score = app.settings.rag_lexical_min_score
    passages, unsafe = _screen([Passage(row.chunk_id, row.document_id, row.title, row.version, row.section,
                                        row.content, score)
                                for score, row in hits if score >= min_score])
    if not passages:
        return RetrievalResult(query=query, mode="lexical_fallback", insufficient=True,
                               unsafe_sources=unsafe,
                               reason="no safe, relevant passage was found to answer this")
    return RetrievalResult(query=query, mode="lexical_fallback", passages=passages,
                           conflicting=_conflicting(passages), unsafe_sources=unsafe)


def _vector(app: AppContext, query: str, k: int, required_document_ids: list[str] | None) -> RetrievalResult | None:
    client = EmbeddingClient(app.settings)
    if not client.configured:
        return None
    try:
        vector = client.embed([query])[0]
    except EmbeddingError:
        return None

    mode = "vector_database"
    scored: list[tuple[float, KnowledgeChunk]] = []
    if app.settings.chroma_url:
        hits = ChromaIndex(app.settings.chroma_url, app.settings.chroma_collection).query(vector, k * 2)
        if hits is not None:
            with app.session() as s:
                rows = {r.chunk_id: r for r in s.scalars(select(KnowledgeChunk))}
            scored = [(h.score, rows[h.chunk_id]) for h in hits if h.chunk_id in rows]
            mode = "vector_chroma"
    if not scored:
        with app.session() as s:
            rows = list(s.scalars(select(KnowledgeChunk).where(KnowledgeChunk.embedding.is_not(None))))
        scored = [(_cosine(vector, r.embedding or []), r) for r in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    # Required documents get a small boost so policy for the observed failure wins ties.
    scored = [(min(1.0, score + (0.05 if required_document_ids and row.document_id in required_document_ids else 0.0)), row)
              for score, row in scored]
    scored.sort(key=lambda x: x[0], reverse=True)
    passages, unsafe = _screen([Passage(row.chunk_id, row.document_id, row.title, row.version, row.section,
                                        row.content, score)
                                for score, row in scored[:k] if score >= app.settings.rag_min_score])
    if not passages:
        return RetrievalResult(query=query, mode=mode, insufficient=True,
                               unsafe_sources=unsafe,
                               reason="no safe, similar passage was found to answer this",
                               embedding_model=client.model)
    return RetrievalResult(query=query, mode=mode, passages=passages, conflicting=_conflicting(passages),
                           unsafe_sources=unsafe, embedding_model=client.model)


_CONTRADICTIONS = [
    (re.compile(r"\b(?:is|are|will be)\s+approved\s+automatically\b", re.IGNORECASE),
     re.compile(r"\b(?:never|not|cannot\s+be)\s+(?:automatically\s+)?approved\b", re.IGNORECASE)),
    (re.compile(r"\b(?:can|could|may)\s+(?:be\s+)?(?:cancelled|canceled|refunded)\b", re.IGNORECASE),
     re.compile(r"\b(?:cannot|can\s+not|never|not)\s+(?:be\s+)?(?:cancelled|canceled|refunded)\b",
                re.IGNORECASE)),
    (re.compile(r"\brefunds?\s+(?:is|are)\s+(?:complete|completed|issued)\b", re.IGNORECASE),
     re.compile(r"\b(?:not|never|cannot).{0,30}\b(?:complete|completed|issued)\b", re.IGNORECASE)),
]


def _contradicts(a: Passage, b: Passage) -> bool:
    """Conservative conflict check: shared subject + directly opposing modal claims.

    A score tie between different documents is NOT treated as a conflict on its own;
    that produced false uncertainty for broad questions.
    """
    shared = set(_tokens(a.content)) & set(_tokens(b.content))
    if len(shared) < 2:
        return False
    for positive, negative in _CONTRADICTIONS:
        if (positive.search(a.content) and negative.search(b.content)) or (
                negative.search(a.content) and positive.search(b.content)):
            return True
    return False


def _conflicting(passages: list[Passage]) -> bool:
    top = passages[:4]
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            if a.document_id != b.document_id and _contradicts(a, b):
                return True
    return False


def _screen(passages: list[Passage]) -> tuple[list[Passage], bool]:
    """Drop retrieved documents that contain instructions aimed at the assistant."""
    safe: list[Passage] = []
    unsafe = False
    for passage in passages:
        if detect_injection(passage.content):
            unsafe = True
        else:
            safe.append(passage)
    return safe, unsafe


def retrieve(app: AppContext, query: str, *, k: int = 4,
             required_document_ids: list[str] | None = None) -> RetrievalResult:
    ensure_ingested(app)
    result = _vector(app, query, k, required_document_ids)
    if result is None:
        result = _lexical(app, query, k, required_document_ids)
    if result.insufficient and result.mode != "lexical_fallback":
        # A configured embedding path that found nothing must not silently become a
        # different retrieval method: report the vector result as insufficient.
        return result
    return result


def answer_policy_question(app: AppContext, query: str, *,
                           required_document_ids: list[str] | None = None) -> dict:
    result = retrieve(app, query, required_document_ids=required_document_ids)
    citations = result.citations()
    if result.unsafe_sources:
        return {
            "answer": ("I found policy text that contains instructions I cannot safely follow, so I will not "
                       "use it as an answer. A human specialist can review this question instead."),
            "citations": citations, "retrieval_mode": result.mode, "retrieval_label": result.mode_label,
            "embedding_model": result.embedding_model, "insufficient": result.insufficient,
            "conflicting": False, "unsafe_sources": True,
            "escalation_advised": True, "engine": "extractive_grounded",
        }
    if result.insufficient:
        return {
            "answer": ("I could not find a policy passage I can rely on for that question. "
                       "I can create a ticket for a human specialist rather than guess."),
            "citations": citations, "retrieval_mode": result.mode, "retrieval_label": result.mode_label,
            "embedding_model": result.embedding_model, "insufficient": True, "conflicting": False,
            "unsafe_sources": False, "escalation_advised": True, "engine": "extractive_grounded",
        }
    if result.conflicting:
        return {
            "answer": ("I found policy passages that point in different directions, so I cannot give a "
                       "single reliable answer. I can escalate this for human review."),
            "citations": citations, "retrieval_mode": result.mode, "retrieval_label": result.mode_label,
            "embedding_model": result.embedding_model, "insufficient": False, "conflicting": True,
            "unsafe_sources": False, "escalation_advised": True, "engine": "extractive_grounded",
        }
    selected: list[Passage] = []
    seen_sections: set[tuple[str, str]] = set()
    for p in result.passages:
        if (p.document_id, p.section) in seen_sections:
            continue
        seen_sections.add((p.document_id, p.section))
        selected.append(p)
        if len(selected) == 2:
            break
    parts = []
    for p in selected:
        text = re.sub(r"\s+", " ", p.content).strip()
        if len(text) > 420:
            text = text[:417].rstrip() + "…"
        parts.append(f'According to "{p.title}" (v{p.version}, §{p.section}): {text}')
    return {
        "answer": " ".join(parts),
        "citations": citations, "retrieval_mode": result.mode, "retrieval_label": result.mode_label,
        "embedding_model": result.embedding_model, "insufficient": False, "conflicting": False,
        "unsafe_sources": False, "escalation_advised": False, "engine": "extractive_grounded",
    }
