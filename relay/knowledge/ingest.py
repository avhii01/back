"""Knowledge ingestion: authored demo policy -> sections -> chunks -> embeddings -> store.

Each chunk keeps a stable `chunk_id`, document title, version and section so every
retrieved passage can be cited precisely. Ingestion is idempotent: unchanged chunks
are left alone, changed chunks are re-embedded and re-upserted.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select

from relay.context import AppContext
from relay.knowledge.embeddings import EmbeddingClient, EmbeddingError
from relay.knowledge.store import ChromaIndex
from relay.models import KnowledgeChunk, utcnow

log = logging.getLogger("relay.knowledge.ingest")
DOCS_DIR = Path(__file__).resolve().parent / "documents"
MAX_CHUNK_CHARS = 900


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    version: str
    section: str
    content: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"')
    return meta, parts[2].lstrip("\n")


def chunk_text(text: str, document_id: str, title: str, version: str) -> list[Chunk]:
    """Split one document into section-addressed, size-bounded chunks."""
    chunks: list[Chunk] = []
    current = "Overview"
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        raw = "\n".join(body).strip()
        if not raw:
            body = []
            return
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        part: list[str] = []
        part_no = 1

        def emit() -> None:
            nonlocal part, part_no
            content = "\n\n".join(part).strip()
            if content:
                chunks.append(Chunk(chunk_id=f"{document_id}::{_slug(current)}::{part_no:02d}",
                                    document_id=document_id, title=title, version=version,
                                    section=current, content=content))
                part_no += 1
            part = []

        for paragraph in paragraphs:
            if part and sum(len(x) + 2 for x in part) + len(paragraph) > MAX_CHUNK_CHARS:
                emit()
            if len(paragraph) > MAX_CHUNK_CHARS:
                # Long paragraph: split on sentence boundaries, preserving readable citations.
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    if part and sum(len(x) + 2 for x in part) + len(sentence) > MAX_CHUNK_CHARS:
                        emit()
                    part.append(sentence)
            else:
                part.append(paragraph)
        emit()
        body = []

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            current = line[3:].strip() or "Overview"
        else:
            body.append(line)
    flush()
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    out: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        meta, body = _frontmatter(path.read_text(encoding="utf-8"))
        document_id = meta.get("document_id") or path.stem
        title = meta.get("title") or document_id.replace("-", " ").title()
        version = meta.get("version") or "unversioned"
        out.extend(chunk_text(body, document_id, title, version))
    return out


def ingest(app: AppContext, *, force: bool = False) -> dict:
    chunks = load_chunks()
    changed: list[Chunk] = []
    with app.session() as s:
        existing = {c.chunk_id: c for c in s.scalars(select(KnowledgeChunk))}
        for chunk in chunks:
            row = existing.get(chunk.chunk_id)
            if row is None or force or row.content_hash != chunk.content_hash:
                changed.append(chunk)

    embeddings: list[list[float]] = []
    client = EmbeddingClient(app.settings)
    embed_error: str | None = None
    if changed and client.configured:
        try:
            embeddings = client.embed([c.content for c in changed])
        except EmbeddingError as e:
            embed_error = str(e)
            log.warning("embedding ingestion failed (%s); chunks stored without vectors", e)

    now = utcnow()
    with app.session() as s:
        for chunk in chunks:
            row = s.get(KnowledgeChunk, chunk.chunk_id)
            if row is None:
                row = KnowledgeChunk(chunk_id=chunk.chunk_id, document_id=chunk.document_id,
                                     title=chunk.title, version=chunk.version, section=chunk.section,
                                     content=chunk.content, content_hash=chunk.content_hash,
                                     updated_at=now)
                s.add(row)
            content_changed = row.content_hash != chunk.content_hash or force
            if content_changed:
                row.content_hash = chunk.content_hash
                row.content = chunk.content
                # Never keep a vector that belongs to different text.
                row.embedding = None
                row.embedding_model = None
            # Metadata changes (title/version/section) must reach citations even when the body is unchanged.
            row.document_id, row.title, row.version, row.section = (
                chunk.document_id, chunk.title, chunk.version, chunk.section)
            if content_changed:
                row.updated_at = now
        s.flush()
        # Assign vectors only after rows exist; changed order matches `changed`.
        for chunk, vector in zip(changed, embeddings, strict=False):
            row = s.get(KnowledgeChunk, chunk.chunk_id)
            row.embedding = vector
            row.embedding_model = client.model
        s.commit()

    chroma_status = "not_configured"
    if app.settings.chroma_url:
        with app.session() as s:
            rows = list(s.scalars(select(KnowledgeChunk).where(KnowledgeChunk.embedding.is_not(None))))
        index = ChromaIndex(app.settings.chroma_url, app.settings.chroma_collection)
        ok = index.upsert(
            [{"chunk_id": r.chunk_id, "content": r.content, "document_id": r.document_id,
              "title": r.title, "version": r.version, "section": r.section} for r in rows],
            [r.embedding for r in rows],
        )
        chroma_status = "synced" if ok else "unavailable"

    return {"documents": len({c.document_id for c in chunks}), "chunks": len(chunks),
            "changed_chunks": len(changed), "embedded_chunks": len(embeddings),
            "embedding_error": embed_error, "embedding_model": client.model if client.configured else None,
            "chroma": chroma_status}


def ensure_ingested(app: AppContext) -> None:
    with app.session() as s:
        count = s.scalar(select(func.count()).select_from(KnowledgeChunk)) or 0
    if count == 0:
        ingest(app)
