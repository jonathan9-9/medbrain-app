"""Parsing + chunking for the corpus.

Design choice (see DESIGN.md section 3): the corpus is organized SOPs /
policies / patient-education handouts that already have meaningful
section structure (YAML frontmatter + "## Section N. ..." headers). We
chunk on that structure first, and only fall back to fixed-size splitting
within a section if a single section is too large to embed as one chunk.
This keeps a chunk's meaning intact and keeps citations pointing at real,
recognizable sections instead of arbitrary token windows.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from ragcore.models import Chunk, ChunkMetadata

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_HEADER_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)

# Rough tokens-per-word heuristic; avoids pulling in a tokenizer dependency
# for a small, English-only corpus. If the corpus grows or goes
# multilingual this should switch to a real tokenizer (see DESIGN.md).
_WORDS_PER_TOKEN = 0.75


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    category: str
    source_path: str
    sections: list[tuple[str, str]]  # (heading, body_text)


def parse_markdown_file(path: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        raise ValueError(f"{path} is missing YAML frontmatter")
    front = yaml.safe_load(match.group(1))
    body = match.group(2).strip()

    # Split body into (heading, text) pairs on "## Section ..." headers.
    headers = list(_HEADER_RE.finditer(body))
    sections: list[tuple[str, str]] = []
    if not headers:
        sections.append(("Document", body))
    else:
        for i, h in enumerate(headers):
            heading = h.group(1).strip()
            start = h.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
            sections.append((heading, body[start:end].strip()))

    return ParsedDocument(
        doc_id=front["doc_id"],
        title=front["title"],
        category=front.get("category", "Uncategorized"),
        source_path=str(path),
        sections=sections,
    )


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) / _WORDS_PER_TOKEN)


def _split_long_section(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Fallback fixed-size splitter, only used when a single section
    exceeds max_tokens. Splits on paragraph boundaries where possible so
    we still avoid cutting mid-sentence when we can.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_tokens(para)
        if current and current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            # carry the tail of the previous chunk forward as overlap
            overlap_words = " ".join(current).split()[-overlap_tokens:]
            current = [" ".join(overlap_words)] if overlap_words else []
            current_tokens = _approx_tokens(" ".join(current))
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]


def chunk_document(
    doc: ParsedDocument, max_chunk_tokens: int, chunk_overlap_tokens: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    for heading, section_text in doc.sections:
        if not section_text:
            continue
        pieces = (
            [section_text]
            if _approx_tokens(section_text) <= max_chunk_tokens
            else _split_long_section(
                section_text, max_chunk_tokens, chunk_overlap_tokens
            )
        )
        for piece in pieces:
            full_text = f"{doc.title} — {heading}\n\n{piece}"
            chunk_id = _stable_chunk_id(doc.doc_id, chunk_index, piece)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    metadata=ChunkMetadata(
                        doc_id=doc.doc_id,
                        title=doc.title,
                        category=doc.category,
                        section_heading=heading,
                        chunk_index=chunk_index,
                        source_path=doc.source_path,
                        text=full_text,
                    ),
                )
            )
            chunk_index += 1
    return chunks


def _stable_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Deterministic id: same document content -> same id -> Pinecone
    upsert overwrites in place instead of creating a duplicate vector.
    This is the core of the ingestion pipeline's idempotency (see
    ingestion/manifest.py for the file-level skip-if-unchanged layer on
    top of this).
    """
    normalized = " ".join(text.split())
    digest = hashlib.sha256(f"{doc_id}:{chunk_index}:{normalized}".encode()).hexdigest()
    return f"{doc_id}-{chunk_index}-{digest[:12]}"
