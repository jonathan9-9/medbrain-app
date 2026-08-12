"""HTML parsing and section-aware chunking for the curated health corpus.

Documents are hand-curated HTML pages from authoritative health sources.
Their heading structure provides the primary chunk boundaries; only oversized
sections fall back to fixed-size splitting. This keeps citations attached to
recognizable document sections instead of arbitrary token windows.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, PageElement, Tag

from ragcore.models import Chunk, ChunkMetadata

_HEADING_TAGS = {f"h{level}" for level in range(1, 7)}
_HEADING_MARKER_RE = re.compile(r"^\[\[HEADING:(\d)\]\]\s*(.*)$")

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


def parse_html_file(path: Path) -> ParsedDocument:
    """Parse a curated HTML document into title and heading-based sections."""
    if path.suffix.lower() != ".html":
        raise ValueError(f"Expected an HTML file, got {path}")

    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    content = _content_root(soup)
    if content is None:
        raise ValueError(f"{path} does not contain extractable document content")

    title = _first_text(content.find("h1")) or _first_text(soup.title)
    sections = _html_sections(content)
    if not sections:
        raise ValueError(f"{path} does not contain extractable document content")

    return ParsedDocument(
        doc_id=_document_id(path),
        title=title or path.stem.replace("_", " "),
        category="Health guidance",
        source_path=str(path),
        sections=sections,
    )


def _content_root(soup: BeautifulSoup) -> Tag | None:
    for element in (
        soup.find("main"),
        soup.find(id="drug-information"),
        soup.find("article"),
        soup.body,
    ):
        if isinstance(element, Tag):
            return element
    return None


def _html_sections(content: Tag) -> list[tuple[str, str]]:
    for element in content.find_all(["script", "style", "noscript", "svg", "template"]):
        element.decompose()
    for table in reversed(content.find_all("table")):
        table.replace_with(NavigableString(f"\n{_render_table(table)}\n"))
    for summary in content.find_all("summary"):
        label = _first_text(summary)
        summary.replace_with(NavigableString(f"\nDetails: {label}\n" if label else ""))
    for select in content.find_all("select"):
        options = _unique([_first_text(option) for option in select.find_all("option")])
        text = f"\nOptions: {'; '.join(options)}\n" if options else ""
        select.replace_with(NavigableString(text))
    for button in content.find_all("button"):
        if (
            button.get("data-bs-toggle") in {"collapse", "dropdown"}
            and button.find_parent(list(_HEADING_TAGS)) is None
        ):
            label = _first_text(button)
            button.replace_with(
                NavigableString(f"\nDetails: {label}\n" if label else "")
            )

    for heading in content.find_all(list(_HEADING_TAGS)):
        label = _first_text(heading)
        level = int(heading.name[1])
        marker = f"\n[[HEADING:{level}]] {label}\n" if label else ""
        heading.replace_with(NavigableString(marker))

    return _sections_from_html_text(content.get_text("\n"))


def _sections_from_html_text(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Document"
    current_parts: list[str] = []
    saw_primary_section = False

    def append_current() -> None:
        if current_parts:
            sections.append((current_heading, current_parts.copy()))

    for line in (_clean_text(line) for line in text.splitlines()):
        if not line:
            continue
        heading = _HEADING_MARKER_RE.match(line)
        if heading is None:
            current_parts.append(line)
            continue
        level = int(heading.group(1))
        label = heading.group(2)
        if level == 1:
            continue
        if level == 2:
            append_current()
            current_heading = label
            current_parts = []
            saw_primary_section = True
        else:
            if saw_primary_section:
                current_parts.append(label)
            else:
                append_current()
                current_heading = label
                current_parts = []
    append_current()

    return [(heading, "\n\n".join(parts)) for heading, parts in sections if parts]


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _first_text(element: PageElement | None) -> str:
    if element is None:
        return ""
    if isinstance(element, NavigableString):
        return _clean_text(str(element))
    return _clean_text(element.get_text(" ", strip=True))


def _render_table(table: Tag) -> str:
    caption = _first_text(table.find("caption"))
    lines = [f"Table: {caption}" if caption else "Table"]
    rows = table.find_all("tr")
    header_row = next((row for row in rows if row.find("th")), None)
    headers = _table_cells(header_row) if header_row else []
    for row in rows:
        if row is header_row:
            continue
        cells = []
        for index, cell in enumerate(_table_cells(row)):
            header = headers[index] if index < len(headers) else f"Column {index + 1}"
            cells.append(f"{header}: {cell}")
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines) if len(lines) > 1 else ""


def _table_cells(row: Tag | None) -> list[str]:
    if row is None:
        return []
    return [_first_text(cell) for cell in row.find_all(["td", "th"], recursive=False)]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _document_id(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


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
