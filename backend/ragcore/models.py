"""Shared data models. Both ingestion (producing chunks) and the backend
(consuming retrieved chunks) import from here so the chunk metadata shape
can never silently diverge between what was indexed and what the API
expects back.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata stored alongside every vector in Pinecone."""

    doc_id: str
    title: str
    category: str
    section_heading: str
    chunk_index: int
    source_path: str
    text: str


class Chunk(BaseModel):
    """A chunk produced by the ingestion pipeline, ready to embed."""

    id: str  # deterministic hash-based id -> makes upsert idempotent
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    id: str
    score: float
    metadata: ChunkMetadata


class Citation(BaseModel):
    """A citation surfaced to the frontend. tag is what the model used
    inline (e.g. "S1"); everything else is backend-owned ground truth
    looked up from the retrieved chunk, so the model cannot fabricate
    where a citation actually points.
    """

    tag: str
    doc_id: str
    title: str
    section_heading: str
    source_path: str


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    UNANSWERABLE = "unanswerable"
    REFUSED_MEDICAL_ADVICE = "refused_medical_advice"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ChatMessageEvent(BaseModel):
    """One SSE event. `type` tells the frontend how to render it."""

    type: str  # "token" | "citations" | "status" | "error" | "done"
    data: str | list[Citation] | None = None
