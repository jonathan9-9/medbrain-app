"""Pinecone serverless wrapper. Upserts are keyed on the deterministic
chunk id from ragcore.chunking, which is what makes re-running ingestion
idempotent (same content -> same id -> overwrite, not a duplicate).
"""

from __future__ import annotations

import logging
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from ragcore.config import Settings
from ragcore.models import Chunk, ChunkMetadata, RetrievedChunk

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 100


class VectorStore:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._ensure_index()
        self._index = self._pc.Index(settings.pinecone_index_name)

    def _ensure_index(self) -> None:
        existing = {idx["name"] for idx in self._pc.list_indexes()}
        if self._settings.pinecone_index_name not in existing:
            logger.info(
                "creating Pinecone index %s", self._settings.pinecone_index_name
            )
            self._pc.create_index(
                name=self._settings.pinecone_index_name,
                dimension=self._settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self._settings.pinecone_cloud,
                    region=self._settings.pinecone_region,
                ),
            )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        vectors = [
            {
                "id": chunk.id,
                "values": vector,
                "metadata": chunk.metadata.model_dump(),
            }
            for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
        count = 0
        for i in range(0, len(vectors), _UPSERT_BATCH_SIZE):
            batch = vectors[i : i + _UPSERT_BATCH_SIZE]
            self._index.upsert(vectors=batch)
            count += len(batch)
        return count

    def delete_by_doc_id(self, doc_id: str) -> None:
        self._index.delete(filter={"doc_id": {"$eq": doc_id}})

    def delete_ids(self, ids: list[str]) -> None:
        if ids:
            self._index.delete(ids=ids)

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        result = self._index.query(vector=embedding, top_k=top_k, include_metadata=True)
        retrieved: list[RetrievedChunk] = []
        for match in result.get("matches", []):
            retrieved.append(
                RetrievedChunk(
                    id=match["id"],
                    score=match["score"],
                    metadata=ChunkMetadata(**match["metadata"]),
                )
            )
        return retrieved

    def stats(self) -> Any:
        return self._index.describe_index_stats()
