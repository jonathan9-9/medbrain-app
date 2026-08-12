"""Orchestrates the ingestion pipeline. Re-running `run()` on an
unchanged corpus is a near-no-op: every source file's hash is checked
against the manifest before any embedding call is made, and Pinecone
upserts are keyed on deterministic chunk ids so even a forced re-run
overwrites in place instead of duplicating (see ragcore/chunking.py and
ingestion/manifest.py for the two layers that make this true).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ragcore.ingestion.manifest import Manifest, file_hash
from ragcore.ingestion.sources import html_source_files
from ragcore.chunking import chunk_document, parse_html_file
from ragcore.config import Settings
from ragcore.embeddings import EmbeddingClient
from ragcore.vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    files_scanned: int = 0
    files_unchanged_skipped: int = 0
    files_ingested: int = 0
    files_removed: int = 0
    chunks_upserted: int = 0


def run(
    settings: Settings,
    corpus_dir: Path,
    manifest_path: Path,
    force: bool = False,
    dry_run: bool = False,
) -> IngestionResult:
    manifest = Manifest(manifest_path)
    result = IngestionResult()

    source_files = html_source_files(corpus_dir)
    seen_paths: set[str] = set()

    embedder = None
    store = None
    if not dry_run:
        embedder = EmbeddingClient(settings)
        store = VectorStore(settings)

    for path in source_files:
        result.files_scanned += 1
        str_path = str(path)
        seen_paths.add(str_path)
        current_hash = file_hash(path)
        existing = manifest.get(str_path)

        if existing and existing.content_hash == current_hash and not force:
            result.files_unchanged_skipped += 1
            logger.info("SKIP (unchanged): %s", path.name)
            continue

        logger.info("INGEST: %s", path.name)
        doc = parse_html_file(path)
        chunks = chunk_document(
            doc, settings.max_chunk_tokens, settings.chunk_overlap_tokens
        )

        if dry_run:
            logger.info("  [dry-run] would upsert %d chunks", len(chunks))
            result.files_ingested += 1
            result.chunks_upserted += len(chunks)
            continue

        assert embedder is not None and store is not None

        # If this file was previously ingested under a different chunking
        # (e.g. section headers changed), stale chunk ids from the old
        # version won't be overwritten by the new ones -- explicitly
        # delete the old set first so the index converges to exactly the
        # current content, not a superset of old + new.
        if existing:
            store.delete_ids(existing.chunk_ids)

        embeddings = [embedder.embed_document(c.metadata.text) for c in chunks]
        upserted = store.upsert_chunks(chunks, embeddings)
        result.chunks_upserted += upserted
        result.files_ingested += 1

        manifest.record(str_path, current_hash, [c.id for c in chunks])

    # Prune files that used to be in the corpus but no longer are.
    for known_path in manifest.known_paths() - seen_paths:
        logger.info("REMOVE (deleted from corpus): %s", known_path)
        entry = manifest.remove(known_path)
        if entry and not dry_run:
            assert store is not None
            store.delete_ids(entry.chunk_ids)
        result.files_removed += 1

    if not dry_run:
        manifest.save()

    return result
