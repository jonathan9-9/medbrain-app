"""Tracks a content hash and the set of chunk ids produced per source
file. This is what lets the pipeline (a) skip re-embedding files that
haven't changed, and (b) prune vectors for files that were deleted or
whose chunking changed -- "idempotent" means convergence to the correct
state, not just "won't duplicate."
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ManifestEntry:
    content_hash: str
    chunk_ids: list[str] = field(default_factory=list)


class Manifest:
    def __init__(self, path: Path):
        self._path = path
        self._entries: dict[str, ManifestEntry] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            self._entries = {k: ManifestEntry(**v) for k, v in raw.items()}

    def get(self, source_path: str) -> ManifestEntry | None:
        return self._entries.get(source_path)

    def record(self, source_path: str, content_hash: str, chunk_ids: list[str]) -> None:
        self._entries[source_path] = ManifestEntry(content_hash, chunk_ids)

    def remove(self, source_path: str) -> ManifestEntry | None:
        return self._entries.pop(source_path, None)

    def known_paths(self) -> set[str]:
        return set(self._entries.keys())

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            k: {"content_hash": v.content_hash, "chunk_ids": v.chunk_ids}
            for k, v in self._entries.items()
        }
        self._path.write_text(json.dumps(serializable, indent=2, sort_keys=True))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
