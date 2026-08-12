"""CLI entrypoint: `python -m ragcore.ingestion.run [--force] [--dry-run]`"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ragcore.ingestion.pipeline import run
from ragcore.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the document corpus into Pinecone."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-embed all files even if unchanged."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk without calling embedding/Pinecone APIs.",
    )
    args = parser.parse_args()

    settings = get_settings()
    result = run(
        settings=settings,
        corpus_dir=_project_path(settings.corpus_dir),
        manifest_path=_project_path(settings.manifest_path),
        force=args.force,
        dry_run=args.dry_run,
    )

    print("\n--- Ingestion Summary ---")
    print(f"Files scanned:           {result.files_scanned}")
    print(f"Files unchanged/skipped: {result.files_unchanged_skipped}")
    print(f"Files ingested:          {result.files_ingested}")
    print(f"Files removed:           {result.files_removed}")
    print(f"Chunks upserted:         {result.chunks_upserted}")


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
