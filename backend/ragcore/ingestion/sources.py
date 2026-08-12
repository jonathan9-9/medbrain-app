"""Corpus source-file discovery."""

from pathlib import Path


def html_source_files(corpus_dir: Path) -> list[Path]:
    """Return every HTML file in the corpus, including nested directories."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")
    return sorted(
        path
        for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".html"
    )
