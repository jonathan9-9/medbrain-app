from pathlib import Path

import pytest

from ragcore.ingestion.sources import html_source_files


def test_html_source_files_recursively_finds_only_html_files(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (tmp_path / "root.html").touch()
    (nested_dir / "nested.HTML").touch()
    (nested_dir / "notes.md").touch()

    assert html_source_files(tmp_path) == [
        nested_dir / "nested.HTML",
        tmp_path / "root.html",
    ]


def test_html_source_files_rejects_missing_corpus_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Corpus directory does not exist"):
        html_source_files(tmp_path / "missing")
