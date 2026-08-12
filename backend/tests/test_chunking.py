from pathlib import Path

import pytest
from ragcore.chunking import parse_html_file


def test_parse_html_file_extracts_main_content_and_heading_sections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clinical_guidance.html"
    path.write_text(
        """
        <html>
          <head><title>Fallback title</title></head>
          <body>
            <nav>Navigation text that must not be indexed</nav>
            <main>
              <h1>Clinical Guidance</h1>
              <h2>Overview</h2>
              <p>Use this document for clinical education.</p>
              <h2>Prevention</h2>
              <h3>Hand hygiene</h3>
              <p>Clean hands before and after patient contact.</p>
              <script>Do not index this script.</script>
            </main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    document = parse_html_file(path)

    assert document.doc_id == "clinical-guidance"
    assert document.title == "Clinical Guidance"
    assert document.category == "Health guidance"
    assert document.sections == [
        ("Overview", "Use this document for clinical education."),
        (
            "Prevention",
            "Hand hygiene\n\nClean hands before and after patient contact.",
        ),
    ]


def test_parse_html_file_falls_back_to_article_content(tmp_path: Path) -> None:
    path = tmp_path / "health_update.html"
    path.write_text(
        """
        <html>
          <head><title>Health Update</title></head>
          <body>
            <header>Site header</header>
            <article><h2>Recommendations</h2><p>Review current guidance.</p></article>
            <footer>Site footer</footer>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    document = parse_html_file(path)

    assert document.title == "Health Update"
    assert document.sections == [("Recommendations", "Review current guidance.")]


def test_parse_html_file_uses_the_document_title_not_svg_titles(tmp_path: Path) -> None:
    path = tmp_path / "screening.html"
    path.write_text(
        """
        <html>
          <head><title>Screening Guidance</title></head>
          <body>
            <article><h2>Recommendation</h2><p>Offer evidence-based screening.</p>
            </article>
            <svg><title>Print</title></svg>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    document = parse_html_file(path)

    assert document.title == "Screening Guidance"


def test_parse_html_file_prefers_named_content_over_earlier_articles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "drug_label.html"
    path.write_text(
        """
        <html>
          <head><title>Drug Label</title></head>
          <body>
            <article><h2>Site news</h2><p>Do not index this content.</p></article>
            <div id="drug-information">
              <h2>Drug information</h2><p>Use as directed in the label.</p>
            </div>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    document = parse_html_file(path)

    assert document.sections == [("Drug information", "Use as directed in the label.")]


def test_parse_html_file_preserves_tables_and_collapsed_content(tmp_path: Path) -> None:
    path = tmp_path / "screening_guidance.html"
    path.write_text(
        """
        <html>
          <head><title>Screening Guidance</title></head>
          <body>
            <main>
              <h1>Screening Guidance</h1>
              <h2>Recommendations</h2>
              <table>
                <caption>Screening intervals</caption>
                <thead><tr><th>Age group</th><th>Recommendation</th></tr></thead>
                <tbody><tr><td>40 to 74</td><td>Screen every two years.</td></tr></tbody>
              </table>
              <details><summary>Implementation notes</summary><p>Discuss risks and benefits.</p>
              </details>
              <button data-bs-toggle="collapse">Exceptions</button>
              <div class="collapse"><p>Use clinical judgment for higher-risk patients.</p></div>
              <select><option>Annual</option><option>Every two years</option></select>
            </main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    document = parse_html_file(path)

    assert document.sections == [
        (
            "Recommendations",
            "Table: Screening intervals\n"
            "\nAge group: 40 to 74 | Recommendation: Screen every two years.\n\n"
            "Details: Implementation notes\n\n"
            "Discuss risks and benefits.\n\n"
            "Details: Exceptions\n\n"
            "Use clinical judgment for higher-risk patients.\n\n"
            "Options: Annual; Every two years",
        )
    ]


def test_parse_html_file_rejects_non_html_files(tmp_path: Path) -> None:
    path = tmp_path / "legacy.md"
    path.write_text("# Old document", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected an HTML file"):
        parse_html_file(path)
