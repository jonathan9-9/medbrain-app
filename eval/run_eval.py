"""Runs the eval dataset through the exact same ChatService used by the
live API, scores retrieval and answer quality, prints a console report,
and writes eval/report.json.

Usage:
    make eval
    python -m eval.run_eval

Requires real GEMINI_API_KEY / PINECONE_API_KEY in backend/.env and an
already-ingested corpus.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Make both `ragcore` (repo root) and `app` (backend/) importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _REPO_ROOT / "backend"

for _path in (_REPO_ROOT, _BACKEND_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from backend.app.dependencies import get_chat_service  # noqa: E402
from backend.ragcore.config import get_settings  # noqa: E402
from backend.ragcore.llm import GenerationClient  # noqa: E402
from backend.ragcore.models import AnswerStatus, RetrievedChunk  # noqa: E402
from eval.dataset import load_dataset  # noqa: E402
from eval.judge import judge_answer  # noqa: E402
from eval.scoring import (  # noqa: E402
    EXPECTED_STATUS_BY_CATEGORY,
    CaseResult,
    aggregate_report,
)

logging.basicConfig(level=logging.WARNING)


# Representative answer-quality sample:
# - 2 standard factual cases
# - 2 required multi-document synthesis cases
#
# Status and retrieval are still evaluated for all cases.
JUDGE_CASE_IDS = {
    "q01",
    "q04",
    "q11",
    "q12",
}


def run() -> None:
    settings = get_settings()

    dataset_path = _REPO_ROOT / "eval" / "dataset.yaml"
    cases = load_dataset(dataset_path)

    chat_service = get_chat_service()
    judge_generator = GenerationClient(settings)

    results: list[CaseResult] = []

    for case in cases:
        print(f"[{case.id}] ({case.category.value}) {case.question[:70]}...")

        retrieved_chunks: list[RetrievedChunk] = []

        try:
            # Run the exact same ChatService path as production.
            #
            # The callback lets us reuse the chunks retrieved by ChatService
            # for the LLM judge without performing another embedding call or
            # Pinecone query.
            events = list(
                chat_service.answer(
                    case.question,
                    on_retrieval=retrieved_chunks.extend,
                )
            )

        except Exception as exc:  # noqa: BLE001
            error_message = f"{type(exc).__name__}: {exc}"

            print(f"  ERROR: {error_message}")

            results.append(
                CaseResult(
                    case_id=case.id,
                    category=case.category,
                    expected_status=EXPECTED_STATUS_BY_CATEGORY[case.category],
                    actual_status=None,
                    retrieved_doc_ids=[
                        chunk.metadata.doc_id for chunk in retrieved_chunks
                    ],
                    expected_doc_ids=case.expected_source_doc_ids,
                    error=error_message,
                )
            )

            # Important: do not count this case as a status failure.
            # It was not successfully evaluated.
            continue

        status_event = next(
            (event for event in events if event.type == "status"),
            None,
        )

        actual_status = AnswerStatus(status_event.data) if status_event else None

        retrieval_event = next(
            (event for event in events if event.type == "retrieval"),
            None,
        )

        retrieved_doc_ids: list[str] = (
            retrieval_event.data
            if retrieval_event
            else [chunk.metadata.doc_id for chunk in retrieved_chunks]
        )

        answer_text = "".join(event.data for event in events if event.type == "token")

        result = CaseResult(
            case_id=case.id,
            category=case.category,
            expected_status=EXPECTED_STATUS_BY_CATEGORY[case.category],
            actual_status=actual_status,
            retrieved_doc_ids=retrieved_doc_ids,
            expected_doc_ids=case.expected_source_doc_ids,
        )

        citations_event = next(
            (event for event in events if event.type == "citations"),
            None,
        )

        result.has_citations = bool(citations_event and citations_event.data)

        # Only invoke the LLM judge for the selected representative cases.
        if (
            case.id in JUDGE_CASE_IDS
            and case.expected_answer_summary
            and actual_status == AnswerStatus.ANSWERED
            and result.status_correct
        ):
            groundedness, correctness, notes = judge_answer(
                generator=judge_generator,
                question=case.question,
                answer=answer_text,
                expected_summary=case.expected_answer_summary,
                retrieved=retrieved_chunks,
            )

            result.groundedness_score = groundedness
            result.correctness_score = correctness
            result.judge_notes = notes

        results.append(result)

    report = aggregate_report(results)

    _print_report(report, results)
    _save_report(report, results)


def _print_report(report, results) -> None:
    print("\n" + "=" * 72)
    print("EVAL REPORT")
    print("=" * 72)

    print(f"Total cases:              {report.total_cases}")
    print(f"Successfully evaluated:   {report.evaluated_cases}")
    print(f"Evaluation errors:        {report.error_count}")

    if report.overall_status_accuracy is not None:
        print(f"Status accuracy:          {report.overall_status_accuracy:.1%}")
    else:
        print("Status accuracy:          N/A")

    if report.overall_mrr is not None:
        print(f"Overall retrieval MRR:   {report.overall_mrr:.3f}")

    print(f"LLM-judged cases:         {report.judged_case_count}")

    if report.judged_case_ids:
        print("Judge case IDs:           " + ", ".join(report.judged_case_ids))

    print()

    header = (
        f"{'category':<24}"
        f"{'n':>4}"
        f"{'eval':>7}"
        f"{'errors':>8}"
        f"{'status_acc':>12}"
        f"{'hit_rate':>10}"
        f"{'mrr':>8}"
        f"{'ground':>8}"
        f"{'correct':>9}"
    )

    print(header)
    print("-" * len(header))

    for category in report.categories:
        status_accuracy = (
            f"{category.status_accuracy:.0%}"
            if category.status_accuracy is not None
            else "-"
        )

        hit_rate = f"{category.hit_rate:.3f}" if category.hit_rate is not None else "-"

        mrr = f"{category.mrr:.3f}" if category.mrr is not None else "-"

        grounded = (
            f"{category.avg_groundedness:.2f}"
            if category.avg_groundedness is not None
            else "-"
        )

        correct = (
            f"{category.avg_correctness:.2f}"
            if category.avg_correctness is not None
            else "-"
        )

        print(
            f"{category.category:<24}"
            f"{category.n:>4}"
            f"{category.evaluated_n:>7}"
            f"{category.error_count:>8}"
            f"{status_accuracy:>11} "
            f"{hit_rate:>9} "
            f"{mrr:>7} "
            f"{grounded:>7} "
            f"{correct:>8}"
        )

    print()

    if report.failures:
        print(
            f"FAILED status checks ({len(report.failures)}): "
            f"{', '.join(report.failures)}"
        )

        for result in results:
            if not result.evaluation_error and not result.status_correct:
                actual = (
                    result.actual_status.value
                    if result.actual_status is not None
                    else "none"
                )

                print(
                    f"  - {result.case_id}: expected "
                    f"{result.expected_status.value}, "
                    f"got {actual}"
                )
    else:
        print("No successfully evaluated cases failed their expected status.")

    if report.errors:
        print(f"\nEVALUATION ERRORS ({len(report.errors)}): {', '.join(report.errors)}")

        for result in results:
            if result.evaluation_error:
                print(f"  - {result.case_id}: {result.error}")

    print("=" * 72)


def _save_report(report, results) -> None:
    out_path = _REPO_ROOT / "eval" / "report.json"

    payload = {
        "summary": json.loads(report.model_dump_json()),
        "cases": [
            {
                "id": result.case_id,
                "category": result.category.value,
                "expected_status": result.expected_status.value,
                "actual_status": (
                    result.actual_status.value
                    if result.actual_status is not None
                    else None
                ),
                "status_correct": (
                    result.status_correct if not result.evaluation_error else None
                ),
                "evaluation_error": result.evaluation_error,
                "error": result.error,
                "retrieved_doc_ids": result.retrieved_doc_ids,
                "expected_doc_ids": result.expected_doc_ids,
                "hit": (result.hit if not result.evaluation_error else None),
                "partial_hit": (
                    result.partial_hit if not result.evaluation_error else None
                ),
                "reciprocal_rank": (result.rr if not result.evaluation_error else None),
                "has_citations": result.has_citations,
                "groundedness_score": result.groundedness_score,
                "correctness_score": result.correctness_score,
                "judge_notes": result.judge_notes,
            }
            for result in results
        ],
    }

    out_path.write_text(json.dumps(payload, indent=2))

    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    run()
