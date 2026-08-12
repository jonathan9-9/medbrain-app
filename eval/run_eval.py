"""Runs the eval dataset through the exact same ChatService used by the
live API (imported from the backend, not reimplemented), scores retrieval
and answer quality, prints a console report, and writes eval/report.json.

Usage: `make eval` (see Makefile) or `python -m eval.run_eval`
Requires real GEMINI_API_KEY / PINECONE_API_KEY in the environment and an
already-ingested corpus (`python -m ragcore.ingestion.run` first).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Make both `ragcore` (repo root) and `app` (backend/) importable, matching
# the same sys.path pattern used by backend/api/index.py and
# backend/tests/conftest.py -- one consistent import graph everywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _REPO_ROOT / "backend"
for _p in (_REPO_ROOT, _BACKEND_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from backend.app.dependencies import (  # noqa: E402
    get_chat_service,
    get_embedder,
    get_vector_store,
)
from backend.ragcore.config import get_settings  # noqa: E402
from backend.ragcore.llm import GenerationClient  # noqa: E402
from backend.ragcore.models import AnswerStatus  # noqa: E402
from eval.dataset import load_dataset  # noqa: E402
from eval.judge import judge_answer  # noqa: E402
from eval.scoring import (  # noqa: E402
    EXPECTED_STATUS_BY_CATEGORY,
    CaseResult,
    aggregate_report,
)

logging.basicConfig(level=logging.WARNING)  # keep eval output focused on the report


def run() -> None:
    settings = get_settings()
    dataset_path = _REPO_ROOT / "eval" / "dataset.yaml"
    cases = load_dataset(dataset_path)

    chat_service = get_chat_service()
    judge_generator = GenerationClient(settings)

    results: list[CaseResult] = []

    for case in cases:
        print(f"[{case.id}] ({case.category.value}) {case.question[:70]}...")
        events = list(chat_service.answer(case.question))

        status_event = next((e for e in events if e.type == "status"), None)
        actual_status = (
            AnswerStatus(status_event.data) if status_event else AnswerStatus.ANSWERED
        )
        retrieval_event = next((e for e in events if e.type == "retrieval"), None)
        retrieved_doc_ids: list[str] = retrieval_event.data if retrieval_event else []
        answer_text = "".join(e.data for e in events if e.type == "token")

        result = CaseResult(
            case_id=case.id,
            category=case.category,
            expected_status=EXPECTED_STATUS_BY_CATEGORY[case.category],
            actual_status=actual_status,
            retrieved_doc_ids=retrieved_doc_ids,
            expected_doc_ids=case.expected_source_doc_ids,
        )

        citations_event = next((e for e in events if e.type == "citations"), None)
        result.has_citations = bool(citations_event and citations_event.data)

        # Only run the LLM judge for cases that were actually supposed to
        # be answered AND were -- grading a refusal or an unanswerable
        # response against an "expected answer" summary doesn't make
        # sense, and those categories are graded far more strictly by the
        # deterministic status check above.
        if (
            case.expected_answer_summary
            and actual_status == AnswerStatus.ANSWERED
            and result.status_correct
        ):
            store = get_vector_store()
            query_embedding = get_embedder().embed_query(case.question)
            retrieved_chunks = store.query(
                query_embedding, top_k=settings.retrieval_top_k
            )
            groundedness, correctness, notes = judge_answer(
                settings=settings,
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
    print(f"Overall status accuracy: {report.overall_status_accuracy:.1%}")
    if report.overall_mrr is not None:
        print(f"Overall retrieval MRR:   {report.overall_mrr:.3f}")
    print()
    header = f"{'category':<24}{'n':>4}{'status_acc':>12}{'hit_rate':>10}{'mrr':>8}{'ground':>8}{'correct':>9}"
    print(header)
    print("-" * len(header))
    for c in report.categories:
        print(
            f"{c.category:<24}{c.n:>4}{c.status_accuracy:>11.0%} "
            f"{c.hit_rate if c.hit_rate is not None else '-':>9} "
            f"{c.mrr if c.mrr is not None else '-':>7} "
            f"{c.avg_groundedness if c.avg_groundedness is not None else '-':>7} "
            f"{c.avg_correctness if c.avg_correctness is not None else '-':>8}"
        )
    print()
    if report.failures:
        print(
            f"FAILED status checks ({len(report.failures)}): {', '.join(report.failures)}"
        )
        for r in results:
            if not r.status_correct:
                print(
                    f"  - {r.case_id}: expected {r.expected_status.value}, "
                    f"got {r.actual_status.value}"
                )
    else:
        print("All cases produced the expected status (answered/unanswerable/refused).")
    print("=" * 72)


def _save_report(report, results) -> None:
    out_path = _REPO_ROOT / "eval" / "report.json"
    payload = {
        "summary": json.loads(report.model_dump_json()),
        "cases": [
            {
                "id": r.case_id,
                "category": r.category.value,
                "expected_status": r.expected_status.value,
                "actual_status": r.actual_status.value,
                "status_correct": r.status_correct,
                "retrieved_doc_ids": r.retrieved_doc_ids,
                "expected_doc_ids": r.expected_doc_ids,
                "hit": r.hit,
                "reciprocal_rank": r.rr,
                "has_citations": r.has_citations,
                "groundedness_score": r.groundedness_score,
                "correctness_score": r.correctness_score,
                "judge_notes": r.judge_notes,
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    run()
