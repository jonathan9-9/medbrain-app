from ragcore.prompts import SYSTEM_INSTRUCTIONS, UNANSWERABLE_TEMPLATE


def test_prompt_describes_curated_health_documents() -> None:
    assert "hand-curated" in SYSTEM_INSTRUCTIONS
    assert "authoritative health guidance" in SYSTEM_INSTRUCTIONS
    assert "SOPs" not in SYSTEM_INSTRUCTIONS
    assert "hand-curated health documents" in UNANSWERABLE_TEMPLATE
