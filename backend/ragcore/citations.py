"""Maps [S1], [S2]... tags the model used in its answer text to the real
retrieved-chunk metadata. This is what makes citations "real, not
decorative": the model only ever chooses WHICH retrieved chunk index to
point at; the actual doc title / section / source path shown to the user
always comes from what was actually retrieved, never from model output.
"""

from __future__ import annotations

import re

from ragcore.models import Citation, RetrievedChunk

_TAG_RE = re.compile(r"\[S(\d+)\]")


def extract_used_citations(
    answer_text: str, retrieved: list[RetrievedChunk]
) -> list[Citation]:
    used_indices = sorted({int(m) for m in _TAG_RE.findall(answer_text)})
    citations: list[Citation] = []
    for idx in used_indices:
        pos = idx - 1  # tags are 1-indexed ([S1] -> retrieved[0])
        if 0 <= pos < len(retrieved):
            chunk = retrieved[pos]
            citations.append(
                Citation(
                    tag=f"S{idx}",
                    doc_id=chunk.metadata.doc_id,
                    title=chunk.metadata.title,
                    section_heading=chunk.metadata.section_heading,
                    source_path=chunk.metadata.source_path,
                )
            )
        # Tags that don't resolve to a retrieved chunk are dropped rather
        # than fabricated -- a grounding failure should shrink the
        # citation list, never invent a fake source.
    return citations


def answer_has_ungrounded_tags(
    answer_text: str, retrieved: list[RetrievedChunk]
) -> bool:
    """True if the model referenced a source index that wasn't actually
    retrieved -- a signal worth logging/monitoring even though we don't
    block the response on it (blocking would mean silently discarding an
    otherwise-good answer over one bad tag).
    """
    used_indices = {int(m) for m in _TAG_RE.findall(answer_text)}
    return any(idx < 1 or idx > len(retrieved) for idx in used_indices)
