from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ragcore.models import ChatMessageEvent, ChatRequest

from app.dependencies import get_chat_service
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse_format(event: ChatMessageEvent) -> str:
    payload = event.model_dump()
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat")
def chat(
    request: ChatRequest, chat_service: ChatService = Depends(get_chat_service)
) -> StreamingResponse:
    def event_stream():
        try:
            for event in chat_service.answer(request.question):
                yield _sse_format(event)
        except Exception:
            # Never let a backend exception surface as a raw 500 mid-stream
            # with no explanation -- the frontend has an explicit "error"
            # event type it renders as a recoverable state (see
            # frontend/lib/api.ts), not a generic crash.
            logger.exception("chat stream failed")
            yield _sse_format(
                ChatMessageEvent(
                    type="error",
                    data=(
                        "Something went wrong generating this answer. "
                        "Please try again in a moment."
                    ),
                )
            )
            yield _sse_format(ChatMessageEvent(type="done", data=None))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering for streaming
        },
    )
