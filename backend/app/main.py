from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ragcore.config import get_settings

from app.routers import chat, health

settings = get_settings()

app = FastAPI(
    title="Clinical Ops Document Information Retrieval API",
    description=(
        "Grounded, cited Q&A over clinical operations document corpus. "
        "Not a source of personal medical advice."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(chat.router, tags=["chat"])
