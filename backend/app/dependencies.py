from __future__ import annotations

from functools import lru_cache

from ragcore.config import get_settings
from ragcore.embeddings import EmbeddingClient
from ragcore.llm import GenerationClient
from ragcore.vectorstore import VectorStore

from app.services.chat_service import ChatService


@lru_cache
def get_embedder() -> EmbeddingClient:
    return EmbeddingClient(get_settings())


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(get_settings())


@lru_cache
def get_generator() -> GenerationClient:
    return GenerationClient(get_settings())


def get_chat_service() -> ChatService:
    # NOTE: deliberately takes no parameters. FastAPI inspects the
    # signature of any callable passed to Depends() to resolve its own
    # sub-dependencies; a plain `Settings` parameter here (even with a
    # None default) gets misread as a second request-body model, which
    # breaks the /chat route's request parsing. Settings is a singleton
    # via get_settings()'s lru_cache, so there's no need to inject it.
    return ChatService(
        settings=get_settings(),
        embedder=get_embedder(),
        store=get_vector_store(),
        generator=get_generator(),
    )
