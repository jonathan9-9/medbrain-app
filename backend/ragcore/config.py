"""Centralized, typed settings. Every secret/config value is read from the
environment exactly once, here, so nothing else in the codebase reaches
into os.environ directly. This is what keeps API keys server-side and
makes the app configurable per-deployment without code changes.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Secrets (server-side only; never sent to the frontend) ---
    gemini_api_key: str = Field(default="")
    pinecone_api_key: str = Field(default="")

    # --- Pinecone ---
    pinecone_index_name: str = Field(default="medbrain-index")
    pinecone_cloud: str = Field(default="aws")
    pinecone_region: str = Field(default="us-east-1")
    # text-embedding-001 outputs 1536-dim vectors.
    embedding_dimension: int = Field(default=1536)

    # --- Models ---
    gemini_generation_model: str = Field(default="gemini-2.0-flash")
    gemini_embedding_model: str = Field(default="text-embedding-001")

    # --- Chunking ---
    max_chunk_tokens: int = Field(default=700)
    chunk_overlap_tokens: int = Field(default=100)

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=6)
    # Cosine similarity below this on the top hit -> treat as "not covered
    # by the corpus" rather than let the model try to answer anyway.
    min_retrieval_score: float = Field(default=0.55)

    # --- Corpus / manifest paths ---
    corpus_dir: str = Field(default="corpus/raw")
    manifest_path: str = Field(default="ingestion/.manifest/ingestion_manifest.json")

    # --- CORS ---
    allowed_origins: str = Field(default="http://localhost:3000")

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
