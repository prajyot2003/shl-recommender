"""Runtime configuration, all env-driven so deploys need no code changes."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Live catalog JSON. If unset/unreachable, loader falls back to snapshot.
    catalog_url: str = os.getenv(
        "SHL_CATALOG_URL",
        "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json",
    )
    use_semantic: bool = os.getenv("SHL_USE_SEMANTIC", "0") == "1"

    # LLM (OpenAI-compatible endpoint: Groq, OpenRouter, OpenAI, local, ...).
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "20"))

    max_turns: int = int(os.getenv("MAX_TURNS", "8"))
    candidate_pool_size: int = int(os.getenv("CANDIDATE_POOL_SIZE", "20"))
    max_recommendations: int = int(os.getenv("MAX_RECOMMENDATIONS", "10"))


settings = Settings()
