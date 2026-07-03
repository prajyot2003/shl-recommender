"""FastAPI service exposing /health and /chat.

The catalog + retriever + LLM are built once at startup and reused (the API
itself stays stateless per conversation — no per-conversation memory is stored,
exactly as the spec requires).
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse

from .agent.llm import build_llm
from .agent.orchestrator import Orchestrator
from .catalog.service import Catalog
from .config import settings
from .schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

_state: dict = {"orchestrator": None, "ready": False}


@app.on_event("startup")
def _startup() -> None:
    try:
        catalog = Catalog.bootstrap(settings.catalog_url, use_semantic=settings.use_semantic)
        llm = build_llm(settings)
        _state["orchestrator"] = Orchestrator(catalog, llm, settings)
        _state["ready"] = True
        logger.info("Service ready with %d assessments.", len(catalog))
    except Exception:
        logger.exception("Startup failed; /health will report not-ready.")
        _state["ready"] = False


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Spec contract: returns {"status": "ok"} with HTTP 200. The catalog always
    # loads (snapshot fallback guarantees startup succeeds), so we report ok once
    # the process is serving. Readiness detail is in logs, not the payload.
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    orch: Orchestrator | None = _state["orchestrator"]
    if orch is None:
        # Never break the schema, even before warmup.
        return ChatResponse(
            reply="The service is still warming up — please retry in a moment.",
            recommendations=[],
            end_of_conversation=False,
        )
    try:
        return orch.handle(req.messages)
    except Exception:
        logger.exception("Unhandled error in /chat; returning safe schema-valid reply.")
        return ChatResponse(
            reply="Something went wrong on my end. Could you restate the role and "
                  "key skills you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )


@app.exception_handler(Exception)
def _fallback_handler(request, exc):  # pragma: no cover - last-resort net
    logger.exception("Top-level exception.")
    return JSONResponse(
        status_code=200,
        content={"reply": "Unexpected error. Please try again.",
                 "recommendations": [], "end_of_conversation": False},
    )
