"""
FastAPI application entry point for the prompt injection detector.

Run from the prompt-injection-detector/ directory:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

OpenAPI docs available at http://localhost:8000/docs once running.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routes.detect import router as detect_router
from app.routes.health import router as health_router
from app.services.llm_judges.llm_judge import get_llm_judge
from app.services.rag_retriever import get_rag_retriever

# ── Logging ────────────────────────────────────────────────────────────────────
# Configured once here; every module uses logging.getLogger(__name__) which
# inherits this root configuration automatically.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-40s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: touch all @lru_cache singletons to pay initialisation costs
    before the first request arrives.

      get_rag_retriever() — loads SentenceTransformer weights (~200 ms)
      get_llm_judge()     — initialises Groq + Gemini clients (~50 ms)

    Shutdown: nothing to clean up — all singletons are process-scoped.
    """
    log.info("Starting up prompt-injection-detector …")

    get_rag_retriever()   # SentenceTransformer model load
    get_llm_judge()       # LLM client initialisation

    log.info("Startup complete. Services ready.")
    yield
    log.info("Shutting down.")


# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Prompt Injection Detector",
    description=(
        "Three-layer API for detecting prompt injection attacks before they reach an LLM.\n\n"
        "**Layer 1** — Rule-based (regex/keyword), ~0 ms, no AI.\n\n"
        "**Layer 2** — RAG retrieval (ChromaDB + all-MiniLM-L6-v2), ~100 ms.\n\n"
        "**Layer 3** — LLM ensemble (Groq llama-3.1-8b-instant + Gemini 2.0 Flash), ~1 s.\n\n"
        "Layer 1 short-circuits on a match — Layers 2 and 3 only run for ambiguous inputs."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Global exception handler ───────────────────────────────────────────────────


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Last-resort stack-trace barrier.

    Logs the full exception with traceback to the server logs and returns a
    clean JSON 500 to the caller — never exposes internal structure to clients.

    Individual route handlers have their own try/except; this handler catches
    anything that escapes all other layers.
    """
    log.exception(
        "Unhandled exception | method=%s | path=%s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Check server logs."},
    )


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(health_router, tags=["Health"])
app.include_router(detect_router, tags=["Detection"])
