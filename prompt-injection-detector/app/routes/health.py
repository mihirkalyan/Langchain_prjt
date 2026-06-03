"""
GET /health — service liveness and readiness check.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.chroma_client import get_chroma_client

log = logging.getLogger(__name__)

router = APIRouter()

_REQUIRED_COLLECTIONS = ("injection_examples", "owasp_reference")


class HealthResponse(BaseModel):
    status: str
    collections: dict[str, int]
    detail: str | None = None


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns service status and ChromaDB collection sizes. "
        "Status is 'ok' when both collections are populated, "
        "'degraded' when they exist but are empty (run ingest scripts), "
        "or 'unhealthy' when ChromaDB is unreachable."
    ),
)
async def health() -> HealthResponse:
    """
    Liveness + readiness probe.

    Checks ChromaDB directly without loading the embedding model —
    health checks must be cheap enough to call from a load balancer.
    """
    client = get_chroma_client()
    collections: dict[str, int] = {}
    status = "ok"
    detail: str | None = None

    try:
        for name in _REQUIRED_COLLECTIONS:
            try:
                col = client.get_collection(name)
                count = col.count()
                collections[name] = count
                if count == 0:
                    status = "degraded"
                    detail = (
                        f"Collection '{name}' is empty — "
                        f"run data/ingest_{'datasets' if 'examples' in name else 'owasp'}.py."
                    )
            except Exception:
                collections[name] = 0
                status = "degraded"
                detail = (
                    f"Collection '{name}' not found — "
                    f"run data/ingest_{'datasets' if 'examples' in name else 'owasp'}.py."
                )

    except Exception as exc:
        log.error("ChromaDB unreachable during health check: %s", exc)
        return HealthResponse(
            status="unhealthy",
            collections={name: 0 for name in _REQUIRED_COLLECTIONS},
            detail=f"ChromaDB unreachable: {type(exc).__name__}",
        )

    return HealthResponse(status=status, collections=collections, detail=detail)
