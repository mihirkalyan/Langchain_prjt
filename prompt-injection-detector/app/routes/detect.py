"""
POST /detect — classify a text payload for prompt injection.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.controllers.detection_controller import DetectionController
from app.models.schemas import DetectionRequest, DetectionResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Classify text for prompt injection",
    description=(
        "Runs the three-layer detection pipeline (rule-based → RAG retrieval → LLM ensemble) "
        "and returns a classification verdict with confidence score and per-judge reasoning."
    ),
)
async def detect(request: DetectionRequest) -> DetectionResponse:
    """
    Classify `request.text` for prompt injection attacks.

    - **200** — classification complete (label is 'injection' or 'benign').
    - **422** — request validation failed (e.g. text too long or empty).
    - **500** — unexpected internal error; details are in server logs only.

    RBAC hook: replace `DetectionController()` with `Depends(get_controller)`
    once an auth layer is in place — no logic changes required.
    """
    try:
        controller = DetectionController()
        return await controller.detect(request)
    except Exception:
        # Log the full traceback server-side; never expose it to the caller.
        log.exception("Unhandled exception in POST /detect")
        raise HTTPException(status_code=500, detail="Internal detection error.")
