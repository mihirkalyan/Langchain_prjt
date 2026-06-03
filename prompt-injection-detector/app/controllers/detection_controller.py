"""
Detection controller — orchestrates the three-layer pipeline.

Layer 1  rule_guard.check()         sync   fast regex/keyword scan
Layer 2  rag_retriever.retrieve()   async  ChromaDB semantic retrieval
Layer 3  llm_judge.judge()          async  concurrent LLM ensemble

Layer 1 short-circuits on a match; Layers 2 and 3 run sequentially
(Layer 3 needs Layer 2's RetrievalContext as input).

Designed for easy RBAC wrapping: inject auth context into __init__ or
decorate detect() — detection logic is untouched either way.
"""

from __future__ import annotations

import logging
import re

from app.models.schemas import DetectionRequest, DetectionResponse
from app.services import rule_guard
from app.services.llm_judges.aggregator import FinalVerdict
from app.services.llm_judges.llm_judge import LLMJudge, get_llm_judge
from app.services.rag_retriever import RagRetriever, get_rag_retriever

log = logging.getLogger(__name__)

# ── Log sanitization ───────────────────────────────────────────────────────────
# Duplicated intentionally from rule_guard to avoid a cross-layer import.
# Both copies are the canonical 3-line implementation and evolve together.

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _sanitize_for_log(text: str, max_len: int = 200) -> str:
    """
    Strip control characters, replace newlines, and truncate for safe logging.
    Prevents log-injection attacks where adversarial input forges log entries.
    """
    sanitized = (
        text.replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
    )
    return _CONTROL_CHAR_RE.sub("", sanitized)[:max_len]


# ── Controller ─────────────────────────────────────────────────────────────────


class DetectionController:
    """
    Single entry point for the prompt injection detection pipeline.

    Instantiate once per request (it holds no mutable state — the services
    are process-wide singletons). Pass mock services in tests.
    """

    def __init__(
        self,
        retriever: RagRetriever | None = None,
        judge: LLMJudge | None = None,
    ) -> None:
        # Fall back to process-wide singletons when not injected.
        self._retriever: RagRetriever = retriever or get_rag_retriever()
        self._judge: LLMJudge = judge or get_llm_judge()

    async def detect(self, request: DetectionRequest) -> DetectionResponse:
        """
        Run the three-layer detection pipeline and return a DetectionResponse.

        Short-circuit path (Layer 1 hit):
            rule_guard fires → return immediately, confidence=1.0, ensemble=[]

        Full path (Layer 1 clear):
            Layer 2 retrieval → Layer 3 ensemble → aggregate → map to response
        """
        text: str = request.text
        safe_text: str = _sanitize_for_log(text)

        # ── Layer 1: rule guard (sync, ~0 ms) ─────────────────────────────────
        rule_result = rule_guard.check(text)

        if rule_result.is_injection:
            log.warning(
                "AUDIT | layer=1 | label=injection | confidence=1.00 "
                "| rule=%s | technique=%s | input=%s",
                rule_result.triggered_rule,
                rule_result.technique,
                safe_text,
            )
            return DetectionResponse(
                label="injection",
                confidence=1.0,
                technique=rule_result.technique,
                triggered_rule=rule_result.triggered_rule,
                reasoning=f"Layer 1 rule match: {rule_result.triggered_rule}.",
                ensemble=[],
            )

        # ── Layer 2: RAG retrieval (async, ~50–200 ms) ─────────────────────────
        context = await self._retriever.retrieve(text)

        # ── Layer 3: LLM ensemble (async, ~500–2000 ms) ────────────────────────
        final: FinalVerdict = await self._judge.judge(text, context)

        log.info(
            "AUDIT | layer=3 | label=%s | confidence=%.2f "
            "| technique=%s | triggered_rule=None | input=%s",
            final.label,
            final.confidence,
            final.technique,
            safe_text,
        )

        return DetectionResponse(
            label=final.label,
            confidence=final.confidence,
            technique=final.technique,
            triggered_rule=None,
            reasoning=final.reasoning,
            ensemble=final.individual_verdicts,
        )
