"""
Layer 3 orchestrator — runs all LLM judges concurrently and aggregates verdicts.

Public API:
    get_llm_judge() -> LLMJudge   (cached singleton)
    await llm_judge.judge(payload, context) -> FinalVerdict

Adding a new judge model:
    1. Create the file (e.g. mistral_judge.py), subclass BaseLLMJudge.
    2. Import it here and add an instance to self._judges in __init__.
    No other files change.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from app.models.schemas import JudgeVerdict, RetrievalContext
from app.services.llm_judges.aggregator import FinalVerdict, aggregate
from app.services.llm_judges.base import BaseLLMJudge
from app.services.llm_judges.gemini_judge import GeminiJudge
from app.services.llm_judges.groq_judge import GroqJudge

log = logging.getLogger(__name__)

# Fallback verdict inserted when a judge raises despite its "never raises" contract.
_FALLBACK_MODEL = "unknown-judge"


def _exception_to_verdict(exc: BaseException, model: str = _FALLBACK_MODEL) -> JudgeVerdict:
    """Convert an unexpected judge exception into a conservative injection verdict."""
    log.error("Judge '%s' raised unexpectedly: %s: %s", model, type(exc).__name__, exc)
    return JudgeVerdict(
        model=model,
        label="injection",
        confidence=0.5,
        technique=None,
        reasoning=(
            f"Judge raised an unexpected exception ({type(exc).__name__}). "
            "Defaulting to conservative injection verdict."
        ),
    )


class LLMJudge:
    """
    Ensemble orchestrator for Layer 3.

    Runs every judge in self._judges concurrently and passes all verdicts to
    the aggregator. Judge exceptions are caught here and converted to fallback
    verdicts — the aggregator always receives a complete list.

    Lifetime: one instance per process (via get_llm_judge()).
    """

    def __init__(self) -> None:
        # ── Judge roster ───────────────────────────────────────────────────────
        # To add a model: import its class above and append an instance here.
        self._judges: list[BaseLLMJudge] = [
            GroqJudge(),
            GeminiJudge(),
        ]
        log.info(
            "LLMJudge initialised with %d judge(s): %s",
            len(self._judges),
            [j.model for j in self._judges],
        )

    async def judge(self, payload: str, context: RetrievalContext) -> FinalVerdict:
        """
        Run all judges concurrently against `payload` + `context`, then aggregate.

        Steps:
          1. Fire every judge coroutine simultaneously via asyncio.gather.
             return_exceptions=True ensures one judge failure never cancels
             its siblings mid-flight.
          2. Convert any unexpected exceptions to conservative fallback verdicts.
          3. Pass the complete verdict list to the aggregator.
          4. Return the FinalVerdict.
        """
        log.debug("LLMJudge: starting ensemble with %d judge(s).", len(self._judges))

        raw_results: list[JudgeVerdict | BaseException] = list(
            await asyncio.gather(
                *[j.judge(payload, context) for j in self._judges],
                return_exceptions=True,
            )
        )

        verdicts: list[JudgeVerdict] = []
        for judge, result in zip(self._judges, raw_results):
            if isinstance(result, BaseException):
                verdicts.append(_exception_to_verdict(result, model=judge.model))
            else:
                verdicts.append(result)

        final: FinalVerdict = aggregate(verdicts)

        log.info(
            "LLMJudge: ensemble complete | label=%s | confidence=%.2f | technique=%s",
            final.label,
            final.confidence,
            final.technique,
        )

        return final


@lru_cache(maxsize=1)
def get_llm_judge() -> LLMJudge:
    """
    Return the process-wide singleton LLMJudge.

    Both judge clients (Groq + Google) are initialised on the first call.
    Subsequent calls return the cached instance instantly.
    """
    return LLMJudge()
