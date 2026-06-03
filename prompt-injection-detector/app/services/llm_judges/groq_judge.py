"""
Layer 3 judge — llama-3.1-8b-instant via the Groq API.

Uses langchain-groq (ChatGroq) for async inference against the shared
canary-token prompt. Temperature is fixed at 0.0 for deterministic output.
"""

from __future__ import annotations

import logging

from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.models.schemas import JudgeVerdict, RetrievalContext
from app.services.llm_judges.base import SHARED_PROMPT, BaseLLMJudge

log = logging.getLogger(__name__)


class GroqJudge(BaseLLMJudge):
    """
    LLM judge backed by llama-3.1-8b-instant via Groq.

    Lifetime: one instance per process — the orchestrator (llm_judge.py)
    creates it once and reuses it across requests.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name: str = settings.groq_model

        self._chat = ChatGroq(
            model=self._model_name,
            api_key=settings.groq_api_key,
            temperature=0.0,
            max_tokens=150,
        )

        # Build the runnable chain once — SHARED_PROMPT formats the messages,
        # then passes them to ChatGroq for inference.
        self._chain = SHARED_PROMPT | self._chat

    # ── BaseLLMJudge interface ────────────────────────────────────────────────

    @property
    def model(self) -> str:
        return self._model_name

    async def judge(self, payload: str, context: RetrievalContext) -> JudgeVerdict:
        """
        Classify `payload` using Groq and return a canary-token-parsed verdict.

        Steps:
          1. Build prompt variables (payload + retrieval context).
          2. Invoke the chain asynchronously.
          3. Parse the canary token from the raw model output.
          4. Derive the OWASP technique from retrieval context (not model output).

        Never raises — any exception produces a conservative injection verdict
        at confidence 0.5 so the aggregator can weigh the other judge's vote.
        """
        try:
            prompt_vars = self._build_prompt_vars(payload, context)
            response = await self._chain.ainvoke(prompt_vars)
            raw_output: str = response.content

            label, confidence, reasoning = self._parse_canary_token(raw_output)

        except Exception as exc:
            log.error(
                "GroqJudge.judge raised %s: %s",
                type(exc).__name__,
                exc,
            )
            return JudgeVerdict(
                model=self.model,
                label="injection",
                confidence=0.5,
                technique=None,
                reasoning=(
                    f"GroqJudge encountered an internal error ({type(exc).__name__}). "
                    "Defaulting to conservative injection verdict."
                ),
            )

        # OWASP technique is sourced from the retriever, not the model,
        # to keep the canary token format clean and reduce hallucination risk.
        technique: str | None = (
            context.owasp_techniques[0]["technique_name"]
            if context.owasp_techniques and label == "injection"
            else None
        )

        log.info(
            "GroqJudge verdict | model=%s | label=%s | confidence=%.2f | technique=%s",
            self.model,
            label,
            confidence,
            technique,
        )

        return JudgeVerdict(
            model=self.model,
            label=label,
            confidence=confidence,
            technique=technique,
            reasoning=reasoning,
        )
