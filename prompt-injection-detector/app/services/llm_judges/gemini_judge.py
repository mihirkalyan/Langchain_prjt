"""
Layer 3 judge — Gemini 2.0 Flash via the Google Generative AI API.

Uses langchain-google-genai (ChatGoogleGenerativeAI) for async inference
against the shared canary-token prompt. Temperature is fixed at 0.0.

Free-tier note: gemini-2.0-flash is on Google's free tier as of 2025.
Swap GEMINI_MODEL in .env for gemini-2.5-flash-preview-05-20 to use
the 2.5 preview with no code changes.
"""

from __future__ import annotations

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.models.schemas import JudgeVerdict, RetrievalContext
from app.services.llm_judges.base import SHARED_PROMPT, BaseLLMJudge

log = logging.getLogger(__name__)


class GeminiJudge(BaseLLMJudge):
    """
    LLM judge backed by Gemini Flash via the Google Generative AI API.

    Lifetime: one instance per process — the orchestrator (llm_judge.py)
    creates it once and reuses it across requests.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name: str = settings.gemini_model

        self._chat = ChatGoogleGenerativeAI(
            model=self._model_name,
            google_api_key=settings.google_api_key,
            temperature=0.0,
            max_output_tokens=150,
        )

        # Build the runnable chain once — SHARED_PROMPT formats the messages,
        # then passes them to ChatGoogleGenerativeAI for inference.
        self._chain = SHARED_PROMPT | self._chat

    # ── BaseLLMJudge interface ────────────────────────────────────────────────

    @property
    def model(self) -> str:
        return self._model_name

    async def judge(self, payload: str, context: RetrievalContext) -> JudgeVerdict:
        """
        Classify `payload` using Gemini Flash and return a canary-token-parsed verdict.

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
                "GeminiJudge.judge raised %s: %s",
                type(exc).__name__,
                exc,
            )
            return JudgeVerdict(
                model=self.model,
                label="injection",
                confidence=0.5,
                technique=None,
                reasoning=(
                    f"GeminiJudge encountered an internal error ({type(exc).__name__}). "
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
            "GeminiJudge verdict | model=%s | label=%s | confidence=%.2f | technique=%s",
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
