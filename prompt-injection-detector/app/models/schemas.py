from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Hard cap on inbound text to prevent resource-exhaustion before any processing.
# Embedding + tokenization cost scales with input length; this is the cheapest guard.
MAX_TEXT_LENGTH: int = 10_000


# ── Layer 0: API boundary ──────────────────────────────────────────────────────


class DetectionRequest(BaseModel):
    """Inbound payload for POST /detect."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="User text to classify for prompt injection.",
    )


# ── Layer 2: RAG retriever output ──────────────────────────────────────────────


class RetrievalContext(BaseModel):
    """Structured output of the RAG retriever; passed as-is to every LLM judge.

    Typed explicitly so judges cannot silently receive a differently-shaped dict
    and misinterpret missing fields as 'no context found'.
    """

    similar_examples: list[dict] = Field(
        default_factory=list,
        description="Top-K injection examples retrieved from the injection_examples collection.",
    )
    owasp_techniques: list[dict] = Field(
        default_factory=list,
        description="Matching OWASP LLM Top-10 technique entries from the owasp_reference collection.",
    )


# ── Layer 3: LLM judge output ──────────────────────────────────────────────────


class JudgeVerdict(BaseModel):
    """Verdict produced by a single LLM judge.

    Using Literal for `label` means Pydantic will reject any value outside the
    two allowed strings — including hallucinated labels like "INJECTION" or
    "unsafe". An invalid label surfaces as a validation error, which the
    aggregator treats as an anomaly (canary token principle at schema level).
    """

    model: str = Field(
        ...,
        description="Identifier of the LLM that produced this verdict.",
    )
    label: Literal["injection", "benign"] = Field(
        ...,
        description="Binary classification result.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's self-reported confidence in its label.",
    )
    technique: str | None = Field(
        None,
        description="OWASP LLM Top-10 technique name if injection was detected.",
    )
    reasoning: str = Field(
        ...,
        description="Human-readable justification from the model.",
    )


# ── Final response ─────────────────────────────────────────────────────────────


class DetectionResponse(BaseModel):
    """Full response payload for POST /detect.

    Mirrors the documented JSON contract exactly. `ensemble` carries the
    individual JudgeVerdicts so callers can audit per-model reasoning.
    """

    label: Literal["injection", "benign"] = Field(
        ...,
        description="Aggregated final label after ensemble voting.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean confidence across all ensemble members.",
    )
    technique: str | None = Field(
        None,
        description="OWASP technique if injection was detected.",
    )
    triggered_rule: str | None = Field(
        None,
        description="Rule name if Layer 1 (rule_guard) fired and short-circuited.",
    )
    reasoning: str = Field(
        ...,
        description="Consensus summary synthesised from ensemble verdicts.",
    )
    ensemble: list[JudgeVerdict] = Field(
        default_factory=list,
        description="Individual verdicts from each LLM judge.",
    )
