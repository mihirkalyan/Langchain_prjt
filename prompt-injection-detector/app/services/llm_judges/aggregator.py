"""
Layer 3 — Ensemble verdict aggregation.

Takes the list of JudgeVerdicts from all active judges and combines them
into a single FinalVerdict using a three-tier decision hierarchy:

  1. Escalation   — any judge with confidence > ESCALATION_THRESHOLD whose
                    label is agreed upon by all other escalating judges wins
                    immediately, overriding the vote count.
  2. Majority     — the label with more votes wins.
  3. Tie-break    — equal vote counts → higher-confidence group wins.
                    On exact confidence tie, injection wins (conservative).

Public API: `aggregate(verdicts) -> FinalVerdict`
"""

from __future__ import annotations

import logging
from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field

from app.models.schemas import JudgeVerdict

log = logging.getLogger(__name__)

# A single judge whose confidence exceeds this threshold can override the vote.
ESCALATION_THRESHOLD: float = 0.95


# ── Output type ────────────────────────────────────────────────────────────────


class FinalVerdict(BaseModel):
    """
    Aggregated ensemble result returned to the detection controller.

    The controller maps this to DetectionResponse, appending `triggered_rule`
    from the Layer 1 result (which the aggregator does not know about).
    """

    label: Literal["injection", "benign"]
    confidence: float = Field(ge=0.0, le=1.0, description="Mean confidence across all judges.")
    technique: str | None = None
    reasoning: str = Field(description="One-sentence consensus narrative.")
    individual_verdicts: list[JudgeVerdict]


# ── Private helpers ────────────────────────────────────────────────────────────


def _winning_technique(verdicts: list[JudgeVerdict], label: str) -> str | None:
    """Return the OWASP technique from the first verdict matching `label`."""
    for v in verdicts:
        if v.label == label and v.technique:
            return v.technique
    return None


def _verdict_summary(verdicts: list[JudgeVerdict]) -> str:
    """Compact per-judge summary for embedding in the reasoning string."""
    return " | ".join(
        f"{v.model}: {v.label} ({v.confidence:.2f})" for v in verdicts
    )


def _escalation_reasoning(
    trigger: JudgeVerdict,
    all_verdicts: list[JudgeVerdict],
    avg_confidence: float,
) -> str:
    return (
        f"Escalated: {trigger.model} reported {trigger.label} "
        f"with confidence {trigger.confidence:.2f} (> {ESCALATION_THRESHOLD} threshold). "
        f"Avg confidence: {avg_confidence:.2f}. "
        f"Ensemble: {_verdict_summary(all_verdicts)}."
    )


def _majority_reasoning(
    label: str,
    verdicts: list[JudgeVerdict],
    injection_votes: list[JudgeVerdict],
    benign_votes: list[JudgeVerdict],
    avg_confidence: float,
    is_tie: bool,
) -> str:
    n_for = len([v for v in verdicts if v.label == label])
    total = len(verdicts)
    summary = _verdict_summary(verdicts)

    if is_tie:
        inj_avg = mean(v.confidence for v in injection_votes) if injection_votes else 0.0
        ben_avg = mean(v.confidence for v in benign_votes) if benign_votes else 0.0
        return (
            f"Tied ({total // 2}/{total // 2}), resolved by group confidence: "
            f"{label} wins "
            f"(injection avg {inj_avg:.2f}, benign avg {ben_avg:.2f}). "
            f"Avg confidence: {avg_confidence:.2f}. "
            f"Ensemble: {summary}."
        )

    return (
        f"Majority vote ({n_for}/{total}): {label}. "
        f"Avg confidence: {avg_confidence:.2f}. "
        f"Ensemble: {summary}."
    )


# ── Public API ─────────────────────────────────────────────────────────────────


def aggregate(verdicts: list[JudgeVerdict]) -> FinalVerdict:
    """
    Combine judge verdicts into a single FinalVerdict.

    Decision hierarchy (evaluated in order, first match wins):
      1. Escalation  — all high-confidence judges agree → escalate immediately.
      2. Majority    — more votes on one label → that label wins.
      3. Tie-break   — equal votes → higher group confidence wins;
                       exact confidence tie defaults to injection (conservative).

    `confidence` is always the mean across ALL verdicts regardless of label,
    reflecting true ensemble certainty rather than winner certainty.
    """
    if not verdicts:
        log.error("aggregate() received an empty verdicts list.")
        return FinalVerdict(
            label="injection",
            confidence=0.5,
            technique=None,
            reasoning=(
                "No judge verdicts received — defaulting to conservative injection verdict."
            ),
            individual_verdicts=[],
        )

    avg_confidence = round(mean(v.confidence for v in verdicts), 4)

    # ── 1. Escalation ──────────────────────────────────────────────────────────
    high_confidence = [v for v in verdicts if v.confidence > ESCALATION_THRESHOLD]
    if high_confidence:
        escalated_labels: set[str] = {v.label for v in high_confidence}
        if len(escalated_labels) == 1:
            # All high-confidence judges agree — escalate unconditionally.
            escalated_label: Literal["injection", "benign"] = next(iter(escalated_labels))  # type: ignore[assignment]
            trigger = max(high_confidence, key=lambda v: v.confidence)
            technique = _winning_technique(verdicts, escalated_label)

            log.info(
                "Aggregator: escalated | trigger=%s | label=%s | confidence=%.2f",
                trigger.model,
                escalated_label,
                trigger.confidence,
            )
            return FinalVerdict(
                label=escalated_label,
                confidence=avg_confidence,
                technique=technique,
                reasoning=_escalation_reasoning(trigger, verdicts, avg_confidence),
                individual_verdicts=verdicts,
            )

        # High-confidence judges disagree — suppress escalation, fall through.
        log.warning(
            "Aggregator: conflicting escalation signals %s — falling through to majority.",
            escalated_labels,
        )

    # ── 2. Majority vote ───────────────────────────────────────────────────────
    injection_votes = [v for v in verdicts if v.label == "injection"]
    benign_votes = [v for v in verdicts if v.label == "benign"]
    is_tie = len(injection_votes) == len(benign_votes)

    if len(injection_votes) > len(benign_votes):
        label: Literal["injection", "benign"] = "injection"
    elif len(benign_votes) > len(injection_votes):
        label = "benign"
    else:
        # ── 3. Tie-break by group confidence ───────────────────────────────────
        inj_avg = mean(v.confidence for v in injection_votes) if injection_votes else 0.0
        ben_avg = mean(v.confidence for v in benign_votes) if benign_votes else 0.0
        # Default to injection on exact tie (conservative).
        label = "injection" if inj_avg >= ben_avg else "benign"

    technique = _winning_technique(verdicts, label)

    log.info(
        "Aggregator: majority | label=%s | avg_confidence=%.2f | injection=%d | benign=%d",
        label,
        avg_confidence,
        len(injection_votes),
        len(benign_votes),
    )

    return FinalVerdict(
        label=label,
        confidence=avg_confidence,
        technique=technique,
        reasoning=_majority_reasoning(
            label, verdicts, injection_votes, benign_votes, avg_confidence, is_tie
        ),
        individual_verdicts=verdicts,
    )
