"""Helpers for parsing reviewer model outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ReviewFeedback:
    """Structured reviewer feedback."""

    score: float
    needs_revision: bool
    feedback: str
    chinese_summary: Optional[str] = None


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def parse_review_feedback(raw_content: str) -> ReviewFeedback:
    """Parse JSON feedback emitted by the reviewer model.

    Falls back gracefully when the reviewer deviates from the expected schema.
    """

    content = raw_content.strip()
    candidate = _extract_json_block(content) or "{}"

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        data = {}

    score = _clamp_score(data.get("score", 0.0))
    needs_revision = bool(data.get("needs_revision", True))
    feedback = (
        data.get("feedback")
        or raw_content.strip()
        or "Reviewer response missing; treat as requiring revision."
    )
    chinese_summary = data.get("chinese_summary")

    return ReviewFeedback(
        score=score,
        needs_revision=needs_revision,
        feedback=feedback,
        chinese_summary=chinese_summary,
    )


def _extract_json_block(text: str) -> Optional[str]:
    match = JSON_BLOCK_PATTERN.search(text)
    if not match:
        return None
    return match.group(0)


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))

