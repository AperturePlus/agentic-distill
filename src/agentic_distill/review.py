"""Helpers for parsing reviewer model outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ReviewFeedback:
    """Structured reviewer feedback."""

    score: float
    needs_revision: bool
    feedback: str
    chinese_summary: Optional[str] = None


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
    needs_revision = _parse_bool(data.get("needs_revision", True))
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
    """Return the first balanced JSON object embedded in ``text``.

    The reviewer model may prepend/append commentary or emit multiple JSON objects.
    Instead of relying on a greedy regular expression, we scan the string while
    tracking nesting depth and quote state to locate the first complete JSON
    object. This keeps nested braces inside strings from prematurely terminating
    the match and avoids spanning across multiple objects.
    """

    start: Optional[int] = None
    depth = 0
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if start is None:
            if char == "{":
                start = idx
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : idx + 1]

    return None


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _parse_bool(value: Any) -> bool:
    """Return a robust boolean interpretation of ``value``.

    Strings such as ``"false"`` or ``"0"`` should evaluate to ``False`` while
    maintaining backwards compatibility with truthy/falsy Python objects. When
    the value cannot be interpreted, fall back to Python's ``bool`` semantics.
    """

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True

    return bool(value)

