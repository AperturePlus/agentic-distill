"""Unit tests for reviewer feedback parsing."""

from __future__ import annotations

import json

import pytest

from agentic_distill.review import parse_review_feedback


def test_parse_review_feedback_handles_surrounding_text_and_strings() -> None:
    raw = """Reviewer notes before JSON.\n{"score": "0.75", "needs_revision": "false", "feedback": "Looks good {mostly}"}\nClosing thoughts."""

    feedback = parse_review_feedback(raw)

    assert feedback.score == pytest.approx(0.75)
    assert feedback.needs_revision is False
    assert feedback.feedback == "Looks good {mostly}"
    assert feedback.chinese_summary is None


@pytest.mark.parametrize(
    "needs_revision, expected",
    [
        ("false", False),
        ("False", False),
        ("no", False),
        ("0", False),
        (0, False),
        (1, True),
        ("yes", True),
    ],
)
def test_parse_review_feedback_coerces_needs_revision_values(
    needs_revision, expected
) -> None:
    payload = {
        "score": 0.1,
        "needs_revision": needs_revision,
        "feedback": "placeholder",
    }
    raw = json.dumps(payload)

    feedback = parse_review_feedback(raw)

    assert feedback.needs_revision is expected


def test_parse_review_feedback_returns_first_json_object() -> None:
    raw = (
        "Preface text. {\"score\": 0.2, \"needs_revision\": false, \"feedback\": \"first\"}"
        " {\"score\": 1.0, \"needs_revision\": false, \"feedback\": \"second\"}"
    )

    feedback = parse_review_feedback(raw)

    assert feedback.score == pytest.approx(0.2)
    assert feedback.feedback == "first"


def test_parse_review_feedback_falls_back_on_invalid_json() -> None:
    raw = "The reviewer responded with unstructured text."

    feedback = parse_review_feedback(raw)

    assert feedback.score == 0.0
    assert feedback.needs_revision is True
    assert feedback.feedback == raw
