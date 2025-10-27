"""Base classes and helper utilities for scenario generators."""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScenarioSample:
    """A concrete prompt package to feed into the teacher model."""

    scenario_id: str
    system_prompt: str
    user_prompt: str
    tools: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Validation output after analysing a teacher trajectory."""

    score: float
    feedback: str
    require_retry: bool = False


class ScenarioGenerator(abc.ABC):
    """Abstract base class for agentic scenario generators."""

    def __init__(self, *, seed: Optional[int] = None, **params: Any):
        self.random = random.Random(seed)
        self.params = params

    @abc.abstractmethod
    def sample(self) -> ScenarioSample:
        """Sample a single scenario instance."""

    @abc.abstractmethod
    def validate(self, messages: List[Dict[str, Any]], metadata: Dict[str, Any]) -> ValidationResult:
        """Validate a completed trajectory."""
