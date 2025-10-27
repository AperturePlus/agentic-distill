"""Core data types used across the distillation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """Chat-style message exchanged with the teacher model."""

    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ToolInvocation:
    """Represents a tool call within an agentic trace."""

    name: str
    arguments: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    success: Optional[bool] = None


@dataclass
class Episode:
    """Structured agentic trajectory."""

    scenario_id: str
    created_at: datetime
    system_prompt: str
    user_prompt: str
    messages: List[Message]
    tool_invocations: List[ToolInvocation] = field(default_factory=list)
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_serializable(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""

        return {
            "scenario_id": self.scenario_id,
            "created_at": self.created_at.isoformat(),
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {}),
                    **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
                }
                for msg in self.messages
            ],
            "tool_invocations": [
                {
                    "name": tool.name,
                    "arguments": tool.arguments,
                    **({"output": tool.output} if tool.output is not None else {}),
                    **({"success": tool.success} if tool.success is not None else {}),
                }
                for tool in self.tool_invocations
            ],
            "score": self.score,
            "metadata": self.metadata,
        }
