"""Core data types used across the distillation framework."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4


@dataclass
class Message:
    """Chat-style message exchanged with the teacher model."""

    role: str
    content: Any
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    thinking: Optional[Any] = None


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
    uuid: str = field(default_factory=lambda: str(uuid4()))
    available_tools: List[Dict[str, Any]] = field(default_factory=list)
    target_tools: List[Dict[str, Any]] = field(default_factory=list)

    def to_serializable(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""

        metadata_copy = deepcopy(self.metadata)
        base_metadata = {
            "scenario_id": self.scenario_id,
            "created_at": self.created_at.isoformat(),
        }
        full_metadata = {**base_metadata, **metadata_copy}

        if "mcp" not in full_metadata:
            mcp_metadata = self._build_mcp_metadata(full_metadata)
            if mcp_metadata:
                full_metadata["mcp"] = mcp_metadata

        subsets = self._infer_subsets(full_metadata)
        if subsets:
            primary_subset = subsets[0]
            full_metadata.setdefault("subset_hint", primary_subset)

        conversation: List[Dict[str, Any]] = []
        for msg in self.messages:
            entry: Dict[str, Any] = {"role": msg.role}
            content = msg.content
            if isinstance(content, (dict, list)):
                entry["content"] = deepcopy(content)
            elif content is None:
                entry["content"] = ""
            else:
                entry["content"] = content
            if msg.name:
                entry["name"] = msg.name
            if msg.tool_calls:
                entry["tool_calls"] = deepcopy(msg.tool_calls)
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.thinking is not None:
                entry["thinking"] = deepcopy(msg.thinking)
            conversation.append(entry)

        available_tools = [deepcopy(tool) for tool in self.available_tools]
        available_names = {
            tool.get("name")
            for tool in available_tools
            if isinstance(tool, dict) and tool.get("name")
        }

        normalised_target_tools: List[Dict[str, Any]] = []
        for raw in self.target_tools:
            entry = deepcopy(raw)
            name = entry.get("name")
            if not name:
                continue
            entry.setdefault("reason", "Highlighted as a target tool by the scenario metadata.")
            entry.setdefault("source", "unspecified")
            entry.setdefault(
                "present_in_available_tools",
                bool(name in available_names),
            )
            normalised_target_tools.append(entry)

        final_answer = self._extract_final_answer()

        question_metadata = self._extract_question_metadata(full_metadata)
        response_metadata = self._extract_response_metadata(full_metadata)

        question_assessments = self._build_question_assessments(
            full_metadata,
            available_names,
            normalised_target_tools,
        )
        response_assessments = self._build_response_assessments(
            final_answer,
            full_metadata,
            normalised_target_tools,
        )

        thinking_traces = self._extract_thinking_traces()

        primary_subset = subsets[0] if subsets else None

        return {
            "uuid": self.uuid,
            "subset": primary_subset,
            "subsets": subsets,
            "metadata": full_metadata,
            "question": {
                "id": self.scenario_id,
                "system_prompt": self.system_prompt,
                "text": self.user_prompt,
                "language_policy": full_metadata.get("language_policy"),
                "metadata": question_metadata,
                "assessments": question_assessments,
            },
            "available_tools": available_tools,
            "target_tools": normalised_target_tools,
            "response": {
                "messages": conversation,
                "final_answer": final_answer,
                "tool_invocations": [
                    {
                        "name": tool.name,
                        "arguments": tool.arguments,
                        **({"output": tool.output} if tool.output is not None else {}),
                        **({"success": tool.success} if tool.success is not None else {}),
                    }
                    for tool in self.tool_invocations
                ],
                "assessments": response_assessments,
                "metadata": response_metadata,
                "thinking_traces": thinking_traces,
            },
        }

    def _extract_final_answer(self) -> str:
        for message in reversed(self.messages):
            if message.role != "assistant":
                continue
            text = self._normalise_content_to_text(message.content)
            if text:
                return text
        return ""

    def _extract_thinking_traces(self) -> List[Dict[str, Any]]:
        traces: List[Dict[str, Any]] = []
        for index, message in enumerate(self.messages):
            if message.role != "assistant":
                continue
            segments = self._normalise_thinking_segments(message)
            if segments:
                traces.append({"turn": index, "segments": segments})
        return traces

    def _normalise_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    candidate = item.strip()
                    if candidate:
                        parts.append(candidate)
                elif isinstance(item, dict):
                    content_type = str(item.get("type", "")).lower()
                    if content_type in {"thinking", "reasoning", "thought"}:
                        continue
                    text_value = item.get("text") or item.get("content") or item.get("value")
                    if isinstance(text_value, str):
                        candidate = text_value.strip()
                        if candidate:
                            parts.append(candidate)
            return "\n".join(parts).strip()
        if isinstance(content, dict):
            text_value = content.get("text") or content.get("content") or content.get("value")
            if isinstance(text_value, str):
                return text_value.strip()
        return ""

    def _normalise_thinking_segments(self, message: Message) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def _add(entry: Any) -> None:
            if isinstance(entry, dict):
                cleaned = {key: deepcopy(value) for key, value in entry.items()}
                serialised = repr(sorted(cleaned.items()))
                if cleaned and serialised not in seen:
                    seen.add(serialised)
                    segments.append(cleaned)
            elif isinstance(entry, str):
                cleaned = entry.strip()
                if cleaned:
                    serialised = f"text:{cleaned}"
                    if serialised not in seen:
                        seen.add(serialised)
                        segments.append({"type": "text", "text": cleaned})

        if message.thinking is not None:
            for entry in self._iter_normalised(message.thinking):
                _add(entry)

        if isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, dict) and item.get("type") in {"thinking", "reasoning", "thought"}:
                    _add(item)

        return segments

    def _extract_question_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        relevant_keys = [
            "scenario_type",
            "risk_level",
            "language_policy",
            "source_server",
            "mission",
            "target_benchmark",
            "small_model_candidates",
            "recommended_tools",
            "tool_focus",
            "tool_definitions",
            "tool_summaries",
            "selected_tool_details",
            "analysis",
            "overview",
        ]
        return {key: metadata[key] for key in relevant_keys if key in metadata}

    def _extract_response_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        relevant_keys = [
            "validation_feedback",
            "generation",
            "language_policy",
        ]
        return {key: metadata[key] for key in relevant_keys if key in metadata}

    def _build_mcp_metadata(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        source_server = metadata.get("source_server")
        tool_summaries = metadata.get("tool_summaries")
        focus_entries = metadata.get("tool_focus")
        featured = metadata.get("selected_tool_details")

        server_payload = self._normalise_mcp_server(source_server)
        summaries_payload = self._normalise_tool_entries(tool_summaries)
        featured_payload = self._normalise_tool_entries(featured)
        focus_list = self._normalise_string_list(focus_entries)
        tool_names = self._normalise_string_list(metadata.get("tool_names"))

        has_sdk = metadata.get("python_sdk_snippet")
        if isinstance(has_sdk, (list, tuple)):
            sdk_joined = "\n".join(str(part) for part in has_sdk if part)
            has_sdk = sdk_joined.strip() if sdk_joined.strip() else None
        elif isinstance(has_sdk, str):
            has_sdk = has_sdk.strip() or None
        else:
            has_sdk = None

        if not any(
            [
                server_payload,
                summaries_payload,
                featured_payload,
                focus_list,
                tool_names,
                has_sdk,
            ]
        ):
            return None

        mcp_metadata: Dict[str, Any] = {}
        if server_payload:
            mcp_metadata["server"] = server_payload
        if summaries_payload:
            mcp_metadata["tool_summaries"] = summaries_payload
        if featured_payload:
            mcp_metadata["featured_tools"] = featured_payload
        if focus_list:
            mcp_metadata["focus"] = focus_list
        if tool_names:
            mcp_metadata["tool_names"] = tool_names

        for key in ("mission", "analysis", "overview"):
            if metadata.get(key):
                mcp_metadata[key] = metadata[key]

        if has_sdk:
            mcp_metadata["python_sdk_snippet"] = has_sdk

        return mcp_metadata or None

    def _normalise_mcp_server(self, server: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(server, dict):
            return None

        keys = {
            "id",
            "name",
            "primary_label",
            "secondary_labels",
            "custom_label",
            "tags",
            "categories",
            "featured",
            "usage_count",
            "author",
            "homepage",
            "repository_url",
            "source_file",
            "connection_url",
            "python_sdk_url",
            "python_sdk_snippet",
        }

        payload: Dict[str, Any] = {}
        for key in keys:
            if key not in server:
                continue
            value = server[key]
            if key in {"secondary_labels", "tags", "categories"}:
                normalised = self._normalise_string_list(value)
                if normalised:
                    payload[key] = normalised
            elif value is not None:
                payload[key] = value

        return payload or None

    def _normalise_tool_entries(self, value: Any) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for item in self._iter_normalised(value):
            if isinstance(item, dict):
                name = item.get("name")
                if not name:
                    continue
                entry: Dict[str, Any] = {"name": name}
                if item.get("description"):
                    entry["description"] = item["description"]
                results.append(entry)
            elif isinstance(item, str):
                name = item.strip()
                if name:
                    results.append({"name": name})
        return results

    def _normalise_string_list(self, value: Any) -> List[str]:
        strings: List[str] = []
        for item in self._iter_normalised(value):
            if isinstance(item, str):
                candidate = item.strip()
                if candidate:
                    strings.append(candidate)
            elif isinstance(item, (int, float)):
                strings.append(str(item))
        return strings

    def _iter_normalised(self, value: Any) -> Iterable[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _infer_subsets(self, metadata: Dict[str, Any]) -> List[str]:
        subsets: List[str] = []
        assistant_turns = sum(1 for msg in self.messages if msg.role == "assistant")
        user_turns = sum(1 for msg in self.messages if msg.role == "user")
        tool_count = len(self.tool_invocations)
        reflection_passes = metadata.get("generation", {}).get("reflection_passes", 0)

        def _coerce_int(value: Any) -> int:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(float(value.strip()))
                except (ValueError, TypeError):
                    return 0
            return 0

        reflection_passes_int = _coerce_int(reflection_passes)

        def _add(tag: str) -> None:
            if tag and tag not in subsets:
                subsets.append(tag)

        if assistant_turns <= 1 and user_turns <= 1 and not tool_count and not reflection_passes_int:
            _add("single_turn")
        else:
            _add("multi_turn")

        if tool_count:
            _add("tool_use")

        if reflection_passes_int:
            _add("reflection")

        teacher_mode = (
            metadata.get("generation", {})
            .get("teacher", {})
            .get("mode")
        )
        if isinstance(teacher_mode, str) and teacher_mode.lower().startswith("thinking"):
            _add("thinking_model")

        scenario_type = metadata.get("scenario_type")
        if isinstance(scenario_type, str):
            normalised = scenario_type.strip()
            if normalised:
                if normalised.lower().startswith("mcp"):
                    _add("mcp")
                else:
                    _add(normalised)

        if metadata.get("mcp"):
            _add("mcp")

        for extra in self._iter_normalised(metadata.get("subsets")):
            if isinstance(extra, str):
                candidate = extra.strip()
                if candidate:
                    _add(candidate)

        return subsets

    def _build_question_assessments(
        self,
        metadata: Dict[str, Any],
        available_tool_names: set[str],
        target_tools: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        assessments: Dict[str, Dict[str, Any]] = {}

        risk_level = (
            metadata.get("risk_level")
            or metadata.get("metadata_overrides", {}).get("risk_level")
            or metadata.get("source_case", {}).get("risk_level")
        )
        if risk_level:
            assessments["difficulty"] = {
                "value": risk_level,
                "reason": (
                    "Scenario metadata specifies risk_level='" + str(risk_level) + "'."
                ),
            }
        else:
            tool_count = len([name for name in available_tool_names if name])
            inferred_level = "high" if tool_count >= 5 else "medium" if tool_count >= 2 else "low"
            assessments["difficulty"] = {
                "value": inferred_level,
                "reason": (
                    f"Inferred from {tool_count} available tool(s) when no explicit risk level was provided."
                ),
            }

        target_names = [tool.get("name") for tool in target_tools if tool.get("name")]
        if target_names:
            focus_value = "focused" if len(target_names) <= 4 else "broad"
            assessments["tooling_intent"] = {
                "value": focus_value,
                "reason": (
                    "Scenario highlights target tools: "
                    + ", ".join(sorted(target_names))
                ),
            }
        else:
            assessments["tooling_intent"] = {
                "value": "exploratory",
                "reason": "Scenario does not mandate specific tools, encouraging exploration.",
            }

        word_count = len(self.user_prompt.split())
        if word_count >= 600:
            density_value = "dense"
        elif word_count >= 250:
            density_value = "balanced"
        else:
            density_value = "concise"
        assessments["context_density"] = {
            "value": density_value,
            "reason": f"User prompt contains {word_count} words providing contextual guidance.",
        }

        source_server = metadata.get("source_server", {})
        if source_server:
            primary_label = source_server.get("primary_label")
            server_name = source_server.get("name")
            label_summary = ", ".join(source_server.get("secondary_labels", []) or [])
            reason_parts = []
            if server_name:
                reason_parts.append(f"Server '{server_name}' drives the scenario focus.")
            if primary_label:
                reason_parts.append(f"Primary label: {primary_label}.")
            if label_summary:
                reason_parts.append(f"Secondary labels: {label_summary}.")
            assessments["domain_relevance"] = {
                "value": primary_label or "unspecified",
                "reason": " ".join(reason_parts) if reason_parts else "Derived from MCP metadata.",
            }
        elif metadata.get("scenario_type"):
            scenario_type = metadata["scenario_type"]
            assessments["domain_relevance"] = {
                "value": scenario_type,
                "reason": f"Scenario declared type '{scenario_type}'.",
            }

        return assessments

    def _build_response_assessments(
        self,
        final_answer: str,
        metadata: Dict[str, Any],
        target_tools: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        assessments: Dict[str, Dict[str, Any]] = {}

        score = self.score if self.score is not None else 0.0
        feedback = metadata.get("validation_feedback") or "No validation feedback recorded."
        assessments["quality"] = {
            "value": round(score, 3),
            "reason": f"Validation scored {score:.2f}. {feedback}",
        }

        target_names = [tool.get("name") for tool in target_tools if tool.get("name")]
        if target_names:
            mentioned = {
                name
                for name in target_names
                if name and name.lower() in final_answer.lower()
            }
            invoked = {
                invocation.name
                for invocation in self.tool_invocations
                if invocation.success and invocation.name in target_names
            }
            covered = mentioned | invoked
            coverage_ratio = len(covered) / len(target_names)
            if coverage_ratio >= 0.9:
                coverage_value = "excellent"
            elif coverage_ratio >= 0.5:
                coverage_value = "adequate"
            else:
                coverage_value = "insufficient"
            assessments["tool_alignment"] = {
                "value": coverage_value,
                "reason": (
                    f"Covered {len(covered)} of {len(target_names)} highlighted tool(s)"
                    f" (mentioned: {sorted(mentioned)}, invoked: {sorted(invoked)})."
                ),
            }
        else:
            assessments["tool_alignment"] = {
                "value": "not_applicable",
                "reason": "No scenario-level target tools to assess alignment against.",
            }

        language_policy = metadata.get("language_policy", "")
        contains_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in final_answer)
        contains_ascii_letters = any("a" <= ch.lower() <= "z" for ch in final_answer)
        if "zh" in language_policy.lower():
            if contains_chinese and contains_ascii_letters:
                compliance_value = "pass"
                compliance_reason = (
                    "Answer includes both English reasoning and Chinese recap as required."
                )
            elif contains_chinese:
                compliance_value = "partial"
                compliance_reason = "Chinese recap present but English coverage appears limited."
            else:
                compliance_value = "partial"
                compliance_reason = "Chinese recap missing despite language policy expectations."
        else:
            compliance_value = "pass" if contains_ascii_letters else "partial"
            compliance_reason = (
                "English narrative detected." if contains_ascii_letters else "English narrative missing."
            )
        assessments["language_compliance"] = {
            "value": compliance_value,
            "reason": compliance_reason,
        }

        reviews = metadata.get("generation", {}).get("review") or []
        if reviews:
            final_review = reviews[-1]
            review_score = round(float(final_review.get("score", 0.0)), 3)
            reviewer = final_review.get("reviewer_endpoint") or final_review.get("reviewer_model")
            needs_revision = final_review.get("needs_revision")
            assessments["review_alignment"] = {
                "value": review_score,
                "reason": (
                    f"Final review by {reviewer or 'reviewer'} scored {review_score}"
                    f" and needs_revision={needs_revision}."
                ),
            }
        else:
            assessments["review_alignment"] = {
                "value": "not_requested",
                "reason": "No reviewer feedback captured for this episode.",
            }

        return assessments
