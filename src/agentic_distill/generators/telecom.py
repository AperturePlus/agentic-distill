"""Telecom customer support scenario generator backed by a question bank."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ScenarioGenerator, ScenarioSample, ValidationResult
from ..question_bank import QuestionBank


class TelecomScenarioGenerator(ScenarioGenerator):
    """Produces multi-step customer support flows with tool interactions."""

    def __init__(
        self,
        *,
        question_bank_path: Optional[str] = None,
        seed: Optional[int] = None,
        **params: Any,
    ):
        super().__init__(seed=seed, **params)
        default_path = Path(question_bank_path) if question_bank_path else Path("data/question_banks/telecom.jsonl")
        try:
            self.question_bank = QuestionBank(default_path, seed=seed)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Question bank not found at {default_path}. "
                "Run `python scripts/generate_cases.py --config configs/casegen.telecom.yaml` first."
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"Question bank at {default_path} is empty. Generate fresh cases before distillation."
            ) from exc

        # Precompute static tool schema used for tool-call style plans
        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": "invoke_tool",
                    "description": "Interact with internal telecom tools.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string"},
                            "inputs": {"type": "object"},
                        },
                        "required": ["tool_name"],
                    },
                },
            }
        ]

    def sample(self) -> ScenarioSample:
        case = self.question_bank.sample()
        issue = case.get("issue", "Undiagnosed telecom escalation")
        customer_tier = case.get("customer_tier", "unspecified tier")
        region = case.get("region")
        symptoms = case.get("symptoms") or case.get("symptom_highlights") or []
        recent_changes = case.get("recent_changes") or []
        objectives = (
            case.get("resolution_objectives")
            or case.get("objectives")
            or ["Restore service stability", "Communicate status to stakeholders"]
        )
        risk_level = case.get("risk_level", "medium")
        evaluation_focus = case.get("evaluation_focus") or case.get("quality_gates") or []
        recommended_tools = case.get("tools") or case.get("recommended_tools") or []
        telemetry = case.get("telemetry_context")

        scenario_id = f"telecom/{case.get('id') or case.get('uid') or self.random.randrange(1_000_000)}"

        system_prompt = (
            "You are a senior telecom support agent coordinating across billing, NOC, and field teams. "
            "Gather facts methodically, call tools to inspect systems, and output a clear resolution plan. "
            "Deliver most of your reasoning in English, adding concise Chinese bullet recaps for stakeholder communication."
        )

        lines = [
            f"Issue: {issue}",
            f"Customer tier: {customer_tier}",
        ]
        if region:
            lines.append(f"Region: {region}")
        if symptoms:
            lines.append(f"Primary symptoms: {', '.join(symptoms)}")
        if recent_changes:
            lines.append(f"Recent changes: {', '.join(recent_changes)}")
        if telemetry:
            lines.append(f"Telemetry hints: {telemetry}")
        if recommended_tools:
            lines.append(f"Candidate tools: {', '.join(recommended_tools)}")
        lines.append(f"Risk level: {risk_level}")
        lines.append("Resolution objectives:")
        for objective in objectives:
            lines.append(f"  - {objective}")
        if evaluation_focus:
            lines.append("Evaluation focus areas:")
            for item in evaluation_focus:
                lines.append(f"  - {item}")

        deliverables = (
            "Deliverables:\n"
            "1. Diagnostic summary covering root-cause hypotheses and ruled-out factors.\n"
            "2. Immediate remediation steps including tool invocation rationale and fallback options.\n"
            "3. Communication plan tailored by stakeholder (Ops lead, account team, customer-facing comms).\n"
            "4. 中文要点: concise Chinese bullet recap of the recovery plan.\n"
            "5. Metadata JSON with keys `scenario_type`, `risk_level`, `telemetry_needed`, `recommended_tools`."
        )

        user_prompt = "\n".join(lines + ["", deliverables])

        metadata = {
            "benchmark": "telecom-agent",
            "recommended_tools": recommended_tools,
            "language_policy": case.get("language_policy", "en-primary zh-secondary"),
            "source_case": case,
        }

        return ScenarioSample(
            scenario_id=scenario_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=self._tools,
            metadata=metadata,
        )

    def validate(
        self,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> ValidationResult:
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        if not assistant_messages:
            return ValidationResult(score=0.0, feedback="Missing assistant response.", require_retry=True)

        final_answer = assistant_messages[-1].get("content", "").lower()
        required_sections = ["diagnostic", "remediation", "communication"]
        coverage = sum(section in final_answer for section in required_sections) / len(required_sections)

        tool_calls = [
            call
            for msg in assistant_messages
            for call in msg.get("tool_calls", [])
        ]
        used_recommended = any(
            call.get("function", {}).get("name") in metadata.get("recommended_tools", [])
            for call in tool_calls
        )
        includes_metadata_json = '"scenario_type"' in final_answer and '"recommended_tools"' in final_answer
        includes_chinese = any("\u4e00" <= char <= "\u9fff" for char in assistant_messages[-1].get("content", ""))

        score_components = [
            coverage,
            1.0 if used_recommended else 0.0,
            1.0 if includes_metadata_json else 0.0,
            1.0 if includes_chinese else 0.0,
        ]
        score = sum(score_components) / len(score_components)

        feedback = "Comprehensive telecom playbook with metadata."
        if score < 0.75:
            feedback = "Ensure diagnostic/remediation/communication sections, metadata JSON, and Chinese recap."

        return ValidationResult(score=score, feedback=feedback, require_retry=score < 0.5)
