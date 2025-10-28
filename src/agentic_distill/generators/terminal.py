"""Terminal troubleshooting scenario generator backed by a curated question bank."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ScenarioGenerator, ScenarioSample, ValidationResult
from ..question_bank import QuestionBank


class TerminalScenarioGenerator(ScenarioGenerator):
    """Produces SRE/terminal workflows from a reviewed question bank."""

    def __init__(
        self,
        *,
        question_bank_path: Optional[str] = None,
        seed: Optional[int] = None,
        **params: Any,
    ):
        super().__init__(seed=seed, **params)
        default_path = Path(question_bank_path) if question_bank_path else Path("data/question_banks/terminal.jsonl")
        try:
            self.question_bank = QuestionBank(default_path, seed=seed)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Question bank not found at {default_path}. "
                "Run `python scripts/generate_cases.py --config configs/casegen.terminal.yaml` first."
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"Question bank at {default_path} is empty. Generate fresh cases before distillation."
            ) from exc

        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a shell command and return its stdout/stderr.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Bash command to run.",
                            }
                        },
                        "required": ["command"],
                    },
                },
            }
        ]

    def sample(self) -> ScenarioSample:
        case = self.question_bank.sample()

        task = case.get("task", "Investigate a latent production incident.")
        environment = case.get("environment", "Linux host (details unspecified)")
        systems = case.get("systems", [])
        symptoms = case.get("symptoms", [])
        telemetry_clues = case.get("telemetry_clues", [])
        recent_changes = case.get("recent_changes", [])
        tools = case.get("tools", [])
        constraints = case.get("constraints", [])
        objectives = case.get("objectives") or case.get("resolution_objectives") or [
            "Determine root cause.",
            "Define remediation plan.",
        ]
        risk_level = case.get("risk_level", "medium")
        language_policy = case.get("language_policy", "en-primary zh-secondary")
        metadata_block = case.get("metadata", {})

        scenario_id = f"terminal/{case.get('id') or case.get('uid') or self.random.randrange(1_000_000)}"

        system_prompt = (
            "You are an SRE operating within a production shell (read-only unless otherwise stated). "
            "Think aloud, detail the commands you intend to run, and never fabricate output—explain what each command would reveal. "
            "Keep the main reasoning in English and finish key sections with concise Chinese bullet recaps."
        )

        sections: List[str] = [
            f"Task: {task}",
            f"Environment: {environment}",
        ]
        if systems:
            sections.append(f"Systems involved: {', '.join(systems)}")
        if symptoms:
            sections.append(f"Primary symptoms: {', '.join(symptoms)}")
        if telemetry_clues:
            sections.append(f"Telemetry clues: {', '.join(telemetry_clues)}")
        if recent_changes:
            sections.append(f"Recent changes: {', '.join(recent_changes)}")
        if tools:
            sections.append(f"Candidate commands/tools: {', '.join(tools)}")
        if constraints:
            sections.append(f"Constraints: {', '.join(constraints)}")
        sections.append(f"Risk level: {risk_level}")
        sections.append("Objectives:")
        for obj in objectives:
            sections.append(f"  - {obj}")

        deliverables = (
            "Deliverables:\n"
            "1. Investigation blueprint – chronological plan referencing exact commands and expected observations.\n"
            "2. Command log with anticipated outputs/errors and how each narrows the hypothesis space.\n"
            "3. Findings & mitigations – root cause narrative, mitigations, and postmortem actions.\n"
            "4. 中文要点 – concise Chinese bullet recap emphasising actions and safeguards.\n"
            "5. Metadata JSON containing keys `scenario_type`, `risk_level`, `recommended_tools`, `constraints`, `telemetry_clues`."
        )

        user_prompt = "\n".join(sections + ["", deliverables])

        metadata = {
            "benchmark": "terminal-bench",
            "recommended_tools": tools,
            "language_policy": language_policy,
            "source_case": case,
            "risk_level": risk_level,
            "constraints": constraints,
            "telemetry_clues": telemetry_clues,
            "metadata_overrides": metadata_block,
            "tool_definitions": [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "parameters": tool["function"].get("parameters", {}),
                }
                for tool in self._tools
                if tool.get("type") == "function" and tool.get("function")
            ],
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

        final_answer = assistant_messages[-1].get("content", "")
        lowered = final_answer.lower()

        includes_sections = all(keyword in lowered for keyword in ["investigation", "command", "findings"])
        tools = metadata.get("recommended_tools", [])
        mentions_tools = any(tool.lower() in lowered for tool in tools)
        includes_metadata_json = all(key in lowered for key in ['"scenario_type"', '"recommended_tools"'])
        includes_chinese = any("\u4e00" <= char <= "\u9fff" for char in final_answer)

        score_components = [
            1.0 if includes_sections else 0.0,
            1.0 if mentions_tools else 0.0,
            1.0 if includes_metadata_json else 0.0,
            1.0 if includes_chinese else 0.0,
        ]
        score = sum(score_components) / len(score_components)

        feedback = "Robust shell troubleshooting playbook."
        if score < 0.75:
            feedback = "Ensure investigation/command/findings sections, reference the provided tools, include metadata JSON and Chinese recap."

        return ValidationResult(score=score, feedback=feedback, require_retry=score < 0.5)

