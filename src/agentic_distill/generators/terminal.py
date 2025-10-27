"""TerminalBench-inspired scenario generator."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import ScenarioGenerator, ScenarioSample, ValidationResult


TASKS = [
    (
        "Inspect disk usage on a Linux server and summarise which directories consume the most space.",
        ["du", "sort", "head"],
    ),
    (
        "Investigate a failing systemd service and propose a remediation plan.",
        ["systemctl", "journalctl"],
    ),
    (
        "Compare configuration differences between two git branches to detect breaking changes.",
        ["git", "diff"],
    ),
    (
        "Parse nginx access logs to extract the top offending IP addresses causing 5xx errors.",
        ["awk", "grep", "sort", "uniq"],
    ),
    (
        "Diagnose why a Kubernetes pod keeps crashing due to OOMKilled events and propose mitigations.",
        ["kubectl", "jq"],
    ),
    (
        "Triages a failed CI pipeline container build that errors out during dependency installation.",
        ["docker", "jq", "grep"],
    ),
    (
        "Investigate TLS handshake failures reported by clients across multiple regions.",
        ["openssl", "curl", "grep"],
    ),
    (
        "Audit SSH access logs to detect suspicious lateral movement attempts.",
        ["journalctl", "grep", "uniq"],
    ),
    (
        "Construct a plan to restore a replicated PostgreSQL cluster after a replica fell behind by hours.",
        ["psql", "pg_isready"],
    ),
]


class TerminalScenarioGenerator(ScenarioGenerator):
    """Produces shell troubleshooting tasks with strong tool affordances."""

    def sample(self) -> ScenarioSample:
        task, recommended_tools = self.random.choice(TASKS)
        scenario_id = f"terminal/{self.random.randrange(1_000_000)}"

        system_prompt = (
            "You are an SRE operating within a read-only production shell. "
            "Think aloud, list the commands you intend to run, then summarise your findings. "
            "Never fabricate command outputs; instead, explain what each command would reveal. "
            "Deliver the reasoning primarily in English, but add concise Chinese bullet recaps at the end when they add clarity."
        )
        user_prompt = (
            f"Task: {task}\n"
            "Workspace: Linux server with standard GNU tools installed.\n"
            "Goal: produce a structured incident report including root cause hypotheses, mitigation steps, and operational safeguards.\n"
            "Language expectations: main narrative in English, optional short Chinese recap sections summarising actions."
        )
        tools = [
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
        metadata = {
            "benchmark": "terminal-bench",
            "recommended_tools": recommended_tools,
            "language_policy": "en-primary zh-secondary",
        }
        return ScenarioSample(
            scenario_id=scenario_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            metadata=metadata,
        )

    def validate(
        self, messages: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> ValidationResult:
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        if not assistant_messages:
            return ValidationResult(score=0.0, feedback="Missing assistant response.", require_retry=True)

        analysis = assistant_messages[-1]["content"].lower()
        heuristics = [
            any(tool in analysis for tool in metadata.get("recommended_tools", [])),
            "incident" in analysis,
            "mitigation" in analysis or "next steps" in analysis,
        ]
        score = sum(heuristics) / len(heuristics)
        feedback = "Includes root cause analysis and mitigation" if score > 0.66 else "Needs richer analysis."
        return ValidationResult(score=score, feedback=feedback, require_retry=score < 0.4)
