"""Telecom customer support scenario generator inspired by agentic benchmarks."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import ScenarioGenerator, ScenarioSample, ValidationResult


CASES = [
    {
        "issue": "Customer reports intermittent packet loss during video calls.",
        "customer_tier": "enterprise",
        "recent_changes": ["Core router firmware upgrade", "Traffic shaping policy tweak"],
        "tools": ["look_up_account", "query_noc_dashboard", "generate_remedy_ticket"],
    },
    {
        "issue": "Billing dispute for roaming charges exceeding plan limits.",
        "customer_tier": "postpaid",
        "recent_changes": ["International roaming plan activation"],
        "tools": ["look_up_account", "adjust_invoice", "summarize_policy"],
    },
    {
        "issue": "5G small-cell outage affecting downtown sector.",
        "customer_tier": "municipal",
        "recent_changes": ["Power maintenance", "Backhaul reroute"],
        "tools": ["query_noc_dashboard", "dispatch_field_team", "notify_stakeholders"],
    },
    {
        "issue": "IoT fleet experiencing SIM authentication failures after a core network upgrade.",
        "customer_tier": "industrial",
        "recent_changes": ["Core HSS patch deployment", "New APN provisioning"],
        "tools": ["look_up_account", "query_noc_dashboard", "simulate_attach"],
    },
    {
        "issue": "VIP enterprise is seeing degraded SLA on MPLS circuits during peak hours.",
        "customer_tier": "strategic-enterprise",
        "recent_changes": ["Capacity rebalancing", "QoS policy adjustments"],
        "tools": ["query_capacity", "generate_exec_update", "dispatch_field_team"],
    },
    {
        "issue": "Fiber break suspected on a metro ring following severe weather.",
        "customer_tier": "broadband-residential",
        "recent_changes": ["Maintenance window notifications"],
        "tools": ["dispatch_field_team", "notify_stakeholders", "query_noc_dashboard"],
    },
    {
        "issue": "Contact center escalations about failed eSIM activations on premium devices.",
        "customer_tier": "premium-postpaid",
        "recent_changes": ["Digital onboarding workflow update", "New handset launch"],
        "tools": ["look_up_account", "simulate_activation", "summarize_policy"],
    },
]


class TelecomScenarioGenerator(ScenarioGenerator):
    """Produces multi-step customer support flows with tool interactions."""

    def sample(self) -> ScenarioSample:
        case = self.random.choice(CASES)
        scenario_id = f"telecom/{self.random.randrange(1_000_000)}"

        system_prompt = (
            "You are a senior telecom support agent coordinating across billing, NOC, and field teams. "
            "Gather facts methodically, call tools to inspect systems, and output a clear resolution plan. "
            "Deliver most of your reasoning in English, adding concise Chinese bullet recaps for stakeholder communication."
        )
        user_prompt = (
            f"Issue: {case['issue']}\n"
            f"Customer tier: {case['customer_tier']}\n"
            f"Recent changes: {', '.join(case['recent_changes'])}\n"
            "Deliverables: 1) diagnostic summary, 2) immediate remediation steps, 3) communication plan.\n"
            "Language expectations: English primary narrative, optional supporting Chinese bullet points."
        )
        tools = [
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

        metadata = {
            "benchmark": "telecom-agent",
            "recommended_tools": case["tools"],
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
        score = 0.5 * coverage + 0.5 * (1.0 if used_recommended else 0.0)

        feedback = "Strong telecom troubleshooting narrative" if score >= 0.7 else "Expand tool usage or structure."
        return ValidationResult(score=score, feedback=feedback, require_retry=score < 0.4)
