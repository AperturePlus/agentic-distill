"""Scenario generator for MCP server integration and prompt engineering."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ScenarioGenerator, ScenarioSample, ValidationResult


@dataclass(frozen=True)
class MCPDescriptor:
    """Lightweight representation of an MCP server entry."""

    server_id: int
    name: str
    analysis: str
    primary_label: str
    secondary_labels: List[str]
    custom_label: Optional[str]
    overview: str
    tool_summaries: List[Dict[str, str]]
    featured: bool
    usage_count: int
    tags: List[str]
    categories: List[str]
    source_file: str
    connection_url: Optional[str] = None
    python_sdk_snippet: Optional[str] = None
    python_sdk_url: Optional[str] = None
    author: Optional[str] = None
    homepage: Optional[str] = None
    repository_url: Optional[str] = None

    @property
    def slug(self) -> str:
        safe_name = re.sub(r"[^a-z0-9]+", "-", self.name.lower())
        return safe_name.strip("-") or f"server-{self.server_id}"

    @property
    def weight(self) -> float:
        """Sampling weight favouring highly used or featured servers."""

        base = max(self.usage_count, 1)
        multiplier = 1.5 if self.featured else 1.0
        return math.log(base + 1) * multiplier


class MCPScenarioGenerator(ScenarioGenerator):
    """Produces scenarios that evaluate and orchestrate MCP servers."""

    SMALL_MODEL_CANDIDATES = [
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Llama-3.1-8B-Instruct",
        "Mistral-Nemo-Instruct",
        "Gemma2-9B-it",
        "Yi-Lite-9B-Chat",
    ]

    BENCHMARK_TARGETS = [
        "AgentBoard",
        "AGI-Eval Tool Bench",
        "TerminalBench",
        "BBH Toolformer Split",
        "Telecom-Contact-Center Eval",
        "Navigator-Instructions v2",
    ]

    MISSION_FRAGMENTS = [
        "spin up an agent skill library for {domain}",
        "stress-test the server for {domain} workflows before distilling traces",
        "curate high-agency exemplars for downstream fine-tuning in {domain}",
        "orchestrate prompt chains that favour low-latency execution in {domain}",
        "design evaluator rubrics for {domain} data generation",
    ]

    _DESCRIPTOR_CACHE: Dict[Path, List[MCPDescriptor]] = {}

    def __init__(
        self,
        *,
        dataset_dir: Optional[str | Path] = None,
        tool_summary_limit: int = 4,
        seed: Optional[int] = None,
        **params: Any,
    ):
        super().__init__(seed=seed, **params)
        self.dataset_dir = Path(dataset_dir) if dataset_dir else Path(__file__).resolve().parent / "mcp_servers"
        self.tool_summary_limit = max(tool_summary_limit, 1)
        self.descriptors = self._load_descriptors(self.dataset_dir)
        if not self.descriptors:
            raise RuntimeError(f"No MCP descriptors found in {self.dataset_dir}")
        self.weights = [descriptor.weight for descriptor in self.descriptors]

    @classmethod
    def _load_descriptors(cls, dataset_dir: Path) -> List[MCPDescriptor]:
        if dataset_dir in cls._DESCRIPTOR_CACHE:
            return cls._DESCRIPTOR_CACHE[dataset_dir]

        descriptors: List[MCPDescriptor] = []
        for path in sorted(dataset_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            labels = data.get("labels", {})
            metadata = data.get("metadata", {})
            server_info = metadata.get("server_info_crawled", {})
            remote = metadata.get("remote_server_response", {})

            tool_candidates = remote.get("tools") or server_info.get("tools") or []
            tool_summaries: List[Dict[str, str]] = []
            for tool in tool_candidates:
                name = tool.get("name")
                description = tool.get("description") or ""
                if not name:
                    continue
                tool_summaries.append({"name": name, "description": description.strip()})

            if not tool_summaries:
                continue

            usage_raw = server_info.get("usage_count") or metadata.get("usage_count") or 0
            usage_count = _parse_int(usage_raw)

            descriptor = MCPDescriptor(
                server_id=int(server_info.get("id") or metadata.get("server_id") or 0),
                name=server_info.get("name") or metadata.get("server_name") or labels.get("custom_label") or path.stem,
                analysis=labels.get("analysis", ""),
                primary_label=labels.get("primary_label") or "General Tools",
                secondary_labels=labels.get("secondary_labels") or [],
                custom_label=labels.get("custom_label"),
                overview=server_info.get("overview") or labels.get("reasoning", ""),
                tool_summaries=tool_summaries,
                featured=bool(labels.get("featured_server")),
                usage_count=usage_count,
                tags=server_info.get("tags") or [],
                categories=server_info.get("categories") or [],
                source_file=str(path.name),
                connection_url=remote.get("url"),
                python_sdk_snippet=remote.get("python_sdk"),
                python_sdk_url=remote.get("python_sdk_url"),
                author=server_info.get("author"),
                homepage=server_info.get("homepage"),
                repository_url=server_info.get("repository_url"),
            )
            descriptors.append(descriptor)

        cls._DESCRIPTOR_CACHE[dataset_dir] = descriptors
        return descriptors

    def sample(self) -> ScenarioSample:
        descriptor = self.random.choices(self.descriptors, weights=self.weights, k=1)[0]

        selected_tools = self._select_tools(descriptor.tool_summaries)
        tool_summary_lines = "\n".join(
            f"- `{tool['name']}`: {tool['description']}"
            for tool in selected_tools
        )

        mission_fragment = self.random.choice(self.MISSION_FRAGMENTS).format(
            domain=descriptor.primary_label.lower()
        )
        benchmark_target = self.random.choice(self.BENCHMARK_TARGETS)
        small_models = self.random.sample(
            self.SMALL_MODEL_CANDIDATES,
            k=min(3, len(self.SMALL_MODEL_CANDIDATES)),
        )

        system_prompt = (
            "You are an autonomous solutions architect specialising in MCP server integrations. "
            "Produce decision-grade analyses that small instruction-tuned models can learn from. "
            "Keep the primary narrative in English and finish each major section with succinct Chinese bullet recaps."
        )

        connection_hint = descriptor.connection_url or descriptor.python_sdk_url or "N/A"
        connection_section = (
            f"Connection endpoint: {connection_hint}\n"
            "If referencing SDK usage, ensure snippets align with the official python client pattern."
        )

        user_prompt = (
            f"Server dossier: **{descriptor.name}** (primary label: {descriptor.primary_label}; "
            f"secondary labels: {', '.join(descriptor.secondary_labels) or 'none'}; "
            f"custom tag: {descriptor.custom_label or 'N/A'}).\n"
            f"{descriptor.analysis}\n\n"
            f"Tools (subset):\n{tool_summary_lines}\n\n"
            f"{connection_section}\n\n"
            f"Mission: {mission_fragment} while preparing data that will boost {benchmark_target} scores for"
            " compact agentic models.\n\n"
            "Deliverables (all in English except the dedicated Chinese section):\n"
            "1. **Capability Deep Dive** - map how each listed tool can be chained; include latency or risk considerations.\n"
            "2. **Workflow Library** - design at least three end-to-end workflows referencing the tool names verbatim and "
            "highlight which steps are good for distillation traces.\n"
            "3. **Evaluation & Guardrails** - propose automatic checks, feedback signals, and telemetry for data quality,\n"
            "   especially for supervising smaller models.\n"
            "4. **Chinese Recap (中文要点)** - 3-5 Chinese bullet points capturing essential actions.\n"
            "5. **Metadata JSON** - output a JSON object with keys: `scenario_type`, `source_server`, "
            "`recommended_small_model_targets`, `risk_level`, `benchmark_alignment`, `connection_url`. "
            "Use double quotes and ensure it is valid JSON.\n\n"
            "Contextual notes: prefer English reasoning, but adopt precise Chinese phrasing in the recap. "
            "Cite tool names, reference potential rate limits, and justify why the workflows create high-agency traces."
        )

        metadata = {
            "scenario_type": "mcp_integration",
            "language_policy": "en-primary zh-secondary",
            "source_server": {
                "id": descriptor.server_id,
                "name": descriptor.name,
                "primary_label": descriptor.primary_label,
                "secondary_labels": descriptor.secondary_labels,
                "custom_label": descriptor.custom_label,
                "tags": descriptor.tags,
                "categories": descriptor.categories,
                "featured": descriptor.featured,
                "usage_count": descriptor.usage_count,
                "author": descriptor.author,
                "homepage": descriptor.homepage,
                "repository_url": descriptor.repository_url,
                "source_file": descriptor.source_file,
                "connection_url": descriptor.connection_url,
                "python_sdk_url": descriptor.python_sdk_url,
            },
            "tool_names": [tool["name"] for tool in descriptor.tool_summaries],
            "tool_focus": [tool["name"] for tool in selected_tools],
            "target_benchmark": benchmark_target,
            "small_model_candidates": small_models,
            "mission": mission_fragment,
            "python_sdk_snippet": descriptor.python_sdk_snippet,
        }

        scenario_id = f"mcp/{descriptor.slug}-{self.random.randrange(1_000_000)}"

        return ScenarioSample(
            scenario_id=scenario_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=metadata,
        )

    def validate(self, messages: List[Dict[str, Any]], metadata: Dict[str, Any]) -> ValidationResult:
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        if not assistant_messages:
            return ValidationResult(
                score=0.0, feedback="Missing assistant response.", require_retry=True
            )

        final_answer = assistant_messages[-1].get("content", "")
        lowered = final_answer.lower()

        tool_names = metadata.get("tool_names", [])
        tool_mentioned = any(name.lower() in lowered for name in tool_names)

        has_metadata_block = '"scenario_type"' in lowered and '"source_server"' in lowered

        contains_chinese = any("\u4e00" <= char <= "\u9fff" for char in final_answer)

        score_components = [
            tool_mentioned,
            has_metadata_block,
            contains_chinese,
        ]
        score = sum(score_components) / len(score_components)

        feedback = "Balanced tool analysis with metadata block."
        if score < 0.66:
            feedback = "Ensure tool names, metadata JSON, and Chinese recap are present."

        return ValidationResult(
            score=score,
            feedback=feedback,
            require_retry=score < 0.4,
        )

    def _select_tools(self, tool_summaries: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(tool_summaries) <= self.tool_summary_limit:
            return tool_summaries
        indices = self.random.sample(range(len(tool_summaries)), k=self.tool_summary_limit)
        return [tool_summaries[i] for i in sorted(indices)]


def _parse_int(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        digits = value.replace(",", "").strip()
        if not digits:
            return 0
        try:
            return int(float(digits))
        except ValueError:
            return 0
    return 0
