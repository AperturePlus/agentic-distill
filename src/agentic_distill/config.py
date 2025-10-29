"""Configuration models for the agentic distillation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .utils import deep_merge_dict


class ModelEndpointConfig(BaseModel):
    """Defines a single LLM endpoint option."""

    name: str = Field(..., description="Unique identifier for this endpoint.")
    provider: str = Field(
        ..., description="Identifier for the API provider (e.g. openai, anthropic, custom)."
    )
    model: str = Field(..., description="Model name or deployment identifier.")
    interaction_mode: Literal["instruct", "thinking", "auto"] = Field(
        "instruct",
        description=(
            "High-level behaviour descriptor for the model. 'thinking' models emit structured "
            "reasoning segments that should be captured in the dataset; 'auto' lets downstream "
            "logic infer behaviour dynamically."
        ),
    )
    api_key_env: str = Field(
        "TEACHER_API_KEY",
        description="Environment variable containing the API key.",
    )
    base_url: Optional[str] = Field(
        None, description="Optional override for API base URL (for custom gateways)."
    )
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_output_tokens: int = Field(2048, gt=0)
    request_timeout: float = Field(90.0, gt=0.0)
    retry_attempts: int = Field(6, ge=0)
    weight: float = Field(1.0, gt=0.0, description="Probability weight when sampling endpoints.")
    completion_path: str = Field(
        "/chat/completions",
        description="Relative request path for chat completions.",
    )
    request_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific payload overrides merged into every request.",
    )
    extra_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers sent with each request.",
    )


class EndpointPoolConfig(BaseModel):
    """Describes how to select between multiple endpoints."""

    selection_strategy: Literal["weighted_random", "round_robin"] = "weighted_random"
    preferred_order: Optional[list[str]] = Field(
        None,
        description="Optional ordered list of endpoint names to prioritise before sampling.",
    )
    endpoints: list[ModelEndpointConfig]

    @field_validator("endpoints")
    @classmethod
    def _validate_endpoints(cls, endpoints: list[ModelEndpointConfig]) -> list[ModelEndpointConfig]:
        if not endpoints:
            raise ValueError("Endpoint pool requires at least one endpoint.")
        names = [endpoint.name for endpoint in endpoints]
        if len(set(names)) != len(names):
            raise ValueError("Endpoint names must be unique within the pool.")
        return endpoints


class ReflectionConfig(BaseModel):
    """Controls optional self-reflection passes for quality assurance."""

    enabled: bool = True
    passes: int = Field(
        1, ge=0, description="Number of self-reflection passes after the initial answer."
    )
    critique_style: str = Field(
        "default",
        description="Reflection prompt style identifier (default, concise, exhaustive).",
    )


class ValidationConfig(BaseModel):
    """Validation thresholds applied to collected traces."""

    min_score: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum scenario-specific score to keep a trace.",
    )
    allow_partial_credit: bool = True
    require_tool_calls: bool = False


class OutputConfig(BaseModel):
    """Output dataset configuration."""

    base_dir: Path = Field(
        Path("data/exports"),
        description="Directory where dataset shards will be written.",
    )
    format: str = Field("jsonl", description="Output format: jsonl or parquet.")
    shard_size: int = Field(500, gt=0)
    target_shard_bytes: int = Field(
        150 * 1024 * 1024,
        gt=1024,
        description="Approximate maximum size (in bytes) for each exported shard.",
    )
    include_metadata: bool = True

    @field_validator("format")
    @classmethod
    def _validate_format(cls, value: str) -> str:
        allowed = {"jsonl", "parquet"}
        if value not in allowed:
            raise ValueError(f"Unsupported output format '{value}', choose {allowed}.")
        return value


class PromptTemplateConfig(BaseModel):
    """Global prompt fragments injected around scenario prompts."""

    global_system_prefix: str = Field(
        (
            "You are an elite agent tasked with producing high-quality, decision-rich traces that teach"
            " smaller models how to act autonomously. Respond primarily in English, adding concise Chinese"
            " summaries only when they materially clarify the solution. Avoid other languages."
        ),
        description="Prefix appended to every system prompt.",
    )
    user_guidelines: str = Field(
        (
            "Guidelines:\n"
            "- Think step-by-step and describe tool intent before executing.\n"
            "- Use clear headings and actionable checklists.\n"
            "- Provide the final answer mostly in English, with optional short Chinese recap sections."
        ),
        description="Prepended to every user prompt to enforce best practices.",
    )
    reviewer_template: str = Field(
        (
            "You are reviewing an agentic transcript. Score quality between 0 and 1.\n"
            "Respond strictly as JSON with keys: score (float), needs_revision (bool),"
            " feedback (string English), chinese_summary (string Chinese, optional)."
        ),
        description="Instructions sent to the reviewer model.",
    )
    revision_template: str = Field(
        (
            "The reviewer provided the following feedback (English with optional Chinese):\n"
            "{feedback}\n\n"
            "Revise your last answer. Incorporate all critical fixes while keeping explanations primarily"
            " in English and adding a brief Chinese recap if helpful."
        ),
        description="Template used when asking the teacher to revise after review.",
    )


class ReviewFlowConfig(BaseModel):
    """Review and refinement settings using an auxiliary model."""

    enabled: bool = False
    min_score: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Minimum reviewer score required to accept a trace.",
    )
    max_rounds: int = Field(
        1,
        ge=0,
        description="Maximum number of reviewer-driven revision cycles.",
    )
    auto_refine: bool = Field(
        True,
        description="If true, automatically feed reviewer feedback back to the teacher for refinement.",
    )


class ConcurrencyConfig(BaseModel):
    """Controls parallelism for the distillation loop."""

    max_workers: int = Field(4, ge=1, description="Maximum number of concurrent episodes being sampled.")


class ScenarioTemplate(BaseModel):
    """Descriptor for a single scenario family."""

    name: str
    generator: str = Field(
        ...,
        description="Python path to the scenario generator (e.g. agentic_distill.generators.terminal:TerminalScenarioGenerator).",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary parameters forwarded to the generator.",
    )
    target_episodes: int = Field(
        100,
        gt=0,
        description="Target number of episodes to sample for this scenario.",
    )
    weight: float = Field(
        1.0,
        gt=0.0,
        description="Relative sampling weight when mixing scenarios.",
    )


class DistillationConfig(BaseModel):
    """Top-level configuration for a distillation run."""

    run_name: str = Field(..., description="Unique identifier for this distillation run.")
    teacher_pool: EndpointPoolConfig
    reviewer_pool: Optional[EndpointPoolConfig] = None
    review_flow: ReviewFlowConfig = ReviewFlowConfig()
    prompts: PromptTemplateConfig = PromptTemplateConfig()
    reflection: ReflectionConfig = ReflectionConfig()
    validation: ValidationConfig = ValidationConfig()
    output: OutputConfig = OutputConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    scenarios: list[ScenarioTemplate]
    seed: Optional[int] = Field(None, description="Seed for reproducible sampling.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata stored with each trace."
    )
    model_presets: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Reusable model configuration snippets referenced via the 'preset' key.",
    )

    @model_validator(mode="after")
    def _check_reviewer_requirements(self) -> DistillationConfig:
        if self.review_flow.enabled and self.reviewer_pool is None:
            raise ValueError("Reviewer pool must be configured when review_flow.enabled is true.")
        return self


def load_config(path: Path | str) -> DistillationConfig:
    """Load configuration from a YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    processed = _apply_model_presets(raw)
    return DistillationConfig.model_validate(processed)


def _apply_model_presets(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Expand preset references inside endpoint pools."""

    if not isinstance(raw, dict):
        return raw

    presets = raw.get("model_presets") or {}
    if not presets:
        return raw

    for pool_key in ("teacher_pool", "reviewer_pool"):
        pool = raw.get(pool_key)
        if not pool:
            continue
        endpoints = []
        for endpoint in pool.get("endpoints", []):
            endpoint_data = dict(endpoint)
            preset_name = endpoint_data.pop("preset", None) or endpoint_data.pop("base", None)
            if preset_name:
                preset = presets.get(preset_name)
                if preset is None:
                    raise ValueError(
                        f"Endpoint in pool '{pool_key}' references unknown preset '{preset_name}'."
                    )
                merged = deep_merge_dict(preset, endpoint_data)
            else:
                merged = endpoint_data
            endpoints.append(merged)
        pool["endpoints"] = endpoints
    return raw
