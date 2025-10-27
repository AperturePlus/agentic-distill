"""Agentic distillation framework package."""

from .config import (
    DistillationConfig,
    EndpointPoolConfig,
    ModelEndpointConfig,
    PromptTemplateConfig,
    ReviewFlowConfig,
    load_config,
)
from .pipelines.agentic import AgenticDistillationPipeline
from .teacher import TeacherClient, TeacherClientError

__all__ = [
    "DistillationConfig",
    "ModelEndpointConfig",
    "EndpointPoolConfig",
    "ReviewFlowConfig",
    "PromptTemplateConfig",
    "TeacherClient",
    "TeacherClientError",
    "load_config",
    "AgenticDistillationPipeline",
]
