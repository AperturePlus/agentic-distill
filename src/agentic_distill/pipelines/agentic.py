"""Main orchestration pipeline for agentic distillation."""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from rich.console import Console
from rich.progress import Progress, TaskID

from ..config import (
    DistillationConfig,
    EndpointPoolConfig,
    ModelEndpointConfig,
    ScenarioTemplate,
)
from ..generators.base import ScenarioGenerator, ScenarioSample
from ..review import ReviewFeedback, parse_review_feedback
from ..storage import DatasetWriter
from ..teacher import TeacherClient, TeacherClientError
from ..types import Episode, Message, ToolInvocation
from ..utils import import_from_path

ToolHandler = Callable[[str, Dict[str, Any]], Dict[str, Any]]


class EndpointSelector:
    """Selects endpoints according to pool strategy and preferences."""

    def __init__(self, pool: EndpointPoolConfig, *, seed: Optional[int] = None):
        self.pool = pool
        self._random = random.Random(seed)
        self._round_robin_index = 0
        self._preferred_index = 0
        self._lookup = {endpoint.name: endpoint for endpoint in pool.endpoints}

    def select(self) -> ModelEndpointConfig:
        if self.pool.preferred_order:
            for _ in range(len(self.pool.preferred_order)):
                name = self.pool.preferred_order[self._preferred_index % len(self.pool.preferred_order)]
                self._preferred_index += 1
                endpoint = self._lookup.get(name)
                if endpoint is not None:
                    return endpoint

        if self.pool.selection_strategy == "round_robin":
            endpoint = self.pool.endpoints[self._round_robin_index % len(self.pool.endpoints)]
            self._round_robin_index += 1
            return endpoint

        total_weight = sum(endpoint.weight for endpoint in self.pool.endpoints)
        threshold = self._random.uniform(0, total_weight)
        cumulative = 0.0
        for endpoint in self.pool.endpoints:
            cumulative += endpoint.weight
            if threshold <= cumulative:
                return endpoint
        return self.pool.endpoints[-1]


class AgenticDistillationPipeline:
    """Coordinates scenario sampling, teacher/reviewer calls, validation, and storage."""

    def __init__(
        self,
        config: DistillationConfig,
        *,
        tool_handler: Optional[ToolHandler] = None,
    ):
        self.config = config
        self.tool_handler = tool_handler
        self.console = Console()
        self.writer = DatasetWriter(
            base_dir=config.output.base_dir,
            format=config.output.format,
            shard_size=config.output.shard_size,
        )

        self.random = random.Random(config.seed)
        self.generators = self._instantiate_generators(config.scenarios)
        self.generator_locks = {name: threading.Lock() for name in self.generators}

        self.teacher_selector = EndpointSelector(config.teacher_pool, seed=config.seed)
        self.teacher_clients = {
            endpoint.name: TeacherClient(endpoint) for endpoint in config.teacher_pool.endpoints
        }

        self.reviewer_selector: Optional[EndpointSelector] = None
        self.reviewer_clients: Dict[str, TeacherClient] = {}
        if config.review_flow.enabled and config.reviewer_pool:
            self.reviewer_selector = EndpointSelector(config.reviewer_pool, seed=(config.seed or 0) + 1)
            self.reviewer_clients = {
                endpoint.name: TeacherClient(endpoint) for endpoint in config.reviewer_pool.endpoints
            }

        self._progress_state: Dict[str, int] = {template.name: 0 for template in config.scenarios}
        self._targets = {template.name: template.target_episodes for template in config.scenarios}

    def run(self) -> Dict[str, int]:
        """Execute the distillation loop until all scenario quotas are satisfied."""

        with Progress(console=self.console) as progress:
            task_ids: Dict[str, TaskID] = {
                template.name: progress.add_task(f"[cyan]{template.name}", total=template.target_episodes)
                for template in self.config.scenarios
            }

            try:
                with ThreadPoolExecutor(max_workers=self.config.concurrency.max_workers) as executor:
                    inflight: set[Future[Tuple[str, Optional[Episode]]]] = set()

                    while inflight or not self._targets_reached():
                        while (
                            len(inflight) < self.config.concurrency.max_workers
                            and not self._targets_reached()
                        ):
                            template = self._choose_next_template()
                            if template is None:
                                break
                            future = executor.submit(self._produce_episode, template)
                            inflight.add(future)

                        if not inflight:
                            break

                        done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                        for future in done:
                            inflight.remove(future)
                            try:
                                scenario_name, episode = future.result()
                            except TeacherClientError as exc:
                                self.console.log(f"[red]Teacher call failed: {exc}")
                                time.sleep(1.0)
                                continue
                            except Exception as exc:  # noqa: BLE001
                                self.console.log(f"[red]Unexpected error in worker: {exc}")
                                continue

                            if episode is None:
                                continue

                            self.writer.write(episode)
                            self._progress_state[scenario_name] += 1
                            progress.advance(task_ids[scenario_name], 1)
            finally:
                self.writer.finalize()
                self._close_clients()

        return self._progress_state

    def _instantiate_generators(self, templates: Iterable[ScenarioTemplate]) -> Dict[str, ScenarioGenerator]:
        generators: Dict[str, ScenarioGenerator] = {}
        for index, template in enumerate(templates):
            factory = import_from_path(template.generator)
            seed = (self.config.seed or 0) + index if self.config.seed is not None else None
            generators[template.name] = factory(seed=seed, **template.params)
        return generators

    def _close_clients(self) -> None:
        for client in self.teacher_clients.values():
            client.close()
        for client in self.reviewer_clients.values():
            client.close()

    def _choose_next_template(self) -> Optional[ScenarioTemplate]:
        remaining = [
            template
            for template in self.config.scenarios
            if self._progress_state[template.name] < template.target_episodes
        ]
        if not remaining:
            return None
        total_weight = sum(template.weight for template in remaining)
        threshold = self.random.uniform(0, total_weight)
        cumulative = 0.0
        for template in remaining:
            cumulative += template.weight
            if threshold <= cumulative:
                return template
        return remaining[-1]

    def _produce_episode(self, template: ScenarioTemplate) -> Tuple[str, Optional[Episode]]:
        generator = self.generators[template.name]
        with self.generator_locks[template.name]:
            sample = generator.sample()

        system_prompt = self._compose_system_prompt(sample.system_prompt)
        user_prompt = self._compose_user_prompt(sample.user_prompt)
        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tool_invocations: List[ToolInvocation] = []

        teacher_endpoint = self.teacher_selector.select()
        teacher_client = self.teacher_clients[teacher_endpoint.name]

        response_message = self._call_teacher(
            client=teacher_client,
            endpoint=teacher_endpoint,
            messages=conversation,
            tools=sample.tools,
        )
        conversation.append(response_message)
        self._record_tool_calls(response_message, tool_invocations, conversation)

        if self.config.reflection.enabled and self.config.reflection.passes > 0:
            for idx in range(self.config.reflection.passes):
                reflection_prompt = self._build_reflection_prompt(idx)
                conversation.append({"role": "user", "content": reflection_prompt})
                reflection_message = self._call_teacher(
                    client=teacher_client,
                    endpoint=teacher_endpoint,
                    messages=conversation,
                    tools=sample.tools,
                )
                conversation.append(reflection_message)
                self._record_tool_calls(reflection_message, tool_invocations, conversation)

        review_records = self._maybe_run_review_cycle(
            conversation=conversation,
            sample=sample,
            teacher_client=teacher_client,
            teacher_endpoint=teacher_endpoint,
            tool_invocations=tool_invocations,
        )

        validation = generator.validate(conversation[2:], sample.metadata or {})
        if validation.require_retry or validation.score < self.config.validation.min_score:
            self.console.log(
                f"[yellow]Discarding episode {sample.scenario_id} "
                f"(score={validation.score:.2f}, feedback={validation.feedback})"
            )
            return template.name, None

        if self.config.review_flow.enabled and review_records:
            final_review = review_records[-1]
            if (
                final_review["score"] < self.config.review_flow.min_score
                or final_review["needs_revision"]
            ):
                self.console.log(
                    f"[yellow]Discarding episode {sample.scenario_id} "
                    f"after reviewer scored {final_review['score']:.2f}."
                )
                return template.name, None

        episode = Episode(
            scenario_id=sample.scenario_id,
            created_at=datetime.utcnow(),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=[self._to_message(msg) for msg in conversation],
            tool_invocations=tool_invocations,
            score=validation.score,
            metadata={
                **(sample.metadata or {}),
                "feedback": validation.feedback,
                "teacher_endpoint": teacher_endpoint.name,
                "review_feedback": review_records,
                "run_name": self.config.run_name,
            },
        )
        return template.name, episode

    def _call_teacher(
        self,
        *,
        client: TeacherClient,
        endpoint: ModelEndpointConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        response = client.generate(
            messages=messages,
            tools=tools,
            temperature=endpoint.temperature,
            top_p=endpoint.top_p,
            max_output_tokens=endpoint.max_output_tokens,
        )
        return response["choices"][0]["message"]

    def _record_tool_calls(
        self,
        message: Dict[str, Any],
        tool_invocations: List[ToolInvocation],
        conversation: List[Dict[str, Any]],
    ) -> None:
        tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "unknown_tool")
            raw_arguments = function.get("arguments", {})
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": raw_arguments}
            else:
                arguments = raw_arguments

            invocation = ToolInvocation(name=name, arguments=arguments)

            if self.tool_handler:
                try:
                    result = self.tool_handler(name, arguments)
                    if not isinstance(result, dict):
                        result = {"content": str(result)}
                    invocation.output = result
                    invocation.success = True
                    conversation.append(
                        {
                            "role": "tool",
                            "name": name,
                            "content": result.get("content", ""),
                            "tool_call_id": call.get("id"),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    invocation.output = {"error": str(exc)}
                    invocation.success = False
                    conversation.append(
                        {
                            "role": "tool",
                            "name": name,
                            "content": f"Tool execution failed: {exc}",
                            "tool_call_id": call.get("id"),
                        }
                    )
            tool_invocations.append(invocation)

    def _maybe_run_review_cycle(
        self,
        *,
        conversation: List[Dict[str, Any]],
        sample: ScenarioSample,
        teacher_client: TeacherClient,
        teacher_endpoint: ModelEndpointConfig,
        tool_invocations: List[ToolInvocation],
    ) -> List[Dict[str, Any]]:
        if not self.config.review_flow.enabled or not self.reviewer_selector:
            return []

        review_records: List[Dict[str, Any]] = []

        for round_index in range(self.config.review_flow.max_rounds + 1):
            reviewer_endpoint = self.reviewer_selector.select()
            reviewer_client = self.reviewer_clients[reviewer_endpoint.name]

            review_messages = [
                {"role": "system", "content": self.config.prompts.reviewer_template},
                {
                    "role": "user",
                    "content": self._build_reviewer_prompt(conversation, sample, round_index),
                },
            ]

            review_response = reviewer_client.generate(
                messages=review_messages,
                temperature=0.0,
                top_p=0.9,
                max_output_tokens=1024,
            )
            review_message = review_response["choices"][0]["message"]
            feedback = parse_review_feedback(review_message.get("content", ""))
            review_record = {
                "round": round_index,
                "reviewer_endpoint": reviewer_endpoint.name,
                "score": feedback.score,
                "needs_revision": feedback.needs_revision,
                "feedback": feedback.feedback,
                "chinese_summary": feedback.chinese_summary,
            }
            review_records.append(review_record)

            if (
                feedback.score >= self.config.review_flow.min_score
                and not feedback.needs_revision
            ):
                break

            allows_revision = (
                self.config.review_flow.auto_refine
                and round_index < self.config.review_flow.max_rounds
            )
            if not allows_revision:
                break

            revision_prompt = self.config.prompts.revision_template.format(
                feedback=self._format_feedback_for_revision(feedback)
            )
            conversation.append({"role": "user", "content": revision_prompt})
            revision_message = self._call_teacher(
                client=teacher_client,
                endpoint=teacher_endpoint,
                messages=conversation,
                tools=sample.tools,
            )
            conversation.append(revision_message)
            self._record_tool_calls(revision_message, tool_invocations, conversation)

        return review_records

    def _build_reflection_prompt(self, pass_index: int) -> str:
        styles = {
            "default": (
                "Review your previous answer. Identify mistakes or missing steps. "
                "Revise the response to be explicit about tool usage and decision justifications."
            ),
            "concise": "Check your last answer for gaps. Provide a crisp, corrected plan.",
            "exhaustive": (
                "Examine every assumption in your last answer. Correct errors, fill in missing command outputs, "
                "and ensure mitigation guidance is actionable. Provide both English reasoning and a short Chinese recap."
            ),
        }
        base_prompt = styles.get(self.config.reflection.critique_style, styles["default"])
        return f"Reflection pass {pass_index + 1}: {base_prompt}"

    def _build_reviewer_prompt(
        self,
        conversation: List[Dict[str, Any]],
        sample: ScenarioSample,
        round_index: int,
    ) -> str:
        transcript = json.dumps(
            [
                {
                    "role": msg.get("role"),
                    "name": msg.get("name"),
                    "content": msg.get("content"),
                    "tool_calls": msg.get("tool_calls"),
                }
                for msg in conversation
            ],
            ensure_ascii=False,
            indent=2,
        )
        return (
            f"Scenario ID: {sample.scenario_id}\n"
            f"Round: {round_index}\n"
            f"Metadata: {json.dumps(sample.metadata, ensure_ascii=False)}\n"
            f"Transcript JSON:\n{transcript}\n"
            "Assess the assistant's latest answer for correctness, completeness, and agentic decision making."
        )

    @staticmethod
    def _format_feedback_for_revision(feedback: ReviewFeedback) -> str:
        chinese = f"\n中文摘要: {feedback.chinese_summary}" if feedback.chinese_summary else ""
        return f"{feedback.feedback}{chinese}"

    @staticmethod
    def _targets_reached_static(progress: Dict[str, int], targets: Dict[str, int]) -> bool:
        return all(progress[name] >= targets[name] for name in targets)

    def _targets_reached(self) -> bool:
        return self._targets_reached_static(self._progress_state, self._targets)

    def _compose_system_prompt(self, original: str) -> str:
        prefix = self.config.prompts.global_system_prefix.strip()
        if not prefix:
            return original
        return f"{prefix}\n\n{original}".strip()

    def _compose_user_prompt(self, original: str) -> str:
        guidelines = self.config.prompts.user_guidelines.strip()
        if not guidelines:
            return original
        return f"{guidelines}\n\n{original}".strip()

    @staticmethod
    def _to_message(data: Dict[str, Any]) -> Message:
        return Message(
            role=data.get("role", ""),
            content=data.get("content") or "",
            name=data.get("name"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
        )

