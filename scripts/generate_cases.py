"""Generate question-bank seeds with teacher/reviewer loops (configurable across domains)."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from rich.console import Console
from rich.progress import Progress

from agentic_distill.config import EndpointPoolConfig, ModelEndpointConfig
from agentic_distill.review import parse_review_feedback
from agentic_distill.teacher import TeacherClient, TeacherClientError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate telecom scenario seeds for the question bank.")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to the case generation YAML configuration.",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def choose_primary_endpoint(pool: EndpointPoolConfig) -> ModelEndpointConfig:
    if pool.preferred_order:
        for name in pool.preferred_order:
            for endpoint in pool.endpoints:
                if endpoint.name == name:
                    return endpoint
    # Weight prioritisation: higher weight, then longer contexts
    return max(pool.endpoints, key=lambda ep: (ep.weight, ep.max_output_tokens))


def choose_reviewer_endpoint(pool: EndpointPoolConfig) -> ModelEndpointConfig:
    if pool.preferred_order:
        for name in pool.preferred_order:
            for endpoint in pool.endpoints:
                if endpoint.name == name:
                    return endpoint
    return pool.endpoints[0]


def extract_json_candidates(text: str) -> Any:
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    candidates = fenced if fenced else [text]
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Unable to parse JSON from model output.")


def load_existing_entries(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(data)
    return entries


def build_dedupe_key(entry: Dict[str, Any], dedupe_fields: Iterable[str]) -> Tuple:
    key = []
    for field in dedupe_fields:
        value = entry.get(field)
        if isinstance(value, list):
            value = tuple(value)
        key.append(value)
    return tuple(key)


def main() -> None:
    args = parse_args()
    console = Console()
    cfg = load_yaml_config(args.config)

    run_name: str = cfg["run_name"]
    target_count: int = cfg.get("target_count", 200)
    batch_size: int = cfg.get("batch_size", 5)
    min_review_score: float = cfg.get("min_review_score", 0.85)
    output_path = Path(cfg["output_path"])
    generation_prompt_path = Path(cfg["generation_prompt"])
    review_prompt_path = Path(cfg["review_prompt"])
    dedupe_fields = cfg.get("dedupe_keys", ["issue", "customer_tier", "region"])
    prompt_variables = cfg.get("prompt_variables", {})
    prompt_variables.setdefault("batch_size", batch_size)
    prompt_variables.setdefault("target_count", target_count)

    generation_prompt = generation_prompt_path.read_text(encoding="utf-8")
    review_prompt = review_prompt_path.read_text(encoding="utf-8")
    system_prompt = cfg.get(
        "system_prompt",
        "You are a scenario designer generating high-agency seeds for dataset distillation. "
        "Follow the JSON-only output requirements exactly.",
    )

    teacher_pool = EndpointPoolConfig.model_validate(cfg["teacher_pool"])
    reviewer_pool = EndpointPoolConfig.model_validate(cfg["reviewer_pool"])
    teacher_endpoint = choose_primary_endpoint(teacher_pool)
    reviewer_endpoint = choose_reviewer_endpoint(reviewer_pool)

    existing_entries = load_existing_entries(output_path)
    dedupe_set = {build_dedupe_key(entry, dedupe_fields) for entry in existing_entries}
    with output_path.open("a", encoding="utf-8") as out_file, TeacherClient(teacher_endpoint) as teacher, TeacherClient(
        reviewer_endpoint
    ) as reviewer, Progress(console=console) as progress:
        task_id = progress.add_task(
            f"[cyan]Generating cases ({run_name})",
            total=target_count,
            completed=len(existing_entries),
        )

        accepted = len(existing_entries)
        failure_streak = 0
        while accepted < target_count:
            user_prompt = generation_prompt.format(**prompt_variables)
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ]

            try:
                response = teacher.generate(
                    messages=messages,
                    temperature=teacher_endpoint.temperature,
                    top_p=teacher_endpoint.top_p,
                    max_output_tokens=teacher_endpoint.max_output_tokens,
                )
            except TeacherClientError as exc:
                console.log(f"[red]Teacher generation failed: {exc}")
                failure_streak += 1
                if failure_streak > 5:
                    raise
                continue

            teacher_message = response["choices"][0]["message"].get("content", "")
            try:
                batch = extract_json_candidates(teacher_message)
            except ValueError as exc:
                console.log(f"[yellow]Skipping batch due to parse error: {exc}")
                failure_streak += 1
                continue

            if not isinstance(batch, list):
                console.log("[yellow]Teacher did not return a list; skipping batch.")
                failure_streak += 1
                continue

            for raw_case in batch:
                if not isinstance(raw_case, dict):
                    continue
                case = dict(raw_case)
                case.setdefault("id", case.get("uid"))
                if not case.get("id"):
                    slug_base = (case.get("issue") or "case").lower().replace(" ", "-")
                    case["id"] = f"{slug_base[:32]}-{accepted}"

                dedupe_key = build_dedupe_key(case, dedupe_fields)
                if dedupe_key in dedupe_set:
                    continue

                review_payload = {
                    "run_name": run_name,
                    "case": case,
                }
                review_messages = [
                    {
                        "role": "system",
                        "content": review_prompt,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(review_payload, ensure_ascii=False),
                    },
                ]

                try:
                    review_response = reviewer.generate(
                        messages=review_messages,
                        temperature=reviewer_endpoint.temperature,
                        top_p=reviewer_endpoint.top_p,
                        max_output_tokens=reviewer_endpoint.max_output_tokens,
                    )
                except TeacherClientError as exc:
                    console.log(f"[yellow]Reviewer call failed: {exc}")
                    continue

                reviewer_message = review_response["choices"][0]["message"].get("content", "")
                feedback = parse_review_feedback(reviewer_message)
                if feedback.score < min_review_score or feedback.needs_revision:
                    continue

                case.setdefault("language_policy", "en-primary zh-secondary")
                case.setdefault("metadata", {})
                case["metadata"]["generation"] = {
                    "run_name": run_name,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "teacher_model": teacher_endpoint.model,
                    "reviewer_model": reviewer_endpoint.model,
                    "review_score": feedback.score,
                    "review_feedback": feedback.feedback,
                }
                if feedback.chinese_summary:
                    case["metadata"]["review_zh"] = feedback.chinese_summary

                out_file.write(json.dumps(case, ensure_ascii=False) + "\n")
                out_file.flush()
                dedupe_set.add(dedupe_key)
                accepted += 1
                progress.update(task_id, advance=1)

                if accepted >= target_count:
                    break

            failure_streak = 0

    console.log(f"[green]Case generation complete. Total accepted entries: {accepted}")


if __name__ == "__main__":
    main()
