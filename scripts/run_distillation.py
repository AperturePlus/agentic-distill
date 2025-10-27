"""CLI entry point for running the agentic distillation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from agentic_distill import AgenticDistillationPipeline, DistillationConfig
from agentic_distill.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agentic data distillation.")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to the distillation run configuration (YAML).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    distill_config: DistillationConfig = load_config(args.config)
    reviewer_state = "enabled" if distill_config.review_flow.enabled else "disabled"
    console.log(
        "Loaded configuration for run "
        f"'{distill_config.run_name}' with {distill_config.concurrency.max_workers} worker(s); "
        f"reviewer flow {reviewer_state}."
    )

    pipeline = AgenticDistillationPipeline(config=distill_config)
    progress = pipeline.run()

    console.log("Distillation run complete:")
    for scenario_name, count in progress.items():
        console.log(f"  - {scenario_name}: {count} episodes")


if __name__ == "__main__":
    main()

