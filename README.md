# Agentic Distillation Framework

This repository provides a modular framework for distilling high-agency behavioural traces from a frontier model into curated training data for smaller instruction-following models. The framework **only generates datasets**; it does not perform any fine-tuning or optimisation, keeping the distillation and training concerns cleanly separated. The focus is on producing tool-use intensive, multi-step decision traces aligned with benchmarks such as TerminalBench, T^2 Bench, and telecom-themed customer support tasks while enforcing an English-first, Chinese-supported language policy.

## High-Level Architecture

1. **Scenario Registry**  
   Describes the agentic situations we want to elicit (e.g., terminal troubleshooting, telco customer care). Each scenario template captures goals, environment assumptions, tool availability, and evaluation rubrics.
2. **Teacher Orchestrator**  
   Wraps the powerful API (OpenAI, Anthropic, etc.), handles prompt assembly, retries, and optional self-reflection passes to obtain high-quality trajectories.
3. **Trace Builder**  
   Normalizes raw teacher responses into structured episodes: system prompt, user turns, tool invocations, assistant rationales, and final answers. An extensible validator scores outputs against per-scenario rubrics.
4. **Dataset Writer**  
   Streams validated traces into JSONL or Parquet shards, along with metadata (scenario ID, difficulty, score, reflection). Supports incremental refresh and deduplication.
5. **Analytics & QA**  
   Lightweight notebooks/scripts for cohort analysis (coverage across capabilities, score distributions) and automatic spot checks to maintain quality.

```mermaid
flowchart LR
    A[Scenario Registry] --> B[Prompt & Context Builder]
    B --> C[Teacher Orchestrator]
    C --> D[Trace Builder & Validation]
    D --> E[Dataset Writer]
    E --> F[Analytics & QA]
```

## Key Features

- **Agentic Focus:** Scenario templates emphasise tool choice, decision branching, and intermediate reasoning states rather than one-shot responses.
- **Multi-Model Pools:** Weighted teacher and reviewer pools let you mix several frontier endpoints, biasing toward preferred models while retaining fallbacks.
- **Reviewer Refinement:** A second model scores each trace, drives auto-refinement, and enforces strict acceptance thresholds.
- **Language Guardrails:** Global prompts ensure outputs stay primarily in English with succinct Chinese summaries when useful.
- **Parallelised Sampling:** Configurable thread pools keep several scenarios distilling simultaneously for higher throughput.
- **MCP Scenario Library:** Dozens of MCP server dossiers are sampled to generate integration blueprints and prompt-engineering workflows, expanding coverage beyond terminal and telecom domains.
- **Extensible Storage:** Pluggable sinks for JSONL (default), Parquet, or direct uploads to object stores.
- **Benchmark Hooks:** Predefined scenario families inspired by TerminalBench, T^2 Bench, and telecom support flows, ready for expansion.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -e .
   ```
2. Configure your teacher/reviewer pools (see `configs/teacher.example.yaml` for structure).
3. Define a scenario mix and run settings (see `configs/run.terminal.yaml`, `configs/run.telecom.yaml`, or `configs/run.mcp.yaml`).
4. Run the distillation loop:
   ```bash
   python scripts/run_distillation.py --config configs/run.terminal.yaml
   ```

The script will produce JSONL shards under `data/exports/` containing fully validated episodes.

### Configuration Highlights

- `teacher_pool` and `reviewer_pool` describe weighted sets of endpoints. Use `preferred_order` to bias toward your favourite model while keeping alternates ready.
- `review_flow` toggles reviewer scoring, minimum score thresholds, and the number of automated refinement rounds.
- `prompts` injects consistent guidance for teachers and reviewers, including the English-first + Chinese recap requirement.
- `model_presets` lets you define reusable endpoint templates that pools reference via `preset`, keeping YAML DRY while still allowing per-run overrides.
- `concurrency.max_workers` controls how many scenarios run in parallel; tune it to match your API throughput budget.
- Each scenario template can pin custom parameters while inheriting the global language and quality guardrails.
- MCP integration scenarios (`agentic_distill.generators.mcp:MCPScenarioGenerator`) automatically ingest MCP server JSON dossiers and emit rich metadata about the chosen server, mission, and tool focus.

### Preparing Question Banks

**Terminal (SRE) seeds**
1. Generate seeds with the strongest teacher model (`configs/casegen.terminal.yaml`).
   ```bash
   python scripts/generate_cases.py --config configs/casegen.terminal.yaml
   ```
   Accepted cases are appended to `data/question_banks/terminal.jsonl`.
2. Ensure `TerminalScenarioGenerator` references the refreshed bank (default path already matches).

**Telecom support seeds**
1. Generate seeds (`configs/casegen.telecom.yaml`).
   ```bash
   python scripts/generate_cases.py --config configs/casegen.telecom.yaml
   ```
   Accepted cases land in `data/question_banks/telecom.jsonl`.
2. `TelecomScenarioGenerator` reads the same bank by default.

After banks are refreshed, proceed with the standard distillation run.
### Verifying Output Quickly

- Peek at the first few samples:
  ```bash
  head -n 2 data/exports/terminal/shard-00000.jsonl | jq .
  ```
- Confirm scenario coverage, reviewer acceptance rates, and discard reasons via logs emitted during the run.
- Ensure the language mix meets expectations (English narratives with optional Chinese recaps).
- Use the QA checklist (`docs/qa_checklist.md`) after each batch.

### Metadata At A Glance

Every episode now stores structured generation metadata under `metadata.generation`, e.g.:

```json
{
  "generation": {
    "run_name": "mcp-batch-001",
    "teacher": {
      "endpoint": "frontier-default",
      "provider": "openai",
      "model": "gpt-4.1",
      "temperature": 0.16,
      "top_p": 0.9,
      "max_output_tokens": 3584
    },
    "review": [
      {"round": 0, "reviewer_endpoint": "reviewer-judge", "reviewer_model": "gpt-4.1-mini", "score": 0.92, "...": "..."}
    ],
    "reflection_passes": 2,
    "seed": 1234
  },
  "scenario_type": "mcp_integration",
  "language_policy": "en-primary zh-secondary",
  "validation_feedback": "Balanced tool analysis with metadata block."
}
```

Use this block to track which models generated or reviewed each trace and to filter by scenario type or language policy.

## Repository Layout

- `src/agentic_distill/` - Python package with core framework modules.
- `configs/` - Example configs for teacher endpoints and scenario mixes.
- `scripts/` - Entry points for batch distillation and analytics.
- `docs/` - Extended documentation (architecture notes, QA checklist).
- `data/` - Output location for generated datasets (gitignored).

## Next Steps

- Add additional scenario families (finance operations, enterprise IT).
- Integrate automated reward models for finer-grained scoring.
- Hook into evaluation harnesses to automatically re-run TerminalBench/T^2 Bench after each distillation batch.
