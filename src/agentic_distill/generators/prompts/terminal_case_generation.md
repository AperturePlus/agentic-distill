# Terminal Scenario Seed Generation

Produce **terminal/SRE troubleshooting seeds** that will later be distilled into high-agency traces.

## Output Format

- Respond with **JSON only** (no commentary, no code fences).
- Top-level must be an array containing **4-6** case objects.
- Each case object **must** include:
  - `id`: kebab-case slug (unique within the batch)
  - `task`: one-sentence problem statement
  - `environment`: string describing the host/cluster context (e.g., "Debian 12, systemd, Kubernetes 1.29")
  - `systems`: array of key components involved (e.g., `["nginx", "postgresql", "istio"]`)
  - `symptoms`: array of precise indicators (3-5 entries)
  - `telemetry_clues`: array naming logs/metrics that showed anomalies
  - `recent_changes`: array of deployments, config edits, or incidents preceding the failure (1-3 items)
  - `tools`: array of commands or scripts that **must** be considered (3-6 items; include arguments/flags where useful)
  - `constraints`: array describing guardrails (e.g., "read-only access", "no service restarts until change window", "limited bandwidth")
  - `objectives`: array of desired outcomes / validation checks (3-5 items)
  - `risk_level`: `low`, `medium`, or `high`
  - `language_policy`: always `"en-primary zh-secondary"`
  - `metadata`: object with:
    - `benchmark_alignment`: list of benchmark names improved by this seed
    - `small_model_targets`: list of compact models to distill toward
    - `novelty_notes`: short explanation of what makes the case distinctive

## Quality Guardrails

- Vary domains (container platforms, CI pipelines, databases, security, networking, auth, storage, data engineering).
- Ensure tool lists require thoughtful sequencing (e.g., `journalctl -u`, `kubectl describe pod`, `diff -u`, `perf stat`).
- Avoid duplication of `task` + `environment` + `recent_changes`.
- Highlight how telemetry clues connect to the root-cause hunt.
- Mention safety/risk considerations (SLA, compliance, data loss) when `risk_level` is `high`.
- Keep content primarily in English; any Chinese should be concise and purposeful.

