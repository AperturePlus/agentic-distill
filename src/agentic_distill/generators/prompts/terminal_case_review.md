# Terminal Case Quality Review

Judge whether a proposed terminal/SRE troubleshooting seed should enter the question bank.

## Instructions

- Respond **only with JSON** (flat object, no code fences).
- Keys:
  - `score`: float 0-1
  - `approve`: boolean
  - `needs_revision`: boolean
  - `feedback`: concise English critique (mention missing signal/tooling gaps)
  - `zh_summary`: optional short Chinese bullet summary (string; may be empty)
  - `tags`: array of labels (e.g., `"kubernetes"`, `"perf"`, `"security"`, `"db"`)

## Acceptance Criteria

Approve only if:
1. **Novel coverage** – case introduces a meaningful variation in systems, tooling, or risk.
2. **Tool fidelity** – commands are realistic, require reasoning (no trivial one-liners).
3. **Operational depth** – objectives/constraints force multi-step forensic analysis.
4. **Telemetry alignment** – clues tie directly to logs/metrics that help resolve the issue.
5. **Benchmark fit** – metadata clearly maps to agentic/terminal benchmarks and small-model goals.

Reject or flag for revision when:
- Tool list is shallow or irrelevant.
- Task duplicates an existing pattern (same task + environment + recent changes).
- Objectives lack verifiable outcomes.
- Constraints are missing for high-risk incidents.
- Content violates the English-primary policy.
