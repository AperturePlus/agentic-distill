# Telecom Case Quality Review

Evaluate the proposed telecom incident seed and decide whether it should be added to the question bank.

## Instructions

- Respond **only with JSON** (no commentary, no code fences).
- JSON keys (flat object):
  - `score`: float between 0 and 1
  - `approve`: boolean (`true` if the case should be accepted without edits)
  - `needs_revision`: boolean (`true` if a revision is required before use)
  - `feedback`: concise English critique focusing on gaps or risks
  - `zh_summary`: optional concise Chinese bullet summary (string; may be empty)
  - `tags`: array of classification strings (e.g., `"sla"`, `"oss"`, `"security"`)

## Evaluation Criteria

1. **Originality & Coverage** – Does the case explore a unique combination of symptoms, customers, and tools?
2. **Tool Fidelity** – Are the suggested tools realistic and do they require non-trivial reasoning?
3. **Operational Depth** – Will the downstream agent need multi-step planning, cross-team coordination, and telemetry usage?
4. **Benchmark Fit** – Does the case clearly map to telecom agentic benchmarks or real KPIs?
5. **Language Policy** – Is the case primarily in English with optional concise Chinese references only where natural?

Reject or request revision (`approve=false`) when any of the following occur:
- Duplicate of a previously approved issue.
- Missing or implausible tools / recent changes.
- Vague objectives or risk profile.
- Overly narrow scenario that cannot support multi-step reasoning.

