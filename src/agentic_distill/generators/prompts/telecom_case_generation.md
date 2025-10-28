# Telecom Scenario Seed Generation

Create **telecom customer-support incident seeds** that will later be distilled into full agentic traces.

## Requirements

- Output **JSON only** (no prose, no code fences). The top level must be an array of objects.
- Generate between **4 and 6** distinct cases per request.
- Each case object must contain the following keys:
  - `id`: short kebab-case slug (unique within the batch)
  - `issue`: one-sentence description of the escalation
  - `customer_tier`: one of `consumer`, `premium-postpaid`, `enterprise`, `strategic-enterprise`, `industrial`, `municipal`, `wholesale`, `mvno-partner`
  - `region`: geographic region or market (e.g., "APAC", "LATAM", "US Northeast")
  - `symptoms`: array of 2-4 crisp bullet phrases
  - `recent_changes`: array of relevant changes that may have triggered the incident (1-3 elements)
  - `tools`: array naming internal tools or MCP endpoints that should be exercised (3-5 elements)
  - `resolution_objectives`: array describing success criteria / milestones (3-5 elements)
  - `risk_level`: `low`, `medium`, or `high`
  - `evaluation_focus`: array describing automatic checks, telemetry, or QA hooks to judge data quality
  - `telemetry_context`: concise note about live data that should be consulted
  - `language_policy`: always `"en-primary zh-secondary"`
  - `metadata`: object with:
    - `benchmark_alignment`: list of benchmark names that benefit from this case
    - `small_model_targets`: list of compact models that should learn from the trace
    - `novelty_notes`: short note about why this case is different from typical ones

## Scenario quality guardrails

- Cover a mix of network layers (RAN, transport, core, OSS/BSS, device workflows).
- Ensure tool lists are actionable and reference realistic operations consoles, CRM systems, or automation playbooks.
- Avoid reusing the same combination of `issue`, `customer_tier`, and `recent_changes`.
- Highlight risks such as regulatory impact, SLA breaches, or revenue exposure when appropriate.
- Keep every field primarily in English except where pinyin/Chinese terms are natural (rare).

