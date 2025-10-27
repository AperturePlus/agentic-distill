# QA Checklist for Agentic Distillation

Use this checklist to maintain dataset quality while running the distillation loop.

## Pre-Run

- [ ] Teacher and reviewer API keys exported (`TEACHER_API_KEY`, `REVIEWER_API_KEY`, etc.).
- [ ] Configured `teacher_pool` / `reviewer_pool` reflect desired weighting and preferred order.
- [ ] Scenario configuration reviewed: target episodes, quotas, seeds, and language policy.
- [ ] Concurrency level sized to API rate limits and budget.
- [ ] Optional tool handler connected if executing real commands/APIs.

## During Run

- [ ] Monitor logs for high discard or reviewer rejection rates (>20% is a warning sign).
- [ ] Validate reviewer feedback is being parsed (look for `review_feedback` entries in metadata).
- [ ] Check that tool calls appear in traces when `require_tool_calls` is enabled.
- [ ] Watch per-endpoint latency; adjust pool weights or concurrency if one model throttles.

## Post-Run

- [ ] Inspect shard samples:
  ```bash
  head -n 1 data/exports/terminal/shard-00000.jsonl | jq .
  ```
- [ ] Confirm language mix: English narrative with concise Chinese summaries only.
- [ ] Review reviewer scores versus scenario validation scores for drift.
- [ ] Spot check episodes for hallucinations and red-team failure cases.
- [ ] Version the config file used for the run alongside generated shards metadata.
