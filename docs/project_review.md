# Project Review Summary

## Confirmed Bug Fix
- **Missing `tool_call_id` in serialized episodes**: `Episode.to_serializable` previously dropped the `tool_call_id` field, breaking the linkage between assistant tool calls and the synthetic tool responses injected into the transcript. This made downstream replay/debugging of tool interactions impossible. The serializer now preserves `tool_call_id` whenever it is present so consumers can reliably correlate messages with their tool invocations.

## Potential Improvements & Risk Areas
1. **`OutputConfig.include_metadata` is currently unused**  
   The configuration flag suggests that users can disable metadata emission, but the pipeline always writes the full metadata blob. Either honour the flag during episode construction/writing or remove it to avoid a misleading contract.

2. **Reviewer invocation ignores endpoint tuning**  
   Reviewer calls are hard-coded to `temperature=0.0`, `top_p=0.9`, and `max_output_tokens=1024`. This bypasses parameters configured on `reviewer_pool` endpoints (e.g., higher `max_output_tokens` for verbose judges). Respecting the endpoint configuration would make reviewer behaviour consistent with what users declare in YAML.

3. **Question bank reuse once fingerprints are exhausted**  
   When every fingerprint has been seen, `QuestionBank.sample()` falls back to a pure random choice, which can reintroduce heavily duplicated seeds in long runs. Consider clearing the fingerprint cache or exposing a configurable policy so extended distillation jobs do not regress to high-duplication sampling.

## Additional Observations
- The retry logic in `TeacherClient` only triggers on the custom `TeacherClientError`. Any unexpected exception raised before we wrap it (e.g., serialization bugs) will skip retries entirely; keep this in mind when extending request shaping.
- Reviewer auto-refinement currently stops after the first review whenever `auto_refine` is `False`, even if `max_rounds` allows more reviewer passes. Clarifying the intended behaviour (or documenting it) would help operators plan review budgets.
