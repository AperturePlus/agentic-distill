from datetime import datetime

from agentic_distill.storage import DatasetWriter
from agentic_distill.types import Episode, Message, ToolInvocation


def _base_messages(system_prompt: str, user_prompt: str, assistant_reply: str) -> list[Message]:
    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
        Message(role="assistant", content=assistant_reply),
    ]


def test_episode_serialization_contains_rich_metadata() -> None:
    system_prompt = "You are an evaluator."
    user_prompt = "Describe workflows for the Kibela MCP server."
    assistant_reply = (
        "Here is the integration plan referencing search_notes and list_spaces."
        " English narrative. 中文总结：覆盖核心步骤。"
    )

    messages = _base_messages(system_prompt, user_prompt, assistant_reply)
    tool_invocations = [
        ToolInvocation(name="search_notes", arguments={"query": "runbooks"}, success=True)
    ]

    metadata = {
        "language_policy": "en-primary zh-secondary",
        "scenario_type": "mcp_integration",
        "risk_level": "high",
        "mission": "stress-test the server",
        "analysis": "Deep dive into the Kibela server",
        "overview": "Knowledge management",
        "target_benchmark": "AgentBoard",
        "small_model_candidates": ["Qwen2.5-7B-Instruct"],
        "tool_focus": "search_notes",
        "tool_summaries": [
            {"name": "search_notes", "description": "Query notes"},
            {"name": "list_spaces", "description": "List workspaces"},
        ],
        "selected_tool_details": [
            {"name": "search_notes", "description": "Query notes"},
            {"name": "list_spaces", "description": "List workspaces"},
        ],
        "source_server": {
            "name": "Kibela MCP Server",
            "primary_label": "Knowledge Base",
            "secondary_labels": ["Productivity"],
        },
        "validation_feedback": "Reviewer confirmed workflow coverage.",
        "generation": {
            "review": [
                {
                    "score": 0.91,
                    "reviewer_endpoint": "reviewer-a",
                    "needs_revision": False,
                }
            ]
        },
    }

    episode = Episode(
        scenario_id="mcp/kibela-123",
        created_at=datetime.utcnow(),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        messages=messages,
        tool_invocations=tool_invocations,
        score=0.92,
        metadata=metadata,
        available_tools=[
            {"name": "search_notes", "description": "Query notes", "source": "metadata.tool_summaries"},
            {"name": "list_spaces", "description": "List spaces", "source": "metadata.tool_summaries"},
        ],
        target_tools=[{"name": "search_notes"}, {"name": "list_spaces"}],
    )

    payload = episode.to_serializable()

    assert payload["uuid"], "uuid should be populated"
    assert payload["subset"] == "multi_turn"
    assert "mcp" in payload["metadata"], "MCP metadata should be normalised"
    assert payload["metadata"]["scenario_id"] == "mcp/kibela-123"
    assert payload["metadata"]["subset_hint"] in payload["subsets"]
    assert "mcp" in payload["subsets"]
    assert payload["metadata"]["mcp"]["focus"] == ["search_notes"]
    assert payload["question"]["assessments"]["difficulty"]["reason"]
    assert payload["question"]["metadata"]["analysis"] == "Deep dive into the Kibela server"
    tool_alignment = payload["response"]["assessments"]["tool_alignment"]
    assert tool_alignment["value"] == "excellent"
    assert "search_notes" in tool_alignment["reason"]
    assert payload["response"]["assessments"]["language_compliance"]["value"] == "pass"
    assert payload["response"]["metadata"]["generation"]["review"], "review metadata should persist"
    assert payload["response"]["thinking_traces"] == []


def test_dataset_writer_splits_by_target_size(tmp_path) -> None:
    writer = DatasetWriter(
        base_dir=tmp_path,
        format="jsonl",
        shard_size=100,
        target_shard_bytes=800,
    )

    for idx in range(5):
        system_prompt = "You are an orchestrator."
        user_prompt = "Outline the plan." * 5
        assistant_reply = f"Plan step {idx} with bilingual recap. 中文概述。"
        episode = Episode(
            scenario_id=f"scenario-{idx}",
            created_at=datetime.utcnow(),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=_base_messages(system_prompt, user_prompt, assistant_reply),
            tool_invocations=[],
            score=0.8,
            metadata={"validation_feedback": "Looks good."},
        )
        writer.write(episode)

    writer.finalize()

    shards = sorted(tmp_path.glob("*.jsonl"))
    assert len(shards) >= 2
    total_records = 0
    for index, shard in enumerate(shards):
        with shard.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        if index < len(shards) - 1:
            # Intermediate shards should flush once the byte budget is exceeded.
            assert len(lines) <= 1
        total_records += len(lines)

    assert total_records == 5


def test_single_turn_subset_detection() -> None:
    system_prompt = "You are a helpful assistant."
    user_prompt = "Answer briefly."
    assistant_reply = "All done."

    episode = Episode(
        scenario_id="simple/1",
        created_at=datetime.utcnow(),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        messages=_base_messages(system_prompt, user_prompt, assistant_reply),
        metadata={},
    )

    payload = episode.to_serializable()

    assert payload["subsets"] == ["single_turn"]
    assert payload["subset"] == "single_turn"
    assert payload["response"]["thinking_traces"] == []


def test_thinking_segments_are_preserved() -> None:
    system_prompt = "System priming"
    user_prompt = "Explain the approach."
    assistant_content = [
        {"type": "thinking", "text": "Evaluating requirements."},
        {"type": "output_text", "text": "Final answer ready."},
    ]

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
        Message(role="assistant", content=assistant_content),
    ]

    episode = Episode(
        scenario_id="thinking/1",
        created_at=datetime.utcnow(),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        messages=messages,
        metadata={
            "generation": {"teacher": {"mode": "thinking"}},
        },
    )

    payload = episode.to_serializable()

    assert payload["subset"] == "single_turn"
    assert "thinking_model" in payload["subsets"]
    assert payload["response"]["final_answer"] == "Final answer ready."
    thinking_traces = payload["response"]["thinking_traces"]
    assert len(thinking_traces) == 1
    assert thinking_traces[0]["turn"] == 2
    assert thinking_traces[0]["segments"][0]["text"] == "Evaluating requirements."
