import asyncio
from pathlib import Path

import agent_1.codegen_tools as react_tools
from agentscope.message import ToolUseBlock
from agentscope.tool import ToolResponse

from agent_1.agentscope_tool_loader import load_toolkit_from_config, wrap_tool_function


CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "agentscope_tools_dataset.json")


async def _collect_tool_responses(toolkit, tool_name: str, tool_input: dict) -> list[object]:
    responses = []
    result = await toolkit.call_tool_function(
        ToolUseBlock(
            type="tool_use",
            id="test-call",
            name=tool_name,
            input=tool_input,
        )
    )
    async for item in result:
        responses.append(item)
    return responses


def test_load_toolkit_wraps_plain_dict_return_value_into_tool_response(tmp_path, monkeypatch):
    toolkit = load_toolkit_from_config(CONFIG_PATH)

    fixture = tmp_path / "labs.csv"
    fixture.write_text("patient_id,value\npatient-001,1\npatient-002,2\n", encoding="utf-8")
    monkeypatch.setattr(
        react_tools,
        "_records_path",
        lambda: tmp_path / "reorganized_output" / "_meta" / "records.json",
    )

    responses = asyncio.run(
        _collect_tool_responses(
            toolkit,
            "record_table_split_strategy",
            {"file_path": str(fixture)},
        )
    )

    assert len(responses) == 1
    response = responses[0]
    assert hasattr(response, "content")
    assert "should_split" in str(response.content)


def test_wrapped_plain_return_value_is_truncated(monkeypatch):
    monkeypatch.setenv("AGENT1_TOOL_RESPONSE_TEXT_LIMIT", "1000")

    def fake_tool():
        return {"payload": "x" * 2000}

    response = wrap_tool_function(fake_tool)()

    text = response.content[0]["text"]
    assert len(text) <= 1000
    assert "tool output truncated" in text


async def _call_wrapped_async_tool():
    async def fake_tool():
        return ToolResponse(content=[{"type": "text", "text": "x" * 2000}])

    return await wrap_tool_function(fake_tool)()


def test_wrapped_tool_response_is_truncated(monkeypatch):
    monkeypatch.setenv("AGENT1_TOOL_RESPONSE_TEXT_LIMIT", "1000")

    response = asyncio.run(_call_wrapped_async_tool())

    text = response.content[0]["text"]
    assert len(text) <= 1000
    assert response.metadata["tool_output_truncated"] is True
    assert "tool output truncated" in text
