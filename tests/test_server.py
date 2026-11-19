"""Integration tests for soup-mcp server tools, prompts, and dispatch."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import soup_mcp.server as mcp_mod


@pytest.fixture
def mcp_test_env(tmp_path, monkeypatch):
    """Set up a test working directory with sample dataset files."""
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "train_sample.jsonl"
    rows = [
        {"instruction": f"Solve problem {i}", "input": "", "output": f"Detailed solution {i}."}
        for i in range(1, 11)
    ]
    file_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return file_path


def test_server_lists_all_nine_tools():
    server = mcp_mod.server
    tools = asyncio.run(server.list_tools())
    tool_names = {t.name for t in tools}
    expected = {
        "soup_inspect_dataset",
        "soup_detect_format",
        "soup_clean_dataset",
        "soup_dedup_dataset",
        "soup_sample_dataset",
        "soup_split_dataset",
        "soup_convert_format",
        "soup_merge_datasets",
        "soup_score_dataset",
    }
    assert expected.issubset(tool_names)


def test_server_lists_prompts():
    server = mcp_mod.server
    prompts = asyncio.run(server.list_prompts())
    prompt_names = {p.name for p in prompts}
    assert "audit_dataset" in prompt_names
    assert "prepare_dataset" in prompt_names


def test_get_audit_prompt():
    server = mcp_mod.server
    res = asyncio.run(server.get_prompt("audit_dataset", {"dataset_path": "test.jsonl"}))
    assert res.messages is not None
    text = res.messages[0].content.text
    assert "soup_inspect_dataset" in text
    assert "test.jsonl" in text


def test_get_prepare_prompt():
    server = mcp_mod.server
    res = asyncio.run(server.get_prompt("prepare_dataset", {"dataset_path": "data.jsonl", "target_format": "alpaca"}))
    assert res.messages is not None
    text = res.messages[0].content.text
    assert "data.jsonl" in text
    assert "alpaca" in text


def test_call_tool_detect_format(mcp_test_env):
    server = mcp_mod.server
    result = asyncio.run(server.call_tool("soup_detect_format", {"path": mcp_test_env.name}))
    assert result.is_error is False
    assert result.structured_content is not None
    data = result.structured_content.get("result", result.structured_content)
    assert data["status"] == "ok"
    assert data["format"] == "alpaca"
    assert data["total_rows"] == 10


def test_call_tool_inspect_dataset(mcp_test_env):
    server = mcp_mod.server
    result = asyncio.run(server.call_tool("soup_inspect_dataset", {"path": mcp_test_env.name}))
    assert result.is_error is False
    data = result.structured_content.get("result", result.structured_content)
    assert data["status"] == "ok"
    assert data["valid_rows"] == 10


def test_call_tool_clean_dataset(mcp_test_env):
    server = mcp_mod.server
    result = asyncio.run(
        server.call_tool(
            "soup_clean_dataset",
            {"path": mcp_test_env.name, "output": "cleaned.jsonl", "strip_boilerplate": True},
        )
    )
    assert result.is_error is False
    data = result.structured_content.get("result", result.structured_content)
    assert data["status"] == "ok"
    assert Path("cleaned.jsonl").exists()


def test_call_tool_dedup_dataset(mcp_test_env):
    server = mcp_mod.server
    result = asyncio.run(
        server.call_tool(
            "soup_dedup_dataset",
            {"path": mcp_test_env.name, "output": "deduped.jsonl"},
        )
    )
    assert result.is_error is False
    data = result.structured_content.get("result", result.structured_content)
    assert data["status"] == "ok"
    assert data["unique_rows"] == 10
    assert Path("deduped.jsonl").exists()


def test_call_tool_sample_dataset(mcp_test_env):
    server = mcp_mod.server
    result = asyncio.run(
        server.call_tool(
            "soup_sample_dataset",
            {"path": mcp_test_env.name, "n": 3, "seed": 42},
        )
    )
    assert result.is_error is False
    data = result.structured_content.get("result", result.structured_content)
    assert data["status"] == "ok"
    assert data["sample_count"] == 3


def test_call_tool_split_dataset(mcp_test_env):
    server = mcp_mod.server
    result = asyncio.run(
        server.call_tool(
            "soup_split_dataset",
            {"path": mcp_test_env.name, "val": 20, "test": 10, "seed": 42},
        )
    )
    assert result.is_error is False
    data = result.structured_content.get("result", result.structured_content)
    assert data["status"] == "ok"
    assert data["counts"]["val"] == 2
    assert data["counts"]["test"] == 1
    assert data["counts"]["train"] == 7


def test_call_tool_convert_format(mcp_test_env):
    server = mcp_mod.server
    result = asyncio.run(
        server.call_tool(
            "soup_convert_format",
            {"path": mcp_test_env.name, "to_format": "sharegpt", "output": "converted.jsonl"},
        )
    )
    assert result.is_error is False
    data = result.structured_content.get("result", result.structured_content)
    assert data["status"] == "ok"
    assert data["target_format"] == "sharegpt"
    assert Path("converted.jsonl").exists()


def test_call_tool_merge_datasets(mcp_test_env):
    server = mcp_mod.server
    result = asyncio.run(
        server.call_tool(
            "soup_merge_datasets",
            {"files": [mcp_test_env.name], "output": "merged_single.jsonl"},
        )
    )
    assert result.is_error is False
    data = result.structured_content.get("result", result.structured_content)
    assert data["status"] == "ok"
    assert data["total_rows"] == 10
    assert Path("merged_single.jsonl").exists()


def test_call_tool_score_dataset(mcp_test_env):
    server = mcp_mod.server
    result = asyncio.run(server.call_tool("soup_score_dataset", {"path": mcp_test_env.name}))
    assert result.is_error is False
    data = result.structured_content.get("result", result.structured_content)
    assert data["status"] == "ok"
    assert data["total_rows"] == 10
