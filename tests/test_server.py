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
        "soup_estimate_vram",
        "soup_validate_config",
        "soup_generate_config",
        "soup_diff_datasets",
    }
    assert expected.issubset(tool_names)

    by_name = {t.name: t for t in tools}
    assert getattr(by_name["soup_inspect_dataset"].annotations, "read_only_hint", None) is True
    assert getattr(by_name["soup_estimate_vram"].annotations, "read_only_hint", None) is True
    assert getattr(by_name["soup_clean_dataset"].annotations, "read_only_hint", None) is False
    assert getattr(by_name["soup_generate_config"].annotations, "destructive_hint", None) is False


def test_server_lists_prompts():
    server = mcp_mod.server
    prompts = asyncio.run(server.list_prompts())
    prompt_names = {p.name for p in prompts}
    assert {"audit_dataset", "prepare_dataset", "plan_finetune", "check_data_leakage"}.issubset(prompt_names)


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


def test_server_resources_and_new_tools(mcp_test_env):
    server = mcp_mod.server

    # List & read resources
    resources = asyncio.run(server.list_resources())
    uris = {str(r.uri) for r in resources}
    assert {"soup://formats", "soup://workspace/datasets", "soup://workspace/configs"}.issubset(uris)

    fmt_contents = asyncio.run(server.read_resource("soup://formats"))
    fmt_items = getattr(fmt_contents, "contents", fmt_contents)
    assert "alpaca" in fmt_items[0].content

    ds_contents = asyncio.run(server.read_resource("soup://workspace/datasets"))
    ds_items = getattr(ds_contents, "contents", ds_contents)
    assert mcp_test_env.name in ds_items[0].content

    # Estimate VRAM tool
    vram_res = asyncio.run(
        server.call_tool(
            "soup_estimate_vram",
            {"model": "Qwen/Qwen2.5-7B-Instruct", "quant": "4bit", "seq_len": 1024},
        )
    )
    assert vram_res.is_error is False
    vram_data = vram_res.structured_content.get("result", vram_res.structured_content)
    assert vram_data["status"] == "ok"
    assert vram_data["total_gb"] > 0

    # Generate & validate config tools
    gen_res = asyncio.run(
        server.call_tool(
            "soup_generate_config",
            {
                "data_path": mcp_test_env.name,
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "output": "soup.yaml",
            },
        )
    )
    assert gen_res.is_error is False
    assert Path("soup.yaml").exists()

    val_res = asyncio.run(server.call_tool("soup_validate_config", {"path": "soup.yaml"}))
    assert val_res.is_error is False
    val_data = val_res.structured_content.get("result", val_res.structured_content)
    assert val_data["status"] == "valid"

    # Diff datasets tool
    diff_res = asyncio.run(
        server.call_tool(
            "soup_diff_datasets",
            {"path_a": mcp_test_env.name, "path_b": mcp_test_env.name},
        )
    )
    assert diff_res.is_error is False
    diff_data = diff_res.structured_content.get("result", diff_res.structured_content)
    assert diff_data["overlap_count"] == 10
    assert diff_data["leakage_detected"] is True

    # New prompts
    plan_p = asyncio.run(
        server.get_prompt("plan_finetune", {"dataset_path": "train.jsonl", "model": "Qwen/Qwen2.5-7B"})
    )
    assert "soup_estimate_vram" in plan_p.messages[0].content.text
