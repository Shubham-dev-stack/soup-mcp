"""Tests for soup-mcp core tools, edge cases, and security guards."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from soup_mcp.tools import (
    clean_dataset_tool,
    convert_dataset_tool,
    detect_dataset_tool,
    inspect_dataset_tool,
    score_dataset_tool,
    split_dataset_tool,
)
from soup_mcp.utils import atomic_write_text, ensure_under_cwd, resolve_soup_import


@pytest.fixture
def sample_alpaca_dataset(tmp_path, monkeypatch):
    """Multi-item positive control fixture: valid 5-row Alpaca dataset inside CWD."""
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "alpaca_samples.jsonl"
    rows = [
        {"instruction": f"Question {i}", "input": "", "output": f"Answer {i} with sufficient context."}
        for i in range(1, 6)
    ]
    file_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return file_path


@pytest.fixture
def sample_dirty_dataset(tmp_path, monkeypatch):
    """Dataset with control chars, AI preamble, and clean rows."""
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "dirty_samples.jsonl"
    rows = [
        {"instruction": "Calculate 2+2", "input": "", "output": "As an AI language model, I can say that 2+2 is 4."},
        {"instruction": "Say hello\x00cleanly", "input": "", "output": "Hello world!"},
        {"instruction": "Valid item 1", "input": "", "output": "Pure valid content."},
        {"instruction": "Valid item 2", "input": "", "output": "Another good piece of data."},
        {"instruction": "Valid item 3", "input": "", "output": "Final valid line."},
    ]
    file_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return file_path


class TestDetectDatasetTool:
    def test_detect_format_positive_control(self, sample_alpaca_dataset):
        result = detect_dataset_tool(sample_alpaca_dataset.name)
        assert result["status"] == "ok"
        assert result["format"] == "alpaca"
        assert result["total_rows"] == 5
        assert set(result["columns"]) == {"instruction", "input", "output"}

    def test_detect_format_empty_dataset(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        result = detect_dataset_tool(empty_file.name)
        assert result["status"] == "empty"
        assert result["total_rows"] == 0

    def test_detect_format_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            detect_dataset_tool("nonexistent.jsonl")

    def test_detect_format_rejects_path_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="Path traversal rejected"):
            detect_dataset_tool("../outside.jsonl")


class TestInspectDatasetTool:
    def test_inspect_dataset_multi_item_positive_control(self, sample_alpaca_dataset):
        result = inspect_dataset_tool(sample_alpaca_dataset.name)
        assert result["status"] == "ok"
        assert result["format"] == "alpaca"
        assert result["total_rows"] == 5
        assert result["valid_rows"] == 5
        assert result["duplicates"] == 0
        assert result["avg_length"] > 0

    def test_inspect_dataset_empty_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        res = inspect_dataset_tool(empty_file.name)
        assert res["status"] == "empty"
        assert res["total_rows"] == 0

    def test_inspect_dataset_custom_format_fallback(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        custom_file = tmp_path / "custom.jsonl"
        custom_file.write_text(json.dumps({"arbitrary_key": "val"}) + "\n", encoding="utf-8")
        res = inspect_dataset_tool(custom_file.name)
        assert res["status"] == "ok"
        assert "format" in res

    def test_inspect_dataset_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            inspect_dataset_tool("nonexistent.jsonl")


class TestCleanDatasetTool:
    def test_clean_dataset_multi_item_positive_control(self, sample_dirty_dataset):
        out_name = "cleaned_data.jsonl"
        result = clean_dataset_tool(
            sample_dirty_dataset.name,
            output=out_name,
            strip_boilerplate=True,
        )
        assert result["status"] == "ok"
        assert result["input_file"] == sample_dirty_dataset.name
        assert result["output_file"] == out_name
        assert result["total_scanned"] == 5
        assert Path(out_name).exists()
        cleaned_content = Path(out_name).read_text(encoding="utf-8").strip().splitlines()
        assert len(cleaned_content) == result["output_rows"]

    def test_clean_dataset_default_output_path(self, sample_dirty_dataset):
        result = clean_dataset_tool(sample_dirty_dataset.name)
        assert result["status"] == "ok"
        expected_output = f"{sample_dirty_dataset.stem}_cleaned.jsonl"
        assert result["output_file"] == expected_output
        assert Path(expected_output).exists()

    def test_clean_dataset_dry_run_does_not_write_file(self, sample_dirty_dataset):
        out_name = "should_not_exist.jsonl"
        result = clean_dataset_tool(
            sample_dirty_dataset.name,
            output=out_name,
            dry_run=True,
        )
        assert result["status"] == "ok"
        assert result["dry_run"] is True
        assert not Path(out_name).exists()

    def test_clean_dataset_empty_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Dataset is empty"):
            clean_dataset_tool(empty_file.name)

    def test_clean_dataset_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            clean_dataset_tool("missing.jsonl")

    def test_clean_dataset_rejects_in_place_overwrite(self, sample_dirty_dataset):
        with pytest.raises(ValueError, match="Output path cannot match input path"):
            clean_dataset_tool(sample_dirty_dataset.name, output=sample_dirty_dataset.name)

    def test_clean_dataset_rejects_negative_min_tokens(self, sample_dirty_dataset):
        with pytest.raises(ValueError, match="min_tokens must be non-negative"):
            clean_dataset_tool(sample_dirty_dataset.name, min_tokens=-5)


class TestSplitDatasetTool:
    def test_split_dataset_percentage_positive_control(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        file_path = tmp_path / "data_10.jsonl"
        rows = [{"instruction": f"Q{i}", "input": "", "output": f"A{i}"} for i in range(10)]
        file_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        result = split_dataset_tool(file_path.name, val=20, test=20, seed=42)
        assert result["status"] == "ok"
        assert result["total_rows"] == 10
        assert result["counts"]["val"] == 2
        assert result["counts"]["test"] == 2
        assert result["counts"]["train"] == 6

        # Deterministic verification with same seed
        result_repro = split_dataset_tool(file_path.name, val=20, test=20, seed=42)
        assert result_repro["counts"] == result["counts"]

    def test_split_dataset_absolute_positive_control(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        file_path = tmp_path / "data_10.jsonl"
        rows = [{"instruction": f"Q{i}", "input": "", "output": f"A{i}"} for i in range(10)]
        file_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        result = split_dataset_tool(file_path.name, val=1, test=3, absolute=True, seed=42)
        assert result["counts"]["val"] == 1
        assert result["counts"]["test"] == 3
        assert result["counts"]["train"] == 6

    def test_split_dataset_custom_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sub = tmp_path / "splits_dir"
        sub.mkdir()
        file_path = tmp_path / "data.jsonl"
        rows = [{"instruction": f"Q{i}", "input": "", "output": f"A{i}"} for i in range(10)]
        file_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        result = split_dataset_tool(file_path.name, val=20, test=10, output_dir="splits_dir")
        assert (sub / "data_train.jsonl").exists()
        assert (sub / "data_val.jsonl").exists()
        assert (sub / "data_test.jsonl").exists()

    def test_split_dataset_absolute_exceeding_total_refusal(self, sample_alpaca_dataset):
        with pytest.raises(ValueError, match="must be less than total"):
            split_dataset_tool(sample_alpaca_dataset.name, val=3, test=3, absolute=True)

    def test_split_dataset_neither_val_nor_test_specified(self, sample_alpaca_dataset):
        with pytest.raises(ValueError, match="Specify at least one of val or test"):
            split_dataset_tool(sample_alpaca_dataset.name)

    def test_split_dataset_both_zero_refusal(self, sample_alpaca_dataset):
        with pytest.raises(ValueError, match="Specify at least one positive split"):
            split_dataset_tool(sample_alpaca_dataset.name, val=0, test=0)

    def test_split_dataset_negative_bounds_refusal(self, sample_alpaca_dataset):
        with pytest.raises(ValueError, match="val and test must be non-negative"):
            split_dataset_tool(sample_alpaca_dataset.name, val=-10)

    def test_split_dataset_exceeding_100_percent_refusal(self, sample_alpaca_dataset):
        with pytest.raises(ValueError, match="Split percentages must sum to less than 100"):
            split_dataset_tool(sample_alpaca_dataset.name, val=60, test=45)

    def test_split_dataset_empty_file_refusal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Dataset is empty"):
            split_dataset_tool(empty_file.name, val=20)

    def test_split_dataset_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            split_dataset_tool("missing.jsonl", val=20)


class TestConvertDatasetTool:
    def test_convert_alpaca_to_sharegpt_multi_item_order_preserved(self, sample_alpaca_dataset):
        out_name = "converted_sharegpt.jsonl"
        result = convert_dataset_tool(sample_alpaca_dataset.name, to_format="sharegpt", output=out_name)
        assert result["status"] == "ok"
        assert result["source_format"] == "alpaca"
        assert result["target_format"] == "sharegpt"
        assert result["total_rows"] == 5

        lines = [json.loads(line) for line in Path(out_name).read_text(encoding="utf-8").strip().splitlines()]
        assert len(lines) == 5
        for i, row in enumerate(lines, 1):
            assert "conversations" in row
            assert row["conversations"][0]["value"] == f"Question {i}"
            assert row["conversations"][1]["value"] == f"Answer {i} with sufficient context."

    def test_convert_default_output_name(self, sample_alpaca_dataset):
        result = convert_dataset_tool(sample_alpaca_dataset.name, to_format="chatml")
        assert result["status"] == "ok"
        expected = f"{sample_alpaca_dataset.stem}_chatml.jsonl"
        assert result["output_file"] == expected
        assert Path(expected).exists()

    def test_convert_same_format_returns_noop(self, sample_alpaca_dataset):
        result = convert_dataset_tool(sample_alpaca_dataset.name, to_format="alpaca")
        assert result["status"] == "noop"
        assert "Nothing to convert" in result["message"]

    def test_convert_invalid_format_raises(self, sample_alpaca_dataset):
        with pytest.raises(ValueError, match="Invalid target format"):
            convert_dataset_tool(sample_alpaca_dataset.name, to_format="nonexistent_fmt")

    def test_convert_empty_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Dataset is empty"):
            convert_dataset_tool(empty_file.name, to_format="sharegpt")

    def test_convert_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            convert_dataset_tool("missing.jsonl", to_format="sharegpt")

    def test_convert_output_matching_input_raises(self, sample_alpaca_dataset):
        with pytest.raises(ValueError, match="Output path cannot match input path"):
            convert_dataset_tool(sample_alpaca_dataset.name, to_format="sharegpt", output=sample_alpaca_dataset.name)


class TestScoreDatasetTool:
    def test_score_dataset_positive_control(self, sample_dirty_dataset):
        result = score_dataset_tool(sample_dirty_dataset.name)
        assert result["status"] == "ok"
        assert result["total_rows"] == 5
        assert "pii_flagged" in result
        assert "toxic_flagged" in result
        assert "languages" in result

    def test_score_dataset_empty_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Dataset is empty"):
            score_dataset_tool(empty_file.name)

    def test_score_dataset_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            score_dataset_tool("missing.jsonl")


class TestSecurityAndUtils:
    def test_ensure_under_cwd_allows_nested_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sub = tmp_path / "subfolder"
        sub.mkdir()
        target = sub / "test.jsonl"
        resolved = ensure_under_cwd("subfolder/test.jsonl")
        assert resolved == target.resolve()

    def test_ensure_under_cwd_rejects_parent_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="Path traversal rejected"):
            ensure_under_cwd("../outside.txt")

    def test_atomic_write_text_creates_file_safely(self, tmp_path):
        target = tmp_path / "atomic.txt"
        atomic_write_text("hello atomic world", target)
        assert target.read_text(encoding="utf-8") == "hello atomic world"

    def test_atomic_write_text_cleans_up_on_failure(self, tmp_path):
        target = tmp_path / "atomic_fail.txt"
        with patch("os.replace", side_effect=OSError("Disk full simulation")):
            with pytest.raises(OSError, match="Disk full simulation"):
                atomic_write_text("data", target)
        # Verify no orphaned .tmp files remain in dir
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_resolve_soup_import_runs_cleanly(self):
        resolve_soup_import()
