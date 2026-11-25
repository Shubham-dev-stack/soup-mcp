"""Tests for soup-mcp core tools, edge cases, and security guards."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from soup_mcp.tools import (
    clean_dataset_tool,
    convert_dataset_tool,
    dedup_dataset_tool,
    detect_dataset_tool,
    diff_datasets_tool,
    estimate_vram_tool,
    generate_config_tool,
    inspect_dataset_tool,
    list_formats_resource,
    list_workspace_configs_resource,
    list_workspace_datasets_resource,
    merge_datasets_tool,
    sample_dataset_tool,
    score_dataset_tool,
    split_dataset_tool,
    validate_config_tool,
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

        split_dataset_tool(file_path.name, val=20, test=10, output_dir="splits_dir")
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


class TestDedupDatasetTool:
    def test_dedup_dataset_multi_item_positive_control(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        file_path = tmp_path / "dup_data.jsonl"
        rows = [
            {"instruction": "Q1", "input": "", "output": "A1"},
            {"instruction": "Q2", "input": "", "output": "A2"},
            {"instruction": "Q1", "input": "", "output": "A1"},  # duplicate
            {"instruction": "Q3", "input": "", "output": "A3"},
            {"instruction": "Q2", "input": "", "output": "A2"},  # duplicate
        ]
        file_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        result = dedup_dataset_tool(file_path.name)
        assert result["status"] == "ok"
        assert result["total_scanned"] == 5
        assert result["unique_rows"] == 3
        assert result["duplicates_dropped"] == 2
        assert Path(result["output_file"]).exists()

        # Verify exact preserved order
        out_rows = [json.loads(line) for line in Path(result["output_file"]).read_text(encoding="utf-8").strip().splitlines()]
        assert len(out_rows) == 3
        assert [r["instruction"] for r in out_rows] == ["Q1", "Q2", "Q3"]

    def test_dedup_dataset_by_field(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        file_path = tmp_path / "dup_field.jsonl"
        rows = [
            {"instruction": "Same Prompt", "output": "Answer 1"},
            {"instruction": "Same Prompt", "output": "Answer 2 (different)"},
            {"instruction": "Unique Prompt", "output": "Answer 3"},
        ]
        file_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        result = dedup_dataset_tool(file_path.name, field="instruction")
        assert result["unique_rows"] == 2
        assert result["duplicates_dropped"] == 1

    def test_dedup_dataset_field_absent_raises(self, sample_alpaca_dataset):
        with pytest.raises(ValueError, match="is absent from all rows"):
            dedup_dataset_tool(sample_alpaca_dataset.name, field="nonexistent_field")

    def test_dedup_dataset_empty_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Dataset is empty"):
            dedup_dataset_tool(empty_file.name)

    def test_dedup_dataset_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            dedup_dataset_tool("missing.jsonl")

    def test_dedup_dataset_in_place_overwrite_raises(self, sample_alpaca_dataset):
        with pytest.raises(ValueError, match="Output path cannot match input path"):
            dedup_dataset_tool(sample_alpaca_dataset.name, output=sample_alpaca_dataset.name)


class TestSampleDatasetTool:
    def test_sample_dataset_by_count_positive_control(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        file_path = tmp_path / "data_10.jsonl"
        rows = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(10)]
        file_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        result = sample_dataset_tool(file_path.name, n=3, seed=42)
        assert result["status"] == "ok"
        assert result["total_rows"] == 10
        assert result["sample_count"] == 3
        assert Path(result["output_file"]).exists()

        # Seed reproducibility
        result2 = sample_dataset_tool(file_path.name, n=3, seed=42)
        lines1 = Path(result["output_file"]).read_text(encoding="utf-8")
        lines2 = Path(result2["output_file"]).read_text(encoding="utf-8")
        assert lines1 == lines2

    def test_sample_dataset_by_pct(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        file_path = tmp_path / "data_10.jsonl"
        rows = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(10)]
        file_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        result = sample_dataset_tool(file_path.name, pct=50.0, seed=42)
        assert result["sample_count"] == 5

    def test_sample_dataset_invalid_parameters_raise(self, sample_alpaca_dataset):
        with pytest.raises(ValueError, match="Specify either n"):
            sample_dataset_tool(sample_alpaca_dataset.name)
        with pytest.raises(ValueError, match="n must be a positive integer"):
            sample_dataset_tool(sample_alpaca_dataset.name, n=-1)
        with pytest.raises(ValueError, match="pct must be a number in"):
            sample_dataset_tool(sample_alpaca_dataset.name, pct=150.0)

    def test_sample_dataset_empty_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Dataset is empty"):
            sample_dataset_tool(empty.name, n=2)

    def test_sample_dataset_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            sample_dataset_tool("missing.jsonl", n=2)

    def test_sample_dataset_in_place_overwrite_raises(self, sample_alpaca_dataset):
        with pytest.raises(ValueError, match="Output path cannot match input path"):
            sample_dataset_tool(sample_alpaca_dataset.name, n=2, output=sample_alpaca_dataset.name)


class TestMergeDatasetsTool:
    def test_merge_datasets_multi_item_positive_control(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f1 = tmp_path / "part1.jsonl"
        f2 = tmp_path / "part2.jsonl"
        f3 = tmp_path / "part3.jsonl"
        f1.write_text(json.dumps({"id": 1}) + "\n" + json.dumps({"id": 2}) + "\n", encoding="utf-8")
        f2.write_text(json.dumps({"id": 3}) + "\n", encoding="utf-8")
        f3.write_text(json.dumps({"id": 4}) + "\n" + json.dumps({"id": 5}) + "\n", encoding="utf-8")

        result = merge_datasets_tool([f1.name, f2.name, f3.name], output="combined.jsonl")
        assert result["status"] == "ok"
        assert result["total_rows"] == 5
        assert Path("combined.jsonl").exists()

        rows = [json.loads(line) for line in Path("combined.jsonl").read_text(encoding="utf-8").strip().splitlines()]
        assert [r["id"] for r in rows] == [1, 2, 3, 4, 5]

    def test_merge_datasets_with_shuffle(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f1 = tmp_path / "part1.jsonl"
        rows = [{"id": i} for i in range(20)]
        f1.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        result = merge_datasets_tool([f1.name], output="shuffled.jsonl", shuffle=True, seed=42)
        assert result["shuffled"] is True
        assert result["total_rows"] == 20

    def test_merge_datasets_empty_files_list_raises(self):
        with pytest.raises(ValueError, match="files list must not be empty"):
            merge_datasets_tool([])

    def test_merge_datasets_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            merge_datasets_tool(["nonexistent1.jsonl"])

    def test_merge_datasets_output_matching_input_raises(self, sample_alpaca_dataset):
        with pytest.raises(ValueError, match="Output path cannot match input file"):
            merge_datasets_tool([sample_alpaca_dataset.name], output=sample_alpaca_dataset.name)

    def test_merge_datasets_all_empty_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        e1 = tmp_path / "e1.jsonl"
        e1.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No data loaded"):
            merge_datasets_tool([e1.name])


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


class TestEstimateVramTool:
    def test_estimate_vram_quant_ordering_positive_control(self):
        res_4bit = estimate_vram_tool("Qwen/Qwen2.5-7B-Instruct", seq_len=1024, batch_size=1, quant="4bit")
        res_8bit = estimate_vram_tool("Qwen/Qwen2.5-7B-Instruct", seq_len=1024, batch_size=1, quant="8bit")
        res_fp16 = estimate_vram_tool("Qwen/Qwen2.5-7B-Instruct", seq_len=1024, batch_size=1, quant="none")

        assert res_4bit["status"] == "ok"
        assert res_4bit["params_b"] == 7.0
        assert res_4bit["total_gb"] < res_8bit["total_gb"] < res_fp16["total_gb"]
        assert res_4bit["recommended_vram_gb"] > res_4bit["total_gb"]
        assert "T4 / RTX 4060 Ti (16GB)" in res_4bit["gpu_fits"]
        assert res_4bit["gpu_fits"]["T4 / RTX 4060 Ti (16GB)"] is True

    @pytest.mark.parametrize("bad_kwargs,match", [
        ({"model": "  "}, "non-empty"),
        ({"model": "Qwen/Qwen2.5-7B", "seq_len": 0}, "seq_len must be positive"),
        ({"model": "Qwen/Qwen2.5-7B", "batch_size": -1}, "batch_size must be positive"),
        ({"model": "Qwen/Qwen2.5-7B", "lora_r": 0}, "lora_r must be positive"),
        ({"model": "Qwen/Qwen2.5-7B", "quant": "2bit"}, "Unsupported quant"),
        ({"model": "Qwen/Qwen2.5-7B", "peft": "ia3"}, "Unsupported peft"),
    ])
    def test_estimate_vram_rejects_invalid_inputs(self, bad_kwargs, match):
        with pytest.raises(ValueError, match=match):
            estimate_vram_tool(**bad_kwargs)


class TestConfigGenerationAndValidation:
    def test_generate_and_validate_config_roundtrip(self, sample_alpaca_dataset):
        gen = generate_config_tool(
            data_path=sample_alpaca_dataset.name,
            model="Qwen/Qwen2.5-1.5B-Instruct",
            goal="chat",
            output="soup.yaml",
            vram_gb=16.0,
        )
        assert gen["status"] == "ok"
        assert gen["dry_run"] is False
        assert gen["output_file"] == "soup.yaml"
        assert Path("soup.yaml").exists()

        val = validate_config_tool("soup.yaml")
        assert val["status"] == "valid"
        assert val["base"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert val["task"] == "sft"
        assert val["train_data"] == sample_alpaca_dataset.name

    def test_generate_config_dry_run_writes_nothing(self, sample_alpaca_dataset):
        gen = generate_config_tool(
            data_path=sample_alpaca_dataset.name,
            model="Qwen/Qwen2.5-3B-Instruct",
            goal="code",
            output="should_not_exist.yaml",
            dry_run=True,
        )
        assert gen["status"] == "ok"
        assert gen["dry_run"] is True
        assert gen["output_file"] is None
        assert "base:" in gen["yaml_preview"]
        assert not Path("should_not_exist.yaml").exists()

    def test_validate_config_reports_invalid_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bad_cfg = tmp_path / "bad_soup.yaml"
        bad_cfg.write_text("base: ''\ntraining:\n  epochs: -5\n", encoding="utf-8")
        res = validate_config_tool("bad_soup.yaml")
        assert res["status"] == "invalid"
        assert len(res["errors"]) >= 1

    def test_validate_config_missing_and_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            validate_config_tool("missing.yaml")
        with pytest.raises(ValueError, match="Path traversal rejected"):
            validate_config_tool("../outside.yaml")


class TestDiffDatasetsTool:
    def test_diff_datasets_detects_exact_leakage(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        train_rows = [
            {"instruction": f"Q{i}", "input": "", "output": f"A{i}"}
            for i in range(1, 5)
        ]
        # Share row Q2 and Q3 with eval split
        eval_rows = [
            {"instruction": "Q2", "input": "", "output": "A2"},
            {"instruction": "Q3", "input": "", "output": "A3"},
            {"instruction": "Q99", "input": "", "output": "A99"},
        ]
        (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train_rows) + "\n", encoding="utf-8")
        (tmp_path / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n", encoding="utf-8")

        res = diff_datasets_tool("train.jsonl", "eval.jsonl")
        assert res["status"] == "ok"
        assert res["rows_a"] == 4
        assert res["rows_b"] == 3
        assert res["overlap_count"] == 2
        assert res["only_in_a"] == 2
        assert res["only_in_b"] == 1
        assert res["leakage_detected"] is True
        assert res["overlap_pct_of_b"] == 66.67

    def test_diff_datasets_disjoint_no_leakage(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.jsonl").write_text(json.dumps({"instruction": "A", "input": "", "output": "1"}) + "\n", encoding="utf-8")
        (tmp_path / "b.jsonl").write_text(json.dumps({"instruction": "B", "input": "", "output": "2"}) + "\n", encoding="utf-8")

        res = diff_datasets_tool("a.jsonl", "b.jsonl")
        assert res["overlap_count"] == 0
        assert res["leakage_detected"] is False


class TestWorkspaceResources:
    def test_list_formats_resource(self):
        fmts = list_formats_resource()
        assert "alpaca" in fmts["valid_formats"]
        assert "sharegpt" in fmts["convertible_formats"]
        assert "chatml" in fmts["schemas"]

    def test_workspace_discovery_resources(self, sample_alpaca_dataset):
        (sample_alpaca_dataset.parent / "soup.yaml").write_text("base: test\n", encoding="utf-8")
        ds = list_workspace_datasets_resource()
        cfgs = list_workspace_configs_resource()
        assert any(d["path"] == sample_alpaca_dataset.name for d in ds["datasets"])
        assert any(c["path"] == "soup.yaml" for c in cfgs["configs"])
