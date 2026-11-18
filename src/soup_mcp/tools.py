"""Core dataset tools wrapping Soup's engine with containment and atomic writes."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from soup_mcp.utils import atomic_write_text, ensure_under_cwd, resolve_soup_import

# Ensure soup_cli is resolvable
resolve_soup_import()

from soup_cli.data.formats import (  # noqa: E402
    CONVERTIBLE_FORMATS,
    VALID_FORMATS,
    detect_format,
    format_to_messages,
    messages_to_format,
)
from soup_cli.data.loader import load_raw_data  # noqa: E402
from soup_cli.data.validator import validate_and_stats  # noqa: E402
from soup_cli.utils.data_clean import clean_dataset  # noqa: E402
from soup_cli.utils.data_score import compute_scorecard  # noqa: E402


def inspect_dataset_tool(path: str) -> dict[str, Any]:
    """Inspect a dataset: check format validity, statistics, and sample rows."""
    file_path = ensure_under_cwd(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows = load_raw_data(file_path)
    if not rows:
        return {
            "status": "empty",
            "file": str(file_path.name),
            "total_rows": 0,
            "format": "unknown",
            "columns": [],
        }

    try:
        detected_fmt = detect_format(rows)
    except Exception:
        detected_fmt = "custom"

    stats = validate_and_stats(rows, expected_format=detected_fmt if detected_fmt in VALID_FORMATS else None)
    return {
        "status": "ok",
        "file": str(file_path.name),
        "format": detected_fmt,
        "total_rows": stats.get("total", len(rows)),
        "valid_rows": stats.get("valid_rows", 0),
        "columns": stats.get("columns", []),
        "avg_length": stats.get("avg_length", 0),
        "min_length": stats.get("min_length", 0),
        "max_length": stats.get("max_length", 0),
        "duplicates": stats.get("duplicates", 0),
        "issues": stats.get("issues", []),
    }


def clean_dataset_tool(
    path: str,
    output: str | None = None,
    min_tokens: int = 0,
    prune_echo: bool = True,
    strip_boilerplate: bool = True,
    repair_code: bool = True,
    repair_json: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Clean and repair fine-tuning datasets: strip control characters, boilerplate, repair code/JSON."""
    file_path = ensure_under_cwd(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if min_tokens < 0:
        raise ValueError(f"min_tokens must be non-negative, got {min_tokens}")

    rows = load_raw_data(file_path)
    if not rows:
        raise ValueError("Dataset is empty.")

    try:
        fmt = detect_format(rows)
    except Exception:
        fmt = "plaintext"

    cleaned_data, report = clean_dataset(
        rows,
        fmt,
        min_tokens=min_tokens,
        prune_echo=prune_echo,
        strip_ai_boilerplate=strip_boilerplate,
        repair_code=repair_code,
        repair_json=repair_json,
    )

    out_file_str: str | None = None
    if not dry_run:
        if output is None:
            out_path = file_path.with_name(f"{file_path.stem}_cleaned{file_path.suffix}")
        else:
            out_path = ensure_under_cwd(output)

        if out_path.resolve() == file_path.resolve():
            raise ValueError("Output path cannot match input path (in-place modification not allowed).")

        jsonl_lines = [json.dumps(r, ensure_ascii=False) for r in cleaned_data]
        content = "\n".join(jsonl_lines) + ("\n" if jsonl_lines else "")
        atomic_write_text(content, out_path)
        out_file_str = str(out_path.name)

    return {
        "status": "ok",
        "dry_run": dry_run,
        "input_file": str(file_path.name),
        "output_file": out_file_str,
        "format": fmt,
        "total_scanned": report.total_scanned,
        "total_clean": report.total_clean,
        "total_modified": report.total_modified,
        "total_dropped": report.total_dropped,
        "output_rows": len(cleaned_data),
        "rule_counts": report.rule_counts,
    }


def split_dataset_tool(
    path: str,
    val: int | None = None,
    test: int | None = None,
    absolute: bool = False,
    seed: int = 42,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Split dataset into deterministic train/val/test partitions."""
    file_path = ensure_under_cwd(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if val is None and test is None:
        raise ValueError("Specify at least one of val or test.")

    if (val is not None and val < 0) or (test is not None and test < 0):
        raise ValueError("val and test must be non-negative.")

    if (val or 0) == 0 and (test or 0) == 0:
        raise ValueError("Specify at least one positive split (val or test).")

    if not absolute and ((val or 0) + (test or 0) >= 100):
        raise ValueError("Split percentages must sum to less than 100.")

    rows = load_raw_data(file_path)
    total = len(rows)
    if total == 0:
        raise ValueError("Dataset is empty.")

    if absolute:
        val_count = val or 0
        test_count = test or 0
        if val_count + test_count >= total:
            raise ValueError(f"val ({val_count}) + test ({test_count}) must be less than total ({total}).")
    else:
        val_count = int(round(total * ((val or 0) / 100.0)))
        test_count = int(round(total * ((test or 0) / 100.0)))
        if val_count + test_count >= total:
            raise ValueError("Rounded split percentages exceed total rows.")

    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    test_indices = set(indices[:test_count])
    val_indices = set(indices[test_count : test_count + val_count])

    train_data = []
    val_data = []
    test_data = []

    for idx in range(total):
        if idx in test_indices:
            test_data.append(rows[idx])
        elif idx in val_indices:
            val_data.append(rows[idx])
        else:
            train_data.append(rows[idx])

    target_dir = ensure_under_cwd(output_dir) if output_dir else file_path.parent
    base_name = file_path.stem

    train_path = target_dir / f"{base_name}_train.jsonl"
    val_path = target_dir / f"{base_name}_val.jsonl"
    test_path = target_dir / f"{base_name}_test.jsonl"

    files_written = {}
    if train_data:
        atomic_write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train_data) + "\n", train_path)
        files_written["train"] = str(train_path.name)
    if val_data:
        atomic_write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val_data) + "\n", val_path)
        files_written["val"] = str(val_path.name)
    if test_data:
        atomic_write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in test_data) + "\n", test_path)
        files_written["test"] = str(test_path.name)

    return {
        "status": "ok",
        "total_rows": total,
        "seed": seed,
        "counts": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
        "files": files_written,
    }


def convert_dataset_tool(
    path: str,
    to_format: str,
    output: str | None = None,
) -> dict[str, Any]:
    """Convert dataset between formats (alpaca, sharegpt, chatml)."""
    file_path = ensure_under_cwd(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    target_fmt = to_format.lower().strip()
    if target_fmt not in CONVERTIBLE_FORMATS:
        raise ValueError(f"Invalid target format: '{target_fmt}'. Supported: {', '.join(CONVERTIBLE_FORMATS)}")

    rows = load_raw_data(file_path)
    if not rows:
        raise ValueError("Dataset is empty.")

    src_fmt = detect_format(rows)
    if src_fmt == target_fmt:
        return {
            "status": "noop",
            "message": f"Source and target formats are both '{target_fmt}'. Nothing to convert.",
            "total_rows": len(rows),
        }

    converted_rows = []
    for r in rows:
        msgs = format_to_messages(r, src_fmt)
        converted = messages_to_format(msgs, target_fmt)
        converted_rows.append(converted)

    out_path = ensure_under_cwd(output) if output else file_path.with_name(f"{file_path.stem}_{target_fmt}.jsonl")
    if out_path.resolve() == file_path.resolve():
        raise ValueError("Output path cannot match input path.")

    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in converted_rows) + "\n"
    atomic_write_text(content, out_path)

    return {
        "status": "ok",
        "source_format": src_fmt,
        "target_format": target_fmt,
        "total_rows": len(converted_rows),
        "output_file": str(out_path.name),
    }


def score_dataset_tool(path: str) -> dict[str, Any]:
    """Score a dataset for PII, toxic keywords, language distribution, and educational value."""
    file_path = ensure_under_cwd(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows = load_raw_data(file_path)
    if not rows:
        raise ValueError("Dataset is empty.")

    rep = compute_scorecard(rows)
    return {
        "status": "ok",
        "file": str(file_path.name),
        "total_rows": rep.total,
        "pii_flagged": rep.pii_flagged,
        "toxic_flagged": rep.toxic_flagged,
        "languages": dict(rep.languages),
        "educational_mean": rep.educational_mean,
    }


def detect_dataset_tool(path: str) -> dict[str, Any]:
    """Detect dataset format and schema columns."""
    file_path = ensure_under_cwd(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows = load_raw_data(file_path)
    if not rows:
        return {"status": "empty", "total_rows": 0, "format": "unknown", "columns": []}

    fmt = detect_format(rows)
    columns = list(rows[0].keys()) if isinstance(rows[0], dict) else []
    return {
        "status": "ok",
        "file": str(file_path.name),
        "format": fmt,
        "total_rows": len(rows),
        "columns": columns,
    }
