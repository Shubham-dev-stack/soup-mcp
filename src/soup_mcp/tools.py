"""Core dataset tools wrapping Soup's engine with containment and atomic writes."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

import yaml

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


def dedup_dataset_tool(
    path: str,
    output: str | None = None,
    field: str | None = None,
    threshold: float = 1.0,
) -> dict[str, Any]:
    """Remove exact or near-duplicate rows from a dataset."""
    file_path = ensure_under_cwd(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows = load_raw_data(file_path)
    if not rows:
        raise ValueError("Dataset is empty.")

    if field and not any(isinstance(r, dict) and field in r for r in rows):
        raise ValueError(f"Field '{field}' is absent from all rows in the dataset.")

    total_scanned = len(rows)
    seen: set[str] = set()
    deduped_rows: list[dict[str, Any]] = []

    for r in rows:
        if field and isinstance(r, dict):
            key = str(r.get(field, ""))
        else:
            key = json.dumps(r, sort_keys=True, ensure_ascii=False)

        if key not in seen:
            seen.add(key)
            deduped_rows.append(r)

    duplicates_dropped = total_scanned - len(deduped_rows)

    out_path = ensure_under_cwd(output) if output else file_path.with_name(f"{file_path.stem}_deduped{file_path.suffix}")
    if out_path.resolve() == file_path.resolve():
        raise ValueError("Output path cannot match input path (in-place modification not allowed).")

    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in deduped_rows) + ("\n" if deduped_rows else "")
    atomic_write_text(content, out_path)

    return {
        "status": "ok",
        "input_file": str(file_path.name),
        "output_file": str(out_path.name),
        "total_scanned": total_scanned,
        "unique_rows": len(deduped_rows),
        "duplicates_dropped": duplicates_dropped,
    }


def sample_dataset_tool(
    path: str,
    output: str | None = None,
    n: int | None = None,
    pct: float | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Sample a deterministic subset of a dataset using random sampling."""
    file_path = ensure_under_cwd(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if n is None and pct is None:
        raise ValueError("Specify either n (count) or pct (percentage).")

    if n is not None and n <= 0:
        raise ValueError("n must be a positive integer.")

    if pct is not None and not (0.0 < pct <= 100.0):
        raise ValueError("pct must be a number in (0, 100].")

    rows = load_raw_data(file_path)
    total = len(rows)
    if total == 0:
        raise ValueError("Dataset is empty.")

    if n is not None:
        sample_count = min(n, total)
    else:
        sample_count = max(1, int(round(total * (pct / 100.0))))

    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(total), sample_count))
    sampled_rows = [rows[i] for i in sampled_indices]

    out_path = ensure_under_cwd(output) if output else file_path.with_name(f"{file_path.stem}_sampled{file_path.suffix}")
    if out_path.resolve() == file_path.resolve():
        raise ValueError("Output path cannot match input path (in-place modification not allowed).")

    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in sampled_rows) + ("\n" if sampled_rows else "")
    atomic_write_text(content, out_path)

    return {
        "status": "ok",
        "input_file": str(file_path.name),
        "output_file": str(out_path.name),
        "total_rows": total,
        "sample_count": len(sampled_rows),
        "seed": seed,
    }


def merge_datasets_tool(
    files: list[str],
    output: str = "merged.jsonl",
    shuffle: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Merge multiple JSONL dataset files into a single unified file."""
    if not files:
        raise ValueError("files list must not be empty.")

    resolved_inputs = []
    for f in files:
        p = ensure_under_cwd(f)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {f}")
        resolved_inputs.append(p)

    out_path = ensure_under_cwd(output)
    for inp in resolved_inputs:
        if out_path.resolve() == inp.resolve():
            raise ValueError(f"Output path cannot match input file: {inp.name}")

    all_rows = []
    file_counts = {}
    for p in resolved_inputs:
        rows = load_raw_data(p)
        file_counts[p.name] = len(rows)
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError("No data loaded from any input file.")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_rows)

    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + ("\n" if all_rows else "")
    atomic_write_text(content, out_path)

    return {
        "status": "ok",
        "output_file": str(out_path.name),
        "total_rows": len(all_rows),
        "input_files": file_counts,
        "shuffled": shuffle,
    }


_SUPPORTED_QUANTS = ("none", "4bit", "8bit", "fp8", "gptq", "awq")
_SUPPORTED_PEFT = ("lora", "none")
_GPU_TIERS_GB = {
    "T4 / RTX 4060 Ti (16GB)": 16.0,
    "RTX 3090 / 4090 / L4 (24GB)": 24.0,
    "A100 (40GB)": 40.0,
    "A100 / H100 (80GB)": 80.0,
}


def estimate_vram_tool(
    model: str,
    seq_len: int = 2048,
    batch_size: int = 1,
    quant: str = "4bit",
    peft: str = "lora",
    lora_r: int = 64,
    optimizer: str = "adamw_bnb_8bit",
    gradient_checkpointing: bool = True,
    allow_network: bool = False,
) -> dict[str, Any]:
    """Estimate peak training VRAM and GPU tier compatibility using Soup's VRAM model."""
    if not model or not model.strip():
        raise ValueError("model must be a non-empty model name or path.")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if lora_r <= 0:
        raise ValueError(f"lora_r must be positive, got {lora_r}")
    if quant not in _SUPPORTED_QUANTS:
        raise ValueError(f"Unsupported quant '{quant}'. Supported: {', '.join(_SUPPORTED_QUANTS)}")
    if peft not in _SUPPORTED_PEFT:
        raise ValueError(f"Unsupported peft '{peft}'. Supported: {', '.join(_SUPPORTED_PEFT)}")

    from soup_cli.utils.model_shape import read_model_shape
    from soup_cli.utils.vram_estimate import (
        GB,
        VRAM_SAFETY_MARGIN,
        RunSpec,
        estimate_vram,
    )

    shape = read_model_shape(model.strip(), allow_network=allow_network)
    run = RunSpec(
        seq_len=seq_len,
        batch_size=batch_size,
        quant=quant,
        peft=peft,
        lora_r=lora_r,
        optimizer=optimizer,
        gradient_checkpointing=gradient_checkpointing,
    )
    est = estimate_vram(shape, run)
    total_gb = round(est.total_gb, 3)
    recommended_gb = round(est.total_gb * (1.0 + VRAM_SAFETY_MARGIN), 3)
    breakdown_gb = {
        term: round(val / GB, 3)
        for term, val in est.terms.items()
        if val > 0
    }
    gpu_fits = {tier: recommended_gb <= cap for tier, cap in _GPU_TIERS_GB.items()}

    return {
        "status": "ok",
        "model": model.strip(),
        "params_b": round(shape.params / 1e9, 3),
        "shape_source": shape.shape_source,
        "quant": quant,
        "peft": peft,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "total_gb": total_gb,
        "recommended_vram_gb": recommended_gb,
        "breakdown_gb": breakdown_gb,
        "gpu_fits": gpu_fits,
        "notes": list(est.notes),
    }


def validate_config_tool(path: str) -> dict[str, Any]:
    """Validate a soup.yaml training configuration against Soup's Pydantic schema."""
    file_path = ensure_under_cwd(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    from soup_cli.config.loader import load_config

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            cfg = load_config(file_path)
    except (Exception, SystemExit) as exc:
        output_text = buf.getvalue().strip()
        lines = [line.strip() for line in output_text.splitlines() if line.strip()]
        if not lines:
            lines = [str(exc)]
        return {
            "status": "invalid",
            "file": file_path.name,
            "errors": lines,
        }

    t = cfg.training
    return {
        "status": "valid",
        "file": file_path.name,
        "base": cfg.base,
        "task": cfg.task,
        "backend": cfg.backend,
        "quantization": t.quantization,
        "lora_r": t.lora.r,
        "epochs": t.epochs,
        "lr": t.lr,
        "batch_size": t.batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "max_length": cfg.data.max_length,
        "train_data": cfg.data.train,
    }


def generate_config_tool(
    data_path: str,
    model: str,
    goal: str = "chat",
    output: str = "soup.yaml",
    vram_gb: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate a tailored soup.yaml training config via Soup Autopilot."""
    in_path = ensure_under_cwd(data_path)
    if not in_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    if not model or not model.strip():
        raise ValueError("model must be a non-empty model name or path.")
    if vram_gb is not None and vram_gb <= 0:
        raise ValueError(f"vram_gb must be positive, got {vram_gb}")

    from soup_cli.autopilot import build_soup_config

    cfg = build_soup_config(
        model=model.strip(),
        data_path=str(in_path),
        goal=goal,
        vram_gb=vram_gb,
    )
    # Keep data.train relative to CWD for portable configs
    cwd = Path.cwd().resolve()
    try:
        cfg.data.train = str(in_path.resolve().relative_to(cwd)).replace("\\", "/")
    except ValueError:
        pass

    data = cfg.model_dump(mode="json", exclude_none=True)
    yaml_text = yaml.safe_dump(data, sort_keys=False)

    out_file_str: str | None = None
    if not dry_run:
        out_path = ensure_under_cwd(output)
        if out_path.resolve() == in_path.resolve():
            raise ValueError("Output path cannot match input data path.")
        atomic_write_text(yaml_text, out_path)
        out_file_str = out_path.name

    return {
        "status": "ok",
        "dry_run": dry_run,
        "output_file": out_file_str,
        "task": cfg.task,
        "base": cfg.base,
        "quantization": cfg.training.quantization,
        "yaml_preview": yaml_text,
    }


def _row_digest(row: dict[str, Any]) -> str:
    canonical = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def diff_datasets_tool(path_a: str, path_b: str) -> dict[str, Any]:
    """Compare two datasets for exact row overlap (train/val leakage) and format parity."""
    file_a = ensure_under_cwd(path_a)
    file_b = ensure_under_cwd(path_b)
    if not file_a.exists():
        raise FileNotFoundError(f"File not found: {path_a}")
    if not file_b.exists():
        raise FileNotFoundError(f"File not found: {path_b}")

    rows_a = load_raw_data(file_a)
    rows_b = load_raw_data(file_b)

    def _safe_fmt(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "empty"
        try:
            return detect_format(rows)
        except Exception:
            return "custom"

    digests_a = {_row_digest(r) for r in rows_a}
    digests_b = {_row_digest(r) for r in rows_b}
    overlap = digests_a & digests_b

    return {
        "status": "ok",
        "file_a": file_a.name,
        "file_b": file_b.name,
        "format_a": _safe_fmt(rows_a),
        "format_b": _safe_fmt(rows_b),
        "rows_a": len(rows_a),
        "rows_b": len(rows_b),
        "unique_a": len(digests_a),
        "unique_b": len(digests_b),
        "overlap_count": len(overlap),
        "overlap_pct_of_b": round((len(overlap) / len(digests_b)) * 100.0, 2) if digests_b else 0.0,
        "only_in_a": len(digests_a - digests_b),
        "only_in_b": len(digests_b - digests_a),
        "leakage_detected": len(overlap) > 0,
    }


_IGNORED_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


def _scan_workspace_files(extensions: tuple[str, ...], max_depth: int = 3) -> list[dict[str, Any]]:
    cwd = Path.cwd().resolve()
    found: list[dict[str, Any]] = []
    for item in cwd.rglob("*"):
        if any(part in _IGNORED_DIRS for part in item.parts):
            continue
        if not item.is_file() or item.suffix.lower() not in extensions:
            continue
        try:
            rel = item.resolve().relative_to(cwd)
        except ValueError:
            continue
        if len(rel.parts) > max_depth:
            continue
        found.append({
            "path": str(rel).replace("\\", "/"),
            "size_bytes": item.stat().st_size,
        })
    found.sort(key=lambda x: x["path"])
    return found


def list_formats_resource() -> dict[str, Any]:
    """Return supported Soup dataset formats and convertible targets."""
    return {
        "valid_formats": sorted(VALID_FORMATS),
        "convertible_formats": sorted(CONVERTIBLE_FORMATS),
        "schemas": {
            "alpaca": ["instruction", "input (optional)", "output"],
            "sharegpt": ["conversations: [{from: human|gpt|system, value: str}]"],
            "chatml": ["messages: [{role: system|user|assistant, content: str}]"],
            "dpo": ["prompt", "chosen", "rejected"],
        },
    }


def list_workspace_datasets_resource() -> dict[str, Any]:
    """Discover dataset files under the current working directory."""
    return {
        "cwd": Path.cwd().name,
        "datasets": _scan_workspace_files((".jsonl", ".json", ".parquet", ".csv")),
    }


def list_workspace_configs_resource() -> dict[str, Any]:
    """Discover YAML configuration files under the current working directory."""
    return {
        "cwd": Path.cwd().name,
        "configs": _scan_workspace_files((".yaml", ".yml")),
    }
