"""Model Context Protocol (MCP) server for Soup dataset & fine-tuning intelligence."""

from __future__ import annotations

import argparse
import json
from typing import Any

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

# Compatibility across MCP 1.x (FastMCP) and MCP 2.x (MCPServer)
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required. Install with: pip install 'mcp>=1.0.0'"
        ) from exc

from mcp.types import ToolAnnotations

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
_SAFE_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)

server = FastMCP(
    "soup-mcp",
    description=(
        "Soup Model Context Protocol server: dataset inspection, cleaning, dedup, "
        "leakage diffing, VRAM estimation, Autopilot config generation, and schema validation."
    ),
)


@server.tool(
    name="soup_inspect_dataset",
    description="Inspect a local dataset file: check format validity, column schemas, token length distribution, duplicates, and issues.",
    annotations=_READ_ONLY,
)
def inspect_tool(path: str) -> dict[str, Any]:
    """Inspect dataset file under working directory."""
    return inspect_dataset_tool(path)


@server.tool(
    name="soup_detect_format",
    description="Detect dataset schema (alpaca, sharegpt, chatml, etc.) and list column structure.",
    annotations=_READ_ONLY,
)
def detect_tool(path: str) -> dict[str, Any]:
    """Detect dataset format."""
    return detect_dataset_tool(path)


@server.tool(
    name="soup_clean_dataset",
    description="Clean and repair a fine-tuning dataset: strip control characters, AI preambles ('As an AI...'), and repair code/JSON formatting.",
    annotations=_SAFE_WRITE,
)
def clean_tool(
    path: str,
    output: str | None = None,
    min_tokens: int = 0,
    prune_echo: bool = True,
    strip_boilerplate: bool = True,
    repair_code: bool = True,
    repair_json: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Clean and sanitize fine-tuning dataset rows."""
    return clean_dataset_tool(
        path=path,
        output=output,
        min_tokens=min_tokens,
        prune_echo=prune_echo,
        strip_boilerplate=strip_boilerplate,
        repair_code=repair_code,
        repair_json=repair_json,
        dry_run=dry_run,
    )


@server.tool(
    name="soup_dedup_dataset",
    description="Remove exact or field-level duplicate rows from a dataset while preserving order.",
    annotations=_SAFE_WRITE,
)
def dedup_tool(
    path: str,
    output: str | None = None,
    field: str | None = None,
    threshold: float = 1.0,
) -> dict[str, Any]:
    """Deduplicate dataset rows under working directory."""
    return dedup_dataset_tool(
        path=path,
        output=output,
        field=field,
        threshold=threshold,
    )


@server.tool(
    name="soup_sample_dataset",
    description="Extract a deterministic random subset of a dataset by count (n) or percentage (pct).",
    annotations=_SAFE_WRITE,
)
def sample_tool(
    path: str,
    output: str | None = None,
    n: int | None = None,
    pct: float | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Sample dataset rows under working directory."""
    return sample_dataset_tool(
        path=path,
        output=output,
        n=n,
        pct=pct,
        seed=seed,
    )


@server.tool(
    name="soup_split_dataset",
    description="Split a dataset into deterministic train/val/test partitions with a reproducible seed.",
    annotations=_SAFE_WRITE,
)
def split_tool(
    path: str,
    val: int | None = None,
    test: int | None = None,
    absolute: bool = False,
    seed: int = 42,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Split dataset into train, val, and test partitions."""
    return split_dataset_tool(
        path=path,
        val=val,
        test=test,
        absolute=absolute,
        seed=seed,
        output_dir=output_dir,
    )


@server.tool(
    name="soup_convert_format",
    description="Convert dataset between formats (alpaca, sharegpt, chatml).",
    annotations=_SAFE_WRITE,
)
def convert_tool(
    path: str,
    to_format: str,
    output: str | None = None,
) -> dict[str, Any]:
    """Convert dataset to a target training format."""
    return convert_dataset_tool(
        path=path,
        to_format=to_format,
        output=output,
    )


@server.tool(
    name="soup_merge_datasets",
    description="Merge multiple JSONL dataset files into a single unified file.",
    annotations=_SAFE_WRITE,
)
def merge_tool(
    files: list[str],
    output: str = "merged.jsonl",
    shuffle: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Merge multiple dataset files under working directory."""
    return merge_datasets_tool(
        files=files,
        output=output,
        shuffle=shuffle,
        seed=seed,
    )


@server.tool(
    name="soup_score_dataset",
    description="Score a dataset for PII, toxic keywords, language distribution, and educational value.",
    annotations=_READ_ONLY,
)
def score_tool(path: str) -> dict[str, Any]:
    """Compute data quality scorecard."""
    return score_dataset_tool(path)


@server.tool(
    name="soup_estimate_vram",
    description="Estimate peak training VRAM (weights, adapters, optimizer, activations, overhead) and GPU tier compatibility.",
    annotations=_READ_ONLY,
)
def vram_tool(
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
    """Estimate peak VRAM for a fine-tuning run."""
    return estimate_vram_tool(
        model=model,
        seq_len=seq_len,
        batch_size=batch_size,
        quant=quant,
        peft=peft,
        lora_r=lora_r,
        optimizer=optimizer,
        gradient_checkpointing=gradient_checkpointing,
        allow_network=allow_network,
    )


@server.tool(
    name="soup_validate_config",
    description="Validate a soup.yaml training config against Soup's Pydantic schema and report errors or resolved hyperparameters.",
    annotations=_READ_ONLY,
)
def validate_config_mcp(path: str = "soup.yaml") -> dict[str, Any]:
    """Validate soup.yaml configuration file."""
    return validate_config_tool(path)


@server.tool(
    name="soup_generate_config",
    description="Generate an optimized soup.yaml training config via Soup Autopilot (goal: chat, code, classification, tool-calling, reasoning, alignment, domain-adapt).",
    annotations=_SAFE_WRITE,
)
def generate_config_mcp(
    data_path: str,
    model: str,
    goal: str = "chat",
    output: str = "soup.yaml",
    vram_gb: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate soup.yaml via Soup Autopilot."""
    return generate_config_tool(
        data_path=data_path,
        model=model,
        goal=goal,
        output=output,
        vram_gb=vram_gb,
        dry_run=dry_run,
    )


@server.tool(
    name="soup_diff_datasets",
    description="Compare two datasets for exact row overlap (train/val leakage detection) and format parity.",
    annotations=_READ_ONLY,
)
def diff_datasets_mcp(path_a: str, path_b: str) -> dict[str, Any]:
    """Diff two dataset files under working directory."""
    return diff_datasets_tool(path_a=path_a, path_b=path_b)


# ---------------------------------------------------------------------------
# MCP Resources (live workspace & schema context)
# ---------------------------------------------------------------------------


@server.resource(
    "soup://formats",
    name="soup_formats",
    title="Supported Soup Dataset Formats",
    description="Supported input/output dataset formats and schema field conventions.",
    mime_type="application/json",
)
def resource_formats() -> str:
    """Expose supported Soup dataset formats."""
    return json.dumps(list_formats_resource(), indent=2)


@server.resource(
    "soup://workspace/datasets",
    name="workspace_datasets",
    title="Workspace Dataset Files",
    description="Discovers dataset files (.jsonl, .json, .parquet, .csv) under the working directory.",
    mime_type="application/json",
)
def resource_workspace_datasets() -> str:
    """Expose dataset files under CWD."""
    return json.dumps(list_workspace_datasets_resource(), indent=2)


@server.resource(
    "soup://workspace/configs",
    name="workspace_configs",
    title="Workspace YAML Configs",
    description="Discovers YAML configuration files under the working directory.",
    mime_type="application/json",
)
def resource_workspace_configs() -> str:
    """Expose YAML configs under CWD."""
    return json.dumps(list_workspace_configs_resource(), indent=2)


# ---------------------------------------------------------------------------
# MCP Prompts (workflow recipes)
# ---------------------------------------------------------------------------


@server.prompt(
    name="audit_dataset",
    description="Step-by-step instructions for performing a quality, safety, and health audit on a fine-tuning dataset.",
)
def prompt_audit(dataset_path: str) -> str:
    """Prompt template for auditing a fine-tuning dataset."""
    return (
        f"Please perform a rigorous data audit on '{dataset_path}' using the Soup MCP tools:\n"
        f"1. Run `soup_inspect_dataset` to check schema columns, total rows, and average token lengths.\n"
        f"2. Run `soup_detect_format` to verify whether the format matches Alpaca, ShareGPT, or ChatML.\n"
        f"3. Run `soup_score_dataset` to detect potential PII leaks (emails, phones), toxicity flags, and language mix.\n"
        f"4. Propose actionable recommendations on whether cleaning or deduplication is required before training."
    )


@server.prompt(
    name="prepare_dataset",
    description="End-to-end recipe prompt for preparing a raw dataset into clean, split train/val files.",
)
def prompt_prepare(dataset_path: str, target_format: str = "sharegpt") -> str:
    """Prompt template for end-to-end dataset preparation."""
    return (
        f"Please prepare the raw dataset '{dataset_path}' for local LLM fine-tuning:\n"
        f"1. Inspect schema with `soup_inspect_dataset`.\n"
        f"2. Clean boilerplate, AI preambles, and broken code blocks with `soup_clean_dataset`.\n"
        f"3. Remove duplicates with `soup_dedup_dataset`.\n"
        f"4. Convert to the target format '{target_format}' using `soup_convert_format` if necessary.\n"
        f"5. Split the resulting data into 80% train / 10% val / 10% test using `soup_split_dataset`.\n"
        f"6. Verify zero train/val leakage with `soup_diff_datasets`.\n"
        f"Confirm that all output files reside safely under the current working directory."
    )


@server.prompt(
    name="plan_finetune",
    description="Plan a VRAM-safe fine-tuning run and generate a validated soup.yaml config.",
)
def prompt_plan_finetune(dataset_path: str, model: str, vram_gb: str = "16") -> str:
    """Prompt template for planning a fine-tuning run with VRAM check and Autopilot config."""
    return (
        f"Plan a fine-tuning run for '{model}' on '{dataset_path}' within a {vram_gb}GB VRAM budget:\n"
        f"1. Run `soup_inspect_dataset` on '{dataset_path}' to verify format and token lengths.\n"
        f"2. Run `soup_estimate_vram` for '{model}' (test 4bit QLoRA with gradient_checkpointing=True) to confirm GPU fit.\n"
        f"3. Run `soup_generate_config` with vram_gb={vram_gb} to produce 'soup.yaml'.\n"
        f"4. Run `soup_validate_config` on 'soup.yaml' and summarize the final hyperparameters."
    )


@server.prompt(
    name="check_data_leakage",
    description="Verify zero overlap between train and validation/test splits.",
)
def prompt_check_leakage(train_path: str, eval_path: str) -> str:
    """Prompt template for checking train/eval data leakage."""
    return (
        f"Check for train/eval contamination between '{train_path}' and '{eval_path}':\n"
        f"1. Run `soup_diff_datasets` with path_a='{train_path}' and path_b='{eval_path}'.\n"
        f"2. If `leakage_detected` is true, report the exact `overlap_count` and recommend deduplicating before splitting."
    )


def main(argv: list[str] | None = None) -> None:
    """Run the Soup MCP server over stdio, sse, or streamable-http transport."""
    parser = argparse.ArgumentParser(description="Soup MCP Server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="MCP transport mode (default: stdio)",
    )
    args = parser.parse_args(argv)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
