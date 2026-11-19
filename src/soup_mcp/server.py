"""Model Context Protocol (MCP) server for Soup dataset intelligence."""

from __future__ import annotations

import sys
from typing import Any

from soup_mcp.tools import (
    clean_dataset_tool,
    convert_dataset_tool,
    dedup_dataset_tool,
    detect_dataset_tool,
    inspect_dataset_tool,
    merge_datasets_tool,
    sample_dataset_tool,
    score_dataset_tool,
    split_dataset_tool,
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

server = FastMCP(
    "soup-mcp",
    description="Soup Model Context Protocol server: comprehensive dataset inspection, cleaning, dedup, sampling, splitting, conversion, and scoring for LLM fine-tuning.",
)


@server.tool(
    name="soup_inspect_dataset",
    description="Inspect a local dataset file: check format validity, column schemas, token length distribution, duplicates, and issues.",
)
def inspect_tool(path: str) -> dict[str, Any]:
    """Inspect dataset file under working directory."""
    return inspect_dataset_tool(path)


@server.tool(
    name="soup_detect_format",
    description="Detect dataset schema (alpaca, sharegpt, chatml, etc.) and list column structure.",
)
def detect_tool(path: str) -> dict[str, Any]:
    """Detect dataset format."""
    return detect_dataset_tool(path)


@server.tool(
    name="soup_clean_dataset",
    description="Clean and repair a fine-tuning dataset: strip control characters, AI preambles ('As an AI...'), and repair code/JSON formatting.",
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
)
def score_tool(path: str) -> dict[str, Any]:
    """Compute data quality scorecard."""
    return score_dataset_tool(path)


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
        f"Confirm that all output files reside safely under the current working directory."
    )


def main() -> None:
    """Run the MCP server over stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
