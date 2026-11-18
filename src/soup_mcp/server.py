"""Model Context Protocol (MCP) server for Soup dataset intelligence."""

from __future__ import annotations

import sys
from typing import Any

from soup_mcp.tools import (
    clean_dataset_tool,
    convert_dataset_tool,
    detect_dataset_tool,
    inspect_dataset_tool,
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
    description="Soup Model Context Protocol server for dataset inspection, cleaning, splitting, conversion, and scoring.",
)


@server.tool(
    name="soup_inspect_dataset",
    description="Inspect a local dataset file: check format validity, column schemas, statistics, and sample rows.",
)
def inspect_tool(path: str) -> dict[str, Any]:
    """Inspect dataset file under working directory."""
    return inspect_dataset_tool(path)


@server.tool(
    name="soup_clean_dataset",
    description="Clean and repair a fine-tuning dataset: strip control characters, AI preambles, and repair code/JSON formatting.",
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
    name="soup_score_dataset",
    description="Score a dataset for PII, toxic keywords, language distribution, and educational value.",
)
def score_tool(path: str) -> dict[str, Any]:
    """Compute data quality scorecard."""
    return score_dataset_tool(path)


@server.tool(
    name="soup_detect_format",
    description="Detect dataset schema (alpaca, sharegpt, chatml, etc.) and list column structure.",
)
def detect_tool(path: str) -> dict[str, Any]:
    """Detect dataset format."""
    return detect_dataset_tool(path)


def main() -> None:
    """Run the MCP server over stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
