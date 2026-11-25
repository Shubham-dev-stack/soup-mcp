# soup-mcp: Model Context Protocol Server for Soup

A lightweight, secure, and production-ready **Model Context Protocol (MCP)** server that exposes Soup's dataset intelligence and processing engine directly to AI developer tools—including **Cursor**, **Claude Code**, **Antigravity**, **Cline**, and **Continue**.

---

## Why soup-mcp?

When AI engineers fine-tune local models in modern AI IDEs, data preparation is the single largest bottleneck:
- Inspecting schemas and field distributions
- Stripping invisible control characters, AI preambles ("As an AI language model..."), and fixing broken markdown code fences
- Performing deterministic train/val/test splits with reproducible seeds
- Removing duplicates and near-duplicate records
- Deterministic sampling for fast testing
- Merging multiple disparate data sources into a single file
- Converting between Alpaca, ShareGPT, and ChatML formats
- Scoring datasets for PII, toxic keywords, and educational value

`soup-mcp` connects these operations directly into the LLM's tool-calling loop, allowing AI assistants to prepare datasets autonomously, safely, and reproducibly.

---

## Available MCP Tools (13 Tools with `ToolAnnotations`)

Every tool includes MCP `ToolAnnotations` (`readOnlyHint`, `destructiveHint`, `idempotentHint`) so clients like Cursor, Claude Code, and Antigravity can safely auto-approve read-only operations.

| Tool | Type | Description |
| :--- | :--- | :--- |
| `soup_inspect_dataset` | Read-only | Analyzes format validity, row counts, token length distributions, empty fields, duplicates, and issues. |
| `soup_detect_format` | Read-only | Detects schema format (`alpaca`, `sharegpt`, `chatml`, `dpo`, etc.) and column structure. |
| `soup_score_dataset` | Read-only | Evaluates dataset for PII (emails/phones), toxic keywords, language distribution, and educational value. |
| `soup_diff_datasets` | Read-only | Compares two datasets for exact row overlap (train/val leakage detection) and format parity. |
| `soup_estimate_vram` | Read-only | Estimates peak training VRAM breakdown (weights, adapters, optimizer, activations, overhead) and GPU fit (T4/4090/A100/H100). |
| `soup_validate_config` | Read-only | Validates a `soup.yaml` training config against Soup's Pydantic schema and returns resolved hyperparameters or exact errors. |
| `soup_clean_dataset` | Safe Mutating | Sanitizes control characters, strips AI disclaimers, repairs unclosed markdown fences, and fixes JSON syntax. |
| `soup_dedup_dataset` | Safe Mutating | Drops exact and field-level duplicates with preserved row order. |
| `soup_sample_dataset` | Safe Mutating | Deterministic random subset sampling by count or percentage with seed. |
| `soup_split_dataset` | Safe Mutating | Deterministic train/validation/test partitioning with reproducible pseudorandom seeds. |
| `soup_convert_format` | Safe Mutating | Lossless format conversion between Alpaca, ShareGPT, and ChatML. |
| `soup_merge_datasets` | Safe Mutating | Merges multiple JSONL dataset files into a single unified file. |
| `soup_generate_config` | Safe Mutating | Generates a tailored `soup.yaml` via Soup Autopilot from dataset profile, model, goal, and optional VRAM budget. |

---

## MCP Resources (Live Workspace Context)

- **`soup://formats`**: Supported Soup dataset formats (`alpaca`, `sharegpt`, `chatml`, `dpo`) and field schemas.
- **`soup://workspace/datasets`**: Auto-discovers `.jsonl`, `.json`, `.parquet`, and `.csv` dataset files under CWD.
- **`soup://workspace/configs`**: Auto-discovers `.yaml` / `.yml` configuration files under CWD.

---

## MCP Prompts (Built-in Workflow Recipes)

`soup-mcp` includes interactive prompts that AI assistants can load:
- **`audit_dataset`**: Guides the AI to run inspection, format detection, and scoring tools to generate an exhaustive dataset health report.
- **`prepare_dataset`**: Guides the AI through the complete preparation recipe: inspection -> cleaning -> deduplication -> format conversion -> train/val splitting -> leakage check.
- **`plan_finetune`**: Plans a VRAM-safe fine-tuning run, checks GPU fit with `soup_estimate_vram`, generates `soup.yaml` via Autopilot, and validates it.
- **`check_data_leakage`**: Runs `soup_diff_datasets` across train/eval splits to guarantee 0% contamination.

---

## Security Invariants

`soup-mcp` adheres strictly to Soup's core security architecture:
1. **Working Directory Containment:** All read and write operations are strictly bound to the operator's Current Working Directory (`CWD`). Any relative traversal (`../`) or absolute path escaping CWD is refused immediately.
2. **Symlink Boundary Checks:** Symlinks pointing outside the workspace boundary are rejected.
3. **Atomic File Writes:** Output datasets are written to temporary files and atomically renamed (`os.replace`). File generation is all-or-nothing; an interrupted operation will never corrupt an existing dataset.
4. **No In-Place Overwrites:** Input datasets are never modified in place. Cleaned or converted files default to distinct filenames (e.g. `data_cleaned.jsonl`).

---

## Quickstart & Client Setup

### 1. Installation

```bash
git clone https://github.com/Shubham-dev-stack/soup-mcp.git
cd soup-mcp
pip install -e .
```

### 2. Configuration for AI Clients

#### **Cursor (`.cursor/mcp.json` or Global MCP Settings)**
```json
{
  "mcpServers": {
    "soup": {
      "command": "python",
      "args": ["-m", "soup_mcp.server"]
    }
  }
}
```

#### **Claude Desktop (`claude_desktop_config.json`)**
```json
{
  "mcpServers": {
    "soup": {
      "command": "python",
      "args": ["-m", "soup_mcp.server"]
    }
  }
}
```

#### **Antigravity / Cline / Continue (`.mcp.json`)**
```json
{
  "mcpServers": {
    "soup": {
      "command": "python",
      "args": ["-m", "soup_mcp.server"]
    }
  }
}
```

---

## Testing & Verification

The test suite enforces 100% test pass rates and multi-item positive control verification:

```bash
pytest -v
```

All 71 tests pass covering:
- Format auto-detection & inspection across multi-row datasets
- Cleaning rules (invisible control characters, zero-width spaces, AI disclaimers)
- De-duplication and preserved item order
- Sample partitioning with deterministic seeds
- Multi-dataset file merging
- Split boundary checks (percentage & absolute modes with seed reproducibility)
- 1-to-1 message preservation during format conversion
- Negative security controls (path traversal, parent symlinks, in-place overwrite prevention)
