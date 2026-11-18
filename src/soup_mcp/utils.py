"""Path containment, atomic file writes, and Soup environment resolution."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def resolve_soup_import() -> None:
    """Ensure soup_cli can be imported by probing standard paths if not on sys.path."""
    try:
        import soup_cli  # noqa: F401
        return
    except ImportError:
        pass

    candidates = [
        Path.cwd() / "src",
        Path.cwd().parent / "Soup" / "src",
        Path("C:/Users/dell/.gemini/antigravity/scratch/Soup/src"),
    ]
    for candidate in candidates:
        if (candidate / "soup_cli").is_dir():
            resolved = str(candidate.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            return


def ensure_under_cwd(path_str: str | Path) -> Path:
    """Validate that path resides under current working directory and does not escape via symlink."""
    cwd = Path.cwd().resolve()
    target = Path(path_str)

    # If relative, resolve against cwd
    if not target.is_absolute():
        resolved = (cwd / target).resolve()
    else:
        resolved = target.resolve()

    try:
        resolved.relative_to(cwd)
    except ValueError as exc:
        raise ValueError(
            f"Path traversal rejected: '{path_str}' is outside current working directory."
        ) from exc

    return resolved


def atomic_write_text(content: str, target_path: Path) -> None:
    """Atomically write text content to target_path using a temporary file in the same directory."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target_path.parent,
        delete=False,
        suffix=".tmp",
    )
    temp_path = Path(temp_file.name)
    try:
        temp_file.write(content)
        temp_file.flush()
        temp_file.close()
        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
