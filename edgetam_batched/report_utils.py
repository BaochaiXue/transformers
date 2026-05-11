"""Report writers for JSON and small markdown summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_markdown(path: str | Path, text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text.rstrip() + "\n", encoding="utf-8")


def _format_cell(cell: Any) -> str:
    if cell is None:
        return ""
    text = str(cell)
    return text.replace("|", "\\|").replace("\n", "<br>")
