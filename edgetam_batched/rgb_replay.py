"""RGB replay manifest helpers.

Real replay capture remains in the QQTT repo.  This module only validates a
manifest path so correctness scripts can fail clearly when replay is missing.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_manifest(replay_root: str | Path) -> dict:
    root = Path(replay_root)
    manifest = root / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"missing RGB replay manifest: {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))
