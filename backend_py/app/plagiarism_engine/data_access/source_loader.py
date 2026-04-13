from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_local_sources() -> list[dict[str, Any]]:
    """Load optional local sources from data_access/sample_sources.jsonl."""

    fp = Path(__file__).parent / "sample_sources.jsonl"
    if not fp.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and (obj.get("text") or obj.get("abstract")):
                out.append(obj)
        except Exception:
            continue
    return out
