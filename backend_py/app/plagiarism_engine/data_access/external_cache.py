from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config import settings


def _cache_dir() -> Path:
    base = Path(settings.cache_dir)
    p = base / "external"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _key(provider: str, query: str) -> str:
    import hashlib

    raw = f"{provider}\n{query}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def load(provider: str, query: str) -> list[dict[str, Any]] | None:
    ttl = int(getattr(settings, "external_cache_ttl_seconds", 0) or 0)
    if ttl <= 0:
        return None

    fp = _cache_dir() / f"{_key(provider, query)}.json"
    try:
        st = fp.stat()
        if (time.time() - st.st_mtime) > ttl:
            return None
        obj = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            out: list[dict[str, Any]] = []
            for it in obj:
                if isinstance(it, dict):
                    out.append(it)
            return out
    except Exception:
        return None
    return None


def save(provider: str, query: str, items: list[dict[str, Any]]) -> None:
    ttl = int(getattr(settings, "external_cache_ttl_seconds", 0) or 0)
    if ttl <= 0:
        return

    fp = _cache_dir() / f"{_key(provider, query)}.json"
    try:
        fp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return
