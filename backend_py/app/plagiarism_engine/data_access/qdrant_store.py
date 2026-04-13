from __future__ import annotations

import os
import uuid
from functools import lru_cache
from typing import Any, Iterable, Optional, Sequence

from ..config import settings


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def qdrant_enabled() -> bool:
    if not bool(getattr(settings, "use_qdrant", True)):
        return False
    # Allow hard-disable consistent with topic engine.
    if not _env_bool("USE_QDRANT", True):
        return False
    return bool(os.getenv("QDRANT_URL") or os.getenv("QDRANT_HOST") or getattr(settings, "qdrant_url", None))


def _collection_name(vector_size: int, *, base: str | None = None) -> str:
    base = (
        (base or "").strip()
        or (getattr(settings, "qdrant_collection", "") or os.getenv("PLAG_QDRANT_COLLECTION") or "plagiarism_sources").strip()
    )
    if not base:
        base = "plagiarism_sources"
    # Avoid dimension-mismatch issues when users switch embedding models.
    return f"{base}_{int(vector_size)}"


@lru_cache(maxsize=1)
def _client():
    if not qdrant_enabled():
        return None
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except Exception:
        return None

    url = (getattr(settings, "qdrant_url", "") or os.getenv("QDRANT_URL") or "").strip()
    if url:
        return QdrantClient(url=url, api_key=(getattr(settings, "qdrant_api_key", None) or os.getenv("QDRANT_API_KEY")))

    host = (getattr(settings, "qdrant_host", "") or os.getenv("QDRANT_HOST") or "qdrant").strip() or "qdrant"
    port = int(getattr(settings, "qdrant_port", 6333) or os.getenv("QDRANT_PORT") or 6333)
    return QdrantClient(host=host, port=port, api_key=(getattr(settings, "qdrant_api_key", None) or os.getenv("QDRANT_API_KEY")))


def ensure_collection(vector_size: int, *, base: str | None = None) -> Optional[str]:
    """Ensure a COSINE collection exists. Returns the collection name or None."""

    cli = _client()
    if cli is None:
        return None

    try:
        from qdrant_client.models import Distance, VectorParams  # type: ignore

        name = _collection_name(vector_size, base=base)
        existing = cli.get_collections().collections
        if any(c.name == name for c in existing):
            return name

        cli.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=int(vector_size), distance=Distance.COSINE),
        )
        return name
    except Exception:
        return None


def to_point_id(raw_id: object, *, namespace: str = "plagiarism") -> str | int:
    """Convert arbitrary IDs to Qdrant-compatible IDs (int or UUID string)."""

    s = str(raw_id).strip()
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            pass

    try:
        return str(uuid.UUID(s))
    except Exception:
        pass

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{s}"))


def upsert_points(
    collection_name: str,
    points: Iterable[tuple[str | int, Sequence[float], dict[str, Any]]],
    *,
    batch_size: int = 64,
) -> bool:
    cli = _client()
    if cli is None:
        return False

    try:
        from qdrant_client.models import PointStruct  # type: ignore

        buf: list[PointStruct] = []
        for pid, vec, payload in points:
            buf.append(PointStruct(id=pid, vector=list(vec), payload=payload))
            if len(buf) >= max(1, int(batch_size)):
                cli.upsert(collection_name=collection_name, points=buf)
                buf = []
        if buf:
            cli.upsert(collection_name=collection_name, points=buf)
        return True
    except Exception:
        return False


def search(
    collection_name: str,
    query_vector: Sequence[float],
    *,
    top_k: int = 5,
    qdrant_filter: Any = None,
) -> Optional[list[tuple[str, float, dict[str, Any]]]]:
    cli = _client()
    if cli is None:
        return None

    try:
        hits = cli.search(
            collection_name=collection_name,
            query_vector=list(query_vector),
            limit=int(top_k),
            query_filter=qdrant_filter,
            with_payload=True,
        )
        out: list[tuple[str, float, dict[str, Any]]] = []
        for h in hits:
            out.append((str(h.id), float(h.score or 0.0), dict(h.payload or {})))
        return out
    except Exception:
        return None


def delete_by_report_id(collection_name: str, report_id: str) -> bool:
    """Best-effort cleanup of per-report points."""

    cli = _client()
    if cli is None:
        return False

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

        flt = Filter(must=[FieldCondition(key="report_id", match=MatchValue(value=str(report_id)))])
        cli.delete(collection_name=collection_name, points_selector=flt)
        return True
    except Exception:
        return False
