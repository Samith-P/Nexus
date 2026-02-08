from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def qdrant_enabled() -> bool:
    """Enable Qdrant by setting QDRANT_URL or QDRANT_HOST.

    Optional: set USE_QDRANT=false to force-disable.
    """

    if not _env_bool("USE_QDRANT", True):
        return False
    return bool(os.getenv("QDRANT_URL") or os.getenv("QDRANT_HOST"))


def _collection_name(collection_name: Optional[str] = None) -> str:
    return collection_name or os.getenv("QDRANT_COLLECTION", "topic_kb")


def _vector_size() -> int:
    # hashing fallback uses 256 dims; IndicBERT can be different, but we keep a fixed size
    # by letting Qdrant accept vectors of this size. If SentenceTransformer outputs a
    # different dim, we fall back to in-memory search.
    return int(os.getenv("QDRANT_VECTOR_SIZE", "256"))


@lru_cache(maxsize=1)
def _client():
    """Create Qdrant client lazily.

    Returns None if qdrant-client isn't installed or configuration is missing.
    """

    if not qdrant_enabled():
        return None
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except Exception:
        return None

    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))

    # Default to the docker-compose service name when QDRANT_HOST isn't set
    host = os.getenv("QDRANT_HOST") or ("qdrant" if _env_bool("USE_QDRANT", True) else "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port, api_key=os.getenv("QDRANT_API_KEY"))


def ensure_collection(vector_size: int, collection_name: Optional[str] = None) -> bool:
    """Ensure the collection exists with the correct vector size."""

    cli = _client()
    if cli is None:
        return False

    try:
        from qdrant_client.models import Distance, VectorParams  # type: ignore

        name = _collection_name(collection_name)
        existing = cli.get_collections().collections
        if any(c.name == name for c in existing):
            return True

        cli.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return True
    except Exception:
        return False


def upsert_topics(
    points: List[Tuple[str, Sequence[float], Dict[str, Any]]],
    collection_name: Optional[str] = None,
) -> bool:
    """Upsert topic vectors + payload into Qdrant.

    Each item: (topic_id, vector, payload)
    """

    cli = _client()
    if cli is None:
        return False

    try:
        from qdrant_client.models import PointStruct  # type: ignore

        name = _collection_name(collection_name)
        pts = [PointStruct(id=pid, vector=list(vec), payload=payload) for pid, vec, payload in points]
        cli.upsert(collection_name=name, points=pts)
        return True
    except Exception:
        return False


def search(
    vector: Sequence[float],
    top_k: int = 50,
    collection_name: Optional[str] = None,
) -> Optional[List[Tuple[str, float, Dict[str, Any]]]]:
    """Search Qdrant and return [(id, score, payload)]."""

    cli = _client()
    if cli is None:
        return None
    try:
        name = _collection_name(collection_name)
        hits = cli.search(collection_name=name, query_vector=list(vector), limit=int(top_k))
        out: List[Tuple[str, float, Dict[str, Any]]] = []
        for h in hits:
            out.append((str(h.id), float(h.score or 0.0), dict(h.payload or {})))
        return out
    except Exception:
        return None


def bootstrap_from_kb(
    topics: Sequence[Any],
    embeddings: Dict[str, Sequence[float]],
    collection_name: Optional[str] = None,
) -> bool:
    """Create collection and upsert the full KB.

    This is safe to call multiple times; it will upsert points.
    """

    cli = _client()
    if cli is None:
        return False

    vec_size = _vector_size()
    if not ensure_collection(vec_size, collection_name=collection_name):
        return False

    points: List[Tuple[str, Sequence[float], Dict[str, Any]]] = []
    for t in topics:
        tid = getattr(t, "topic_id", None) or (t.get("topic_id") if isinstance(t, dict) else None)
        if not tid:
            continue
        vec = embeddings.get(str(tid))
        if not vec or len(vec) != vec_size:
            continue

        if hasattr(t, "__dict__"):
            payload = {k: v for k, v in t.__dict__.items() if k != "topic_id"}
        else:
            payload = dict(t)
            payload.pop("topic_id", None)

        points.append((str(tid), vec, payload))

    if not points:
        return False
    return upsert_topics(points, collection_name=collection_name)
