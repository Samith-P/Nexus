from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import os

from embedding import batch_cosine_similarity, embed_text
from qdrant_store import bootstrap_from_kb, qdrant_enabled, search as qdrant_search


@dataclass(frozen=True)
class Topic:
    topic_id: str
    title: str
    domain: str
    keywords: List[str]
    policy_tags: List[str]
    citations: int
    year: int

    def text_for_embedding(self) -> str:
        kws = ", ".join(self.keywords or [])
        return f"{self.title}. Keywords: {kws}. Domain: {self.domain}."


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def load_topics(path: Optional[Path] = None) -> List[Topic]:
    # User requested to not use JSON files under data/.
    # Topic candidates are produced at request-time from academic APIs + Datasets.
    _ = path
    return []


def search_topics(query: str, topics: Sequence[Topic], top_k: int = 50) -> List[Tuple[Topic, float]]:
    if not topics:
        return []

    qv = embed_text(query)

    # Prefer Qdrant if configured and vector size matches
    # (keeps the system fully functional even without Qdrant running)
    if qdrant_enabled() and len(qv) == int(os.getenv("QDRANT_VECTOR_SIZE", "256")):
        # Best-effort bootstrap: upsert KB once per process if requested
        if os.getenv("QDRANT_AUTO_INDEX", "").strip().lower() in {"1", "true", "yes"}:
            embeddings = {t.topic_id: embed_text(t.text_for_embedding()) for t in topics}
            bootstrap_from_kb(topics, embeddings)

        hits = qdrant_search(qv, top_k=top_k)
        if hits:
            by_id = {t.topic_id: t for t in topics}
            pairs: List[Tuple[Topic, float]] = []
            for tid, score, _payload in hits:
                t = by_id.get(tid)
                if t is not None:
                    pairs.append((t, float(score)))
            if pairs:
                return pairs

    # Fallback: in-memory cosine similarity
    vectors = [embed_text(t.text_for_embedding()) for t in topics]
    sims = batch_cosine_similarity(qv, vectors)
    pairs = list(zip(topics, sims))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[: max(1, top_k)]


def trend_score(citations: int, year: int, now_year: int = 2026) -> float:
    age = max(1, now_year - int(year or now_year))
    return math.log1p(max(0, citations)) / float(age)
