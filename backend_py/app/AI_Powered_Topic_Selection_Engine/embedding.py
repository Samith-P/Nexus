from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, List, Sequence


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return float(dot / denom) if denom else 0.0


def _hashing_bow(text: str, dim: int = 256) -> List[float]:
    vec = [0.0] * dim
    for tok in (text or "").lower().split():
        vec[hash(tok) % dim] += 1.0
    return vec


@lru_cache(maxsize=1)
def _try_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer("ai4bharat/indic-bert")
    except Exception:
        return None


def embed_text(text: str) -> List[float]:
    """Multilingual embedding for user queries/topics.

    Uses `ai4bharat/indic-bert` when `sentence-transformers` is installed.
    Falls back to a hashing BoW vector to keep the PoC runnable.
    """

    model = _try_sentence_transformer()
    if model is None:
        return _hashing_bow(text)
    vec = model.encode(text or "")
    try:
        return [float(x) for x in vec.tolist()]
    except Exception:
        return [float(x) for x in vec]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return _cosine(a, b)


def batch_cosine_similarity(query_vec: Sequence[float], vectors: Iterable[Sequence[float]]) -> List[float]:
    return [cosine_similarity(query_vec, v) for v in vectors]
