from __future__ import annotations

import math
import os
import re
import zlib
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


def _stable_hash(token: str) -> int:
    return int(zlib.crc32((token or "").encode("utf-8", errors="ignore")) & 0xFFFFFFFF)


def _hashing_bow(text: str, dim: int = 256) -> List[float]:
    vec = [0.0] * dim
    # Tokenize more like a word tokenizer (punctuation-safe) and use a stable hash.
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}", (text or "").lower()):
        vec[_stable_hash(tok) % dim] += 1.0
    return vec


@lru_cache(maxsize=1)
def _try_sentence_transformer():
    enabled = os.getenv("USE_SENTENCE_TRANSFORMERS", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    if not enabled:
        return None

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model_name = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "ai4bharat/indic-bert").strip() or "ai4bharat/indic-bert"
        return SentenceTransformer(model_name)
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
