from __future__ import annotations

import math
import os
import time
import threading
from typing import Iterable, Sequence

import requests

from ..config import settings


_LOCAL_MODEL_LOCK = threading.Lock()
_LOCAL_MODEL = None
_LOCAL_MODEL_NAME: str | None = None


def _get_local_model():
    global _LOCAL_MODEL, _LOCAL_MODEL_NAME
    model_name = (getattr(settings, "local_model_name", "") or os.getenv("SENTENCE_TRANSFORMERS_MODEL") or "all-MiniLM-L6-v2").strip()
    if _LOCAL_MODEL is not None and _LOCAL_MODEL_NAME == model_name:
        return _LOCAL_MODEL
    with _LOCAL_MODEL_LOCK:
        if _LOCAL_MODEL is not None and _LOCAL_MODEL_NAME == model_name:
            return _LOCAL_MODEL
        from sentence_transformers import SentenceTransformer  # type: ignore

        device = (getattr(settings, "local_device", "cpu") or "cpu").strip() or "cpu"
        _LOCAL_MODEL = SentenceTransformer(model_name, device=device)
        _LOCAL_MODEL_NAME = model_name
        return _LOCAL_MODEL


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
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
    import zlib

    return int(zlib.crc32((token or "").encode("utf-8", errors="ignore")) & 0xFFFFFFFF)


def _hashing_bow(text: str, dim: int = 256) -> list[float]:
    import re

    vec = [0.0] * dim
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}", (text or "").lower()):
        vec[_stable_hash(tok) % dim] += 1.0
    return vec


def _mean_pool(features: list) -> list[float]:
    if not features:
        return []
    if isinstance(features, list) and len(features) == 1 and isinstance(features[0], list) and features and isinstance(features[0][0], list):
        features = features[0]
    if not features or not isinstance(features[0], list):
        return []

    seq = features
    hidden = len(seq[0]) if seq and isinstance(seq[0], list) else 0
    if hidden <= 0:
        return []
    out = [0.0] * hidden
    count = 0
    for row in seq:
        if not isinstance(row, list) or len(row) != hidden:
            continue
        for i, v in enumerate(row):
            try:
                out[i] += float(v)
            except Exception:
                out[i] += 0.0
        count += 1
    if count <= 0:
        return []
    return [v / count for v in out]


def embed_text_hf_api(text: str) -> list[float]:
    token = settings.hf_token
    model = settings.hf_model
    if not token or not model:
        return _hashing_bow(text)

    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": text or ""}

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 503:
                time.sleep(1.5 + attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            vec = _mean_pool(data)
            return vec if vec else _hashing_bow(text)
        except Exception:
            time.sleep(0.5)
    return _hashing_bow(text)


def batch_embed_text_hf_api(texts: list[str]) -> list[list[float]]:
    token = settings.hf_token
    model = settings.hf_model
    if not token or not model:
        return [_hashing_bow(t) for t in texts]

    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
    headers = {"Authorization": f"Bearer {token}"}

    # HF Inference API can become slow/unreliable with very large batches.
    batch_size = max(1, min(int(getattr(settings, "hf_batch_size", 16) or 16), 64))
    if len(texts) > batch_size:
        out_all: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            out_all.extend(batch_embed_text_hf_api(texts[i : i + batch_size]))
        return out_all

    payload = {"inputs": texts}

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 503:
                time.sleep(1.5 + attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            # Expect: [batch][seq][hidden]
            if isinstance(data, list) and data and isinstance(data[0], list):
                out: list[list[float]] = []
                for item in data:
                    out.append(_mean_pool(item) or _hashing_bow(""))
                if len(out) == len(texts):
                    return out
        except Exception:
            time.sleep(0.5)

    # Fallback: per item
    return [embed_text_hf_api(t) for t in texts]


def embed_text(text: str) -> list[float]:
    provider = (settings.embed_provider or "hf_api").strip().lower()
    if provider == "hf_api":
        return embed_text_hf_api(text)
    if provider == "hash":
        return _hashing_bow(text)

    if provider in {"sentence_transformers", "local"}:
        # Use the batched path to avoid re-loading the model per call.
        return batch_embed_text([text or ""])[0]

    return embed_text_hf_api(text)


def batch_embed_text(texts: Iterable[str]) -> list[list[float]]:
    items = list(texts)
    provider = (settings.embed_provider or "hf_api").strip().lower()
    if provider == "hf_api":
        return batch_embed_text_hf_api(items)
    if provider == "hash":
        return [_hashing_bow(t) for t in items]
    if provider in {"sentence_transformers", "local"}:
        if not items:
            return []
        try:
            model = _get_local_model()
        except Exception as e:
            raise RuntimeError(
                "Local embeddings are enabled (PLAG_EMBED_PROVIDER=local) but sentence-transformers could not be loaded. "
                "Install 'sentence-transformers' in the runtime environment."
            ) from e

        batch_size = max(1, min(int(getattr(settings, "local_batch_size", 64) or 64), 512))
        # Normalized embeddings make cosine similarity a cheap dot product.
        vecs = model.encode(
            [t or "" for t in items],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        out: list[list[float]] = []
        # sentence-transformers returns numpy arrays by default.
        for v in vecs:
            try:
                out.append([float(x) for x in v.tolist()])
            except Exception:
                out.append([float(x) for x in v])
        return out

    return [embed_text(t) for t in items]
