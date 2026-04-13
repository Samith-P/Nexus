from __future__ import annotations

import re
import zlib
from dataclasses import dataclass


def _normalize(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _hash(s: str) -> int:
    return int(zlib.crc32(s.encode("utf-8", errors="ignore")) & 0xFFFFFFFF)


@dataclass(frozen=True)
class Fingerprint:
    h: int
    pos: int


def winnow(text: str, *, k: int = 5, w: int = 4) -> list[Fingerprint]:
    """Winnowing fingerprinting over word tokens."""

    norm = _normalize(text)
    if not norm:
        return []

    tokens = norm.split()
    if len(tokens) < k:
        return []

    hashes: list[tuple[int, int]] = []
    for i in range(0, len(tokens) - k + 1):
        gram = " ".join(tokens[i : i + k])
        hashes.append((_hash(gram), i))

    window = max(1, int(w))
    fps: list[Fingerprint] = []
    last_selected: tuple[int, int] | None = None

    for start in range(0, max(1, len(hashes) - window + 1)):
        win = hashes[start : start + window]
        min_h = min(h for h, _ in win)
        candidates = [item for item in win if item[0] == min_h]
        chosen = candidates[-1]  # rightmost on ties
        if last_selected != chosen:
            fps.append(Fingerprint(h=chosen[0], pos=chosen[1]))
            last_selected = chosen

    return fps


def fingerprint_set(text: str, *, k: int = 5, w: int = 4) -> set[int]:
    return {fp.h for fp in winnow(text, k=k, w=w)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return float(inter / union) if union else 0.0
