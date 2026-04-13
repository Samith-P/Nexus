from __future__ import annotations

import re


def split_into_paragraphs(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]


def split_into_sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\(\[])", t)
    out: list[str] = []
    for p in parts:
        s = p.strip()
        if len(s) < 20:
            continue
        out.append(s)
    return out
