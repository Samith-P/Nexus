from __future__ import annotations

import re


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text or "") if m.group(0)}


def token_jaccard(a: str, b: str) -> float:
    """Simple lexical similarity in [0,1] using Jaccard over token sets.

    This is intentionally dependency-free (no rapidfuzz) and fast enough for
    reranking a handful of candidates.
    """

    sa = _tokenize(a)
    sb = _tokenize(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / union) if union else 0.0
