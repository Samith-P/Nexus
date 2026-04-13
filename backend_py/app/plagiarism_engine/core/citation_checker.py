from __future__ import annotations

import re


_CITATION_PATTERNS = [
    # Numeric: [1], [1, 2], [1-3]
    re.compile(r"\[[0-9]{1,4}(\s*(,|-)\s*[0-9]{1,4})*\]"),
    # Author-year variants: (Smith, 2020), (Smith 2020), (Smith et al., 2020)
    re.compile(r"\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?\s*,?\s*(19|20)\d{2}[a-z]?\)"),
    # (Smith & Doe, 2020)
    re.compile(r"\([A-Z][A-Za-z\-]+\s*(?:&|and)\s*[A-Z][A-Za-z\-]+\s*,\s*(19|20)\d{2}[a-z]?\)"),
    # Smith et al. (2020)
    re.compile(r"\b[A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?\s*\(\s*(19|20)\d{2}[a-z]?\s*\)"),
    # DOI / URL
    re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b"),
    re.compile(r"https?://\S+"),
]


def has_citation(text: str) -> bool:
    t = text or ""
    return any(p.search(t) for p in _CITATION_PATTERNS)


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_source_cited_in_document(*, document_text: str, doi: str | None = None, title: str | None = None, authors: list[str] | None = None, year: int | str | None = None, url: str | None = None) -> bool:
    """Heuristic check: does the paper appear to be cited somewhere in the document?

    We treat DOI as strongest evidence. Otherwise fall back to checking
    author last-name + year, or a reasonably specific title substring.
    """

    doc = _norm(document_text)
    if not doc:
        return False

    # DOI match (very strong)
    if doi:
        d = _norm(doi)
        d = d.replace("https://doi.org/", "").replace("http://doi.org/", "")
        if d and d in doc:
            return True

    # URL match (medium)
    if url:
        u = _norm(url)
        # avoid matching just the scheme
        if len(u) >= 12 and u in doc:
            return True

    # Author + year match (medium)
    y = None
    try:
        y = int(str(year)) if year is not None else None
    except Exception:
        y = None

    last_names: list[str] = []
    if authors:
        for a in authors:
            a = (a or "").strip()
            if not a:
                continue
            parts = a.replace(",", " ").split()
            if parts:
                last = parts[-1].strip(". ")
                if 2 <= len(last) <= 32:
                    last_names.append(last.lower())

    if last_names and y:
        # Check a few last names to avoid overly strict matching.
        for ln in last_names[:3]:
            if ln in doc and str(y) in doc:
                return True

    # Title match (weak-medium). Check for a specific substring of the title.
    if title:
        t = _norm(title)
        if len(t) >= 18 and t in doc:
            return True

        # Token-based fallback: require multiple uncommon tokens
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", t) if len(tok) >= 5]
        tokens = tokens[:8]
        if len(tokens) >= 4:
            hits = sum(1 for tok in tokens if tok in doc)
            if hits >= 3:
                return True

    return False
