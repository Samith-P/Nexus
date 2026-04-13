from __future__ import annotations

from typing import Any


def _authors_str(authors: list[str] | None, *, max_authors: int = 4) -> str:
    if not authors:
        return ""
    names = [str(a).strip() for a in authors if str(a).strip()]
    if not names:
        return ""
    if len(names) > max_authors:
        names = names[:max_authors] + ["et al."]
    return ", ".join(names)


def format_apa(meta: dict[str, Any]) -> str:
    authors = _authors_str(meta.get("authors"))
    year = meta.get("year")
    title = (meta.get("title") or "").strip()
    venue = (meta.get("venue") or meta.get("source") or "").strip()
    doi = (meta.get("doi") or "").strip()
    url = (meta.get("url") or "").strip()

    parts: list[str] = []
    if authors:
        parts.append(f"{authors}.")
    if year:
        parts.append(f"({year}).")
    if title:
        parts.append(f"{title}.")
    if venue:
        parts.append(f"{venue}.")
    if doi:
        parts.append(f"https://doi.org/{doi}")
    elif url:
        parts.append(url)
    return " ".join(p for p in parts if p).strip() or (title or url or "")


def format_ieee(meta: dict[str, Any]) -> str:
    authors = _authors_str(meta.get("authors"))
    year = meta.get("year")
    title = (meta.get("title") or "").strip()
    venue = (meta.get("venue") or meta.get("source") or "").strip()
    doi = (meta.get("doi") or "").strip()
    url = (meta.get("url") or "").strip()

    parts: list[str] = []
    if authors:
        parts.append(authors)
    if title:
        parts.append(f"\"{title}\"")
    if venue:
        parts.append(venue)
    if year:
        parts.append(str(year))
    if doi:
        parts.append(f"doi:{doi}")
    elif url:
        parts.append(url)
    return ", ".join(p for p in parts if p).strip() or (title or url or "")
