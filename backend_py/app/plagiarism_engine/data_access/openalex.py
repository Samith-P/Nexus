from __future__ import annotations

from typing import Any

import requests

from ..config import settings


def _invert_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv or not isinstance(inv, dict):
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inv.items():
        if not isinstance(word, str) or not isinstance(idxs, list):
            continue
        for i in idxs:
            if isinstance(i, int):
                positions[i] = word
    if not positions:
        return ""
    # OpenAlex abstracts can be long; cap to ~2000 tokens.
    ordered = [positions[i] for i in sorted(positions.keys())][:2000]
    return " ".join(ordered)


def _normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip()
    if not d:
        return None
    d = d.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return d.strip() or None


def search_works(query: str, *, limit: int = 10, debug: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    try:
        params: dict[str, Any] = {
            "search": q,
            "per-page": max(1, min(int(limit), 200)),
        }

        # Optional fields that improve rate limits and API friendliness.
        if settings.openalex_api_key:
            params["api_key"] = settings.openalex_api_key
        if settings.openalex_mailto:
            params["mailto"] = settings.openalex_mailto

        resp = requests.get(
            "https://api.openalex.org/works",
            params=params,
            headers={"User-Agent": "ResearchAI-Nexus/1.0 (plagiarism-engine)"},
            timeout=20,
        )
        if debug is not None:
            debug.setdefault("openalex", []).append({"query": q, "status": int(getattr(resp, "status_code", 0) or 0)})
        resp.raise_for_status()
        data = resp.json() or {}

        out: list[dict[str, Any]] = []
        for w in (data.get("results") or []):
            if not isinstance(w, dict):
                continue

            oa_id = str(w.get("id") or "").strip()
            title = str(w.get("title") or "").strip() or None
            year = w.get("publication_year")
            year = int(year) if isinstance(year, int) else None

            abstract = _invert_abstract(w.get("abstract_inverted_index"))
            doi = _normalize_doi(w.get("doi"))

            venue = None
            host = w.get("host_venue")
            if isinstance(host, dict):
                venue = host.get("display_name")

            url = None
            primary_location = w.get("primary_location")
            if isinstance(primary_location, dict):
                url = (primary_location.get("landing_page_url") or primary_location.get("pdf_url") or None)

            if not url and oa_id:
                url = oa_id

            authors: list[dict[str, Any]] = []
            for a in (w.get("authorships") or [])[:20]:
                if not isinstance(a, dict):
                    continue
                author = a.get("author")
                if isinstance(author, dict):
                    name = str(author.get("display_name") or "").strip()
                    if name:
                        authors.append({"name": name})

            out.append(
                {
                    "paperId": f"openalex:{oa_id}" if oa_id else "openalex",
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "venue": venue,
                    "url": url,
                    "externalIds": {"DOI": doi} if doi else {},
                    "authors": authors,
                    "source": "OpenAlex",
                }
            )

        return out
    except Exception as e:
        if debug is not None:
            debug.setdefault("openalex_errors", []).append({"query": q, "error": str(e)})
        return []
