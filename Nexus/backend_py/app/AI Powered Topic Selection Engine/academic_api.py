import os
import re
import xml.etree.ElementTree as ET

import requests
try:
    from api_keys import SEMANTIC_SCHOLAR_API_KEY  # type: ignore
except Exception:
    SEMANTIC_SCHOLAR_API_KEY = None

# Prefer environment variable if api_keys.py isn't providing a value
if not SEMANTIC_SCHOLAR_API_KEY:
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

try:
    from openalexapi import OpenAlex  # type: ignore
except Exception:
    OpenAlex = None

def fetch_academic_topics(query, limit=6):
    results = []

    def _norm_title(t: str) -> str:
        return re.sub(r"\s+", " ", (t or "").strip()).lower()

    def _append_unique(item: dict):
        title = str(item.get("title") or "").strip()
        if not title:
            return
        k = _norm_title(title)
        if not k:
            return
        if k in seen_titles:
            return
        seen_titles.add(k)
        results.append(item)

    seen_titles: set[str] = set()

    # ===== Semantic Scholar =====
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    try:
        ss = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,year,citationCount"
            },
            headers=headers,
            timeout=10
        ).json()

        for p in ss.get("data", []):
            _append_unique(
                {
                    "title": p.get("title"),
                    "citations": p.get("citationCount", 0),
                    "year": p.get("year", 2022),
                    "source": "Semantic Scholar",
                }
            )
    except Exception:
        pass

    # ===== Crossref (fallback only) =====
    # Uses the free Crossref REST API. Enable by default, but only hit it when we
    # still need more candidates.
    crossref_fallback = os.getenv("CROSSREF_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
    if crossref_fallback and len(results) < limit:
        try:
            rows = max(5, min(int(limit) * 3, 50))
        except Exception:
            rows = 20

        try:
            cr = requests.get(
                "https://api.crossref.org/works",
                params={"query": query, "rows": rows},
                timeout=12,
                headers={"User-Agent": "ResearchAI-Nexus/1.0 (topic-engine)"},
            ).json()

            items = ((cr.get("message") or {}).get("items") or [])
            for it in items:
                if not isinstance(it, dict):
                    continue
                titles = it.get("title") or []
                title = titles[0] if isinstance(titles, list) and titles else (it.get("title") if isinstance(it.get("title"), str) else None)
                year = 2022
                try:
                    issued = (it.get("issued") or {}).get("date-parts") or []
                    if issued and isinstance(issued, list) and issued[0] and isinstance(issued[0], list):
                        year = int(issued[0][0])
                except Exception:
                    year = 2022

                _append_unique(
                    {
                        "title": title,
                        "citations": int(it.get("is-referenced-by-count", 0) or 0),
                        "year": year,
                        "source": "Crossref",
                    }
                )
                if len(results) >= limit:
                    break
        except Exception:
            pass

    # ===== arXiv (fallback only) =====
    # Uses the arXiv Atom API (no key required).
    arxiv_fallback = os.getenv("ARXIV_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
    if arxiv_fallback and len(results) < limit:
        try:
            max_results = max(5, min(int(limit) * 3, 50))
        except Exception:
            max_results = 20

        try:
            resp = requests.get(
                "http://export.arxiv.org/api/query",
                params={"search_query": f"all:{query}", "start": 0, "max_results": max_results},
                timeout=12,
                headers={"User-Agent": "ResearchAI-Nexus/1.0 (topic-engine)"},
            )
            resp.raise_for_status()

            # Parse Atom feed.
            root = ET.fromstring(resp.text)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("a:entry", ns):
                title_el = entry.find("a:title", ns)
                published_el = entry.find("a:published", ns)

                title = (title_el.text or "").strip() if title_el is not None else ""
                title = re.sub(r"\s+", " ", title).strip()
                year = 2022
                if published_el is not None and (published_el.text or "").strip():
                    try:
                        year = int(str(published_el.text).strip()[:4])
                    except Exception:
                        year = 2022

                _append_unique({"title": title, "citations": 0, "year": year, "source": "arXiv"})
                if len(results) >= limit:
                    break
        except Exception:
            pass

    # ===== OpenAlex (fallback only) =====
    # Semantic Scholar is PRIMARY. OpenAlex is used only when:
    # - Semantic Scholar returned fewer than `limit` results, AND
    # - OpenAlex is installed, AND
    # - OPENALEX_FALLBACK is not disabled.
    openalex_fallback = os.getenv("OPENALEX_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
    if openalex_fallback and len(results) < limit and OpenAlex is not None:
        try:
            oa = OpenAlex()
            works = oa.get_works(search=query, per_page=limit)
            for w in works:
                _append_unique(
                    {
                        "title": w.get("title"),
                        "citations": w.get("cited_by_count", 0),
                        "year": w.get("publication_year", 2022),
                        "source": "OpenAlex",
                    }
                )
                if len(results) >= limit:
                    break
        except Exception:
            pass

    return results[: max(1, int(limit) if str(limit).isdigit() else 6)]
