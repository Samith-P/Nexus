import os

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
            results.append({
                "title": p["title"],
                "citations": p.get("citationCount", 0),
                "year": p.get("year", 2022),
                "source": "Semantic Scholar"
            })
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
                results.append({
                    "title": w.get("title"),
                    "citations": w.get("cited_by_count", 0),
                    "year": w.get("publication_year", 2022),
                    "source": "OpenAlex"
                })
        except Exception:
            pass

    return results
