from __future__ import annotations

from typing import Any

import requests

from ..config import settings


def search_tavily(query: str, *, limit: int = 10, debug: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    if not settings.tavily_api_key:
        if debug is not None:
            debug.setdefault("tavily_errors", []).append({"query": q, "error": "TAVILY_API_KEY is not set"})
        return []

    max_results = max(1, min(int(limit), 20))

    try:
        payload = {
            "api_key": settings.tavily_api_key,
            "query": q,
            "search_depth": settings.tavily_search_depth or "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
        }
        resp = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            headers={"User-Agent": "ResearchAI-Nexus/1.0 (plagiarism-engine)"},
            timeout=25,
        )
        if debug is not None:
            debug.setdefault("tavily", []).append({"query": q, "status": int(getattr(resp, "status_code", 0) or 0)})
        resp.raise_for_status()

        data = resp.json() or {}
        results = data.get("results") or []
        out: list[dict[str, Any]] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            url = str(r.get("url") or "").strip() or None
            title = str(r.get("title") or "").strip() or None
            content = str(r.get("content") or "").strip()
            if not (url or title or content):
                continue
            out.append(
                {
                    "paperId": f"tavily:{url}" if url else "tavily",
                    "title": title,
                    "abstract": content,
                    "year": None,
                    "venue": None,
                    "url": url,
                    "externalIds": {},
                    "authors": [],
                    "source": "Tavily",
                }
            )

        return out
    except Exception as e:
        if debug is not None:
            debug.setdefault("tavily_errors", []).append({"query": q, "error": str(e)})
        return []
