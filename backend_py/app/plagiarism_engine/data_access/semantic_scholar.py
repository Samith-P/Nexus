from __future__ import annotations

import re
import time
from typing import Any

import requests

from ..config import settings


def _headers() -> dict[str, str]:
    h: dict[str, str] = {"User-Agent": "ResearchAI-Nexus/1.0 (plagiarism-engine)"}
    if settings.semantic_scholar_api_key:
        h["x-api-key"] = settings.semantic_scholar_api_key
    return h


def _get_with_backoff(
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str],
    timeout: int = 20,
    retries: int = 2,
    debug: dict[str, Any] | None = None,
    debug_query: str | None = None,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            status = int(getattr(resp, "status_code", 0) or 0)
            if debug is not None:
                debug.setdefault("semantic_scholar", []).append(
                    {
                        "query": debug_query,
                        "status": status,
                        "attempt": attempt,
                    }
                )

            # Retry on rate-limit and transient server errors.
            if status == 429 or 500 <= status < 600:
                if attempt < retries:
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        wait_s = float(retry_after) if retry_after is not None else float(1.0 * (2**attempt))
                    except Exception:
                        wait_s = float(1.0 * (2**attempt))
                    # Keep it bounded to avoid long request stalls.
                    if wait_s < 0.5:
                        wait_s = 0.5
                    if wait_s > 4.0:
                        wait_s = 4.0
                    time.sleep(wait_s)
                    continue

            return resp
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(float(0.5 * (2**attempt)))
                continue
            raise

    # Should be unreachable, but keep mypy happy.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Semantic Scholar request failed")


def search_papers(query: str, *, limit: int = 10, debug: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    try:
        resp = _get_with_backoff(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": q,
                "limit": max(1, min(int(limit), 100)),
                "fields": "paperId,title,abstract,year,venue,url,externalIds,authors",
            },
            headers=_headers(),
            timeout=20,
            retries=2,
            debug=debug,
            debug_query=q,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        out = []
        for p in data.get("data", []) or []:
            if isinstance(p, dict):
                out.append(p)
        return out
    except Exception as e:
        if debug is not None:
            debug.setdefault("semantic_scholar_errors", []).append({"query": q, "error": str(e)})
        return []


def get_paper(paper_id: str) -> dict[str, Any] | None:
    pid = (paper_id or "").strip()
    if not pid:
        return None
    try:
        resp = _get_with_backoff(
            f"https://api.semanticscholar.org/graph/v1/paper/{pid}",
            params={"fields": "paperId,title,abstract,year,venue,url,externalIds,authors"},
            headers=_headers(),
            timeout=20,
            retries=2,
        )
        resp.raise_for_status()
        obj = resp.json() or {}
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def get_paper_by_doi(doi: str) -> dict[str, Any] | None:
    d = (doi or "").strip()
    if not d:
        return None
    d = re.sub(r"^doi:\s*", "", d, flags=re.IGNORECASE).strip()
    try:
        resp = _get_with_backoff(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{d}",
            params={"fields": "paperId,title,abstract,year,venue,url,externalIds,authors"},
            headers=_headers(),
            timeout=20,
            retries=2,
        )
        resp.raise_for_status()
        obj = resp.json() or {}
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
