from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def search_arxiv(query: str, *, limit: int = 10, debug: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    max_results = max(1, min(int(limit), 50))
    url = (
        "http://export.arxiv.org/api/query?search_query="
        + quote_plus(f"all:{q}")
        + f"&start=0&max_results={max_results}"
    )

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "ResearchAI-Nexus/1.0 (plagiarism-engine)"},
            timeout=20,
        )
        if debug is not None:
            debug.setdefault("arxiv", []).append({"query": q, "status": int(getattr(resp, "status_code", 0) or 0)})
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        out: list[dict[str, Any]] = []

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip() or None
            summary = (entry.findtext("{http://www.w3.org/2005/Atom}summary") or "").strip()
            eid = (entry.findtext("{http://www.w3.org/2005/Atom}id") or "").strip()
            published = (entry.findtext("{http://www.w3.org/2005/Atom}published") or "").strip()

            authors: list[dict[str, Any]] = []
            for a in entry.findall("{http://www.w3.org/2005/Atom}author"):
                name = (a.findtext("{http://www.w3.org/2005/Atom}name") or "").strip()
                if name:
                    authors.append({"name": name})

            year = None
            if len(published) >= 4 and published[:4].isdigit():
                year = int(published[:4])

            out.append(
                {
                    "paperId": f"arxiv:{eid}" if eid else "arxiv",
                    "title": title,
                    "abstract": summary,
                    "year": year,
                    "venue": "arXiv",
                    "url": eid or None,
                    "externalIds": {},
                    "authors": authors,
                    "source": "arXiv",
                }
            )

        return out
    except Exception as e:
        if debug is not None:
            debug.setdefault("arxiv_errors", []).append({"query": q, "error": str(e)})
        return []
