# app/integrations/semantic_scholar.py
# Semantic Scholar API client — free, no key needed (rate limited)

import httpx
import asyncio
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarClient:

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def search_papers(
        self, query: str, limit: int = 5, year_range: Optional[str] = None
    ) -> list[dict]:
        """Search for papers by query string.

        Args:
            query: Search query (title, keywords, etc.).
            limit: Max number of results.
            year_range: Optional year filter like "2020-2024".

        Returns:
            List of paper dicts with title, authors, year, abstract, citationCount, url.
        """
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,citationCount,url,externalIds",
        }
        if year_range:
            params["year"] = year_range

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{BASE_URL}/paper/search", params=params)
                resp.raise_for_status()
                data = resp.json()

            papers = []
            for item in data.get("data", []):
                papers.append({
                    "title": item.get("title", ""),
                    "authors": [a.get("name", "") for a in item.get("authors", [])],
                    "year": item.get("year"),
                    "abstract": item.get("abstract", ""),
                    "citation_count": item.get("citationCount", 0),
                    "url": item.get("url", ""),
                    "source": "semantic_scholar",
                })
            logger.info(f"Semantic Scholar returned {len(papers)} results for '{query[:50]}'.")
            return papers

        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []

    def get_related_papers(self, paper_title: str, limit: int = 5) -> list[dict]:
        """Get related papers by searching the title."""
        return self.search_papers(paper_title, limit=limit)
