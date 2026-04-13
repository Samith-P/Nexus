# app/integrations/crossref.py
# CrossRef API client — free, no key needed

import httpx
import re
from typing import Optional

from literature_review.utils.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.crossref.org/works"


class CrossRefClient:

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def search_by_title(self, title: str, limit: int = 5) -> list[dict]:
        """Search CrossRef for papers by title.

        Returns:
            List of paper dicts with title, authors, year, doi, url, source.
        """
        params = {
            "query.bibliographic": title,
            "rows": limit,
            "select": "title,author,published-print,published-online,DOI,URL",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

            papers = []
            for item in data.get("message", {}).get("items", []):
                # Extract year
                year = None
                for date_field in ["published-print", "published-online"]:
                    date_parts = item.get(date_field, {}).get("date-parts", [[]])
                    if date_parts and date_parts[0]:
                        year = date_parts[0][0]
                        break

                papers.append({
                    "title": (item.get("title") or [""])[0],
                    "authors": [
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in item.get("author", [])
                    ],
                    "year": year,
                    "doi": item.get("DOI", ""),
                    "url": item.get("URL", ""),
                    "source": "crossref",
                })

            logger.info(f"CrossRef returned {len(papers)} results for '{title[:50]}'.")
            return papers

        except Exception as e:
            logger.error(f"CrossRef search failed: {e}")
            return []

    def get_by_doi(self, doi: str) -> Optional[dict]:
        """Fetch a specific paper by DOI."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{BASE_URL}/{doi}")
                resp.raise_for_status()
                item = resp.json().get("message", {})

            year = None
            for date_field in ["published-print", "published-online"]:
                date_parts = item.get(date_field, {}).get("date-parts", [[]])
                if date_parts and date_parts[0]:
                    year = date_parts[0][0]
                    break

            return {
                "title": (item.get("title") or [""])[0],
                "authors": [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in item.get("author", [])
                ],
                "year": year,
                "doi": item.get("DOI", ""),
                "url": item.get("URL", ""),
                "source": "crossref",
            }
        except Exception as e:
            logger.error(f"CrossRef DOI lookup failed for {doi}: {e}")
            return None

    @staticmethod
    def extract_dois_from_text(text: str) -> list[str]:
        """Extract DOI strings from paper text."""
        pattern = r"10\.\d{4,9}/[-._;/:A-Z0-9]+[A-Z0-9]"
        return list(set(re.findall(pattern, text, re.IGNORECASE)))
