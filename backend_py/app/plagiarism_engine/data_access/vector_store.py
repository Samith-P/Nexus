from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..core.semantic_similarity import batch_embed_text, cosine_similarity


@dataclass
class VectorDoc:
    id: str
    text: str
    meta: dict[str, Any]
    vector: list[float]


def build_docs(sources: list[dict[str, Any]]) -> list[VectorDoc]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for i, s in enumerate(sources):
        text = (s.get("abstract") or s.get("text") or "").strip()
        if not text:
            continue
        doi = (s.get("externalIds") or {}).get("DOI") if isinstance(s.get("externalIds"), dict) else s.get("doi")
        authors = None
        if isinstance(s.get("authors"), list):
            authors = [
                str(a.get("name") or "").strip()
                for a in s.get("authors")
                if isinstance(a, dict) and (a.get("name") or "").strip()
            ]
        src_name = str(s.get("source") or s.get("venue") or "Source")
        pid = str(s.get("paperId") or s.get("id") or "")
        is_external = False
        if pid.startswith(("openalex:", "arxiv:", "tavily:", "s2:")):
            is_external = True
        if src_name.strip().lower() in {"openalex", "semantic scholar", "arxiv", "tavily"}:
            is_external = True

        meta = {
            "source": src_name,
            "title": s.get("title"),
            "url": s.get("url"),
            "doi": doi,
            "venue": s.get("venue"),
            "year": s.get("year"),
            "authors": authors,
            "is_external": bool(is_external),
        }
        rows.append((str(s.get("paperId") or s.get("id") or i), text, meta))

    if not rows:
        return []

    vecs = batch_embed_text([t for _, t, _ in rows])
    docs: list[VectorDoc] = []
    for (doc_id, text, meta), vec in zip(rows, vecs):
        docs.append(VectorDoc(id=doc_id, text=text, meta=meta, vector=list(vec)))
    return docs


def search(query_vec: Sequence[float], docs: list[VectorDoc], *, top_k: int = 5) -> list[tuple[float, VectorDoc]]:
    scored: list[tuple[float, VectorDoc]] = [(cosine_similarity(query_vec, d.vector), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[: max(1, int(top_k))]
