from __future__ import annotations

import hashlib
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .embedding import cosine_similarity, embed_text
from .qdrant_store import ensure_collection, qdrant_enabled, search as qdrant_search, upsert_topics


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}

_BANNED_TOKENS = {
    "grant",
    "grants",
    "incentive",
    "incentives",
    "subsidy",
    "subsidies",
    "budget",
    "tax",
    "rebate",
    "loan",
    "scheme",
    "goms",
    "ms",
    "annexure",
    "section",
    "chapter",
    "article",
    "rupees",
    "crore",
    "lakh",
}

_DOMAIN_KEYWORDS = {
    "Clean Energy": ["solar", "wind", "energy", "storage", "hydrogen", "smart grid", "ev", "battery"],
    "AgriTech": ["agri", "crop", "irrig", "soil", "farm", "yield", "livestock"],
    "EdTech": ["education", "learning", "curriculum", "teacher", "student", "school"],
    "HealthTech": ["health", "clinical", "hospital", "medical", "diagnos", "disease"],
    "Aerospace": ["drone", "uav", "aerospace"],
    "Innovation": ["innovation", "startup", "incubator", "accelerator", "research"],
    "Remote Sensing": [
        "satellite",
        "remote",
        "sensing",
        "geospatial",
        "gis",
        "ndvi",
        "multispectral",
        "hyperspectral",
        "imagery",
    ],
}


@lru_cache(maxsize=1)
def _vector_size() -> int:
    return len(embed_text("vector-size-probe"))


def _engine_root() -> Path:
    return Path(__file__).resolve().parent


def _datasets_root() -> Path:
    # The repo includes: Datasets/AI-Powered-Topic-Selection-Engine/
    # (ASCII hyphens inside the Datasets folder)
    root = _engine_root() / "Datasets"
    if not root.exists():
        return root

    direct = root / "AI-Powered-Topic-Selection-Engine"
    if direct.exists():
        return direct

    # Fallback: pick first directory under Datasets/
    for p in root.iterdir():
        if p.is_dir():
            return p
    return root


def iter_dataset_files() -> Iterable[Path]:
    base = _datasets_root()
    if not base.exists():
        return []

    files: List[Path] = []
    for ext in ("*.pdf", "*.PDF", "*.xlsx", "*.xls", "*.XLSX", "*.XLS"):
        files.extend(base.rglob(ext))

    # Keep deterministic order
    return sorted({p.resolve() for p in files})


def _chunk_text(text: str, chunk_chars: int = 900, overlap: int = 150) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if chunk_chars <= 0:
        return [t]
    overlap = max(0, min(overlap, chunk_chars - 1))

    chunks: List[str] = []
    i = 0
    n = len(t)
    while i < n:
        j = min(n, i + chunk_chars)
        chunk = t[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(j - overlap, i + 1)
    return chunks


def _tokenize(text: str) -> List[str]:
    # Keep letters/numbers; split on non-word
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", (text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 3]


def _top_terms(tokens: Sequence[str], top_n: int) -> List[str]:
    if not tokens:
        return []
    cnt = Counter(tokens)
    # Drop extremely frequent generic tokens in policy PDFs
    for generic in ("government", "andhra", "pradesh", "department", "policy"):
        cnt.pop(generic, None)
    return [t for t, _ in cnt.most_common(top_n)]


def _top_bigrams(tokens: Sequence[str], top_n: int) -> List[str]:
    if len(tokens) < 2:
        return []
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    cnt = Counter(bigrams)
    return [t for t, _ in cnt.most_common(top_n)]


def _infer_domains(tokens: Sequence[str]) -> List[str]:
    domains: List[str] = []
    tokset = set(tokens)
    for domain, kws in _DOMAIN_KEYWORDS.items():
        for kw in kws:
            if any(t.startswith(kw) for t in tokset):
                domains.append(domain)
                break
    return domains or ["Other"]


def extract_text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""

    try:
        reader = PdfReader(str(path))
        out_parts: List[str] = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                out_parts.append(txt)
        return "\n".join(out_parts)
    except Exception:
        return ""


def extract_strings_from_excel(path: Path, max_cells: int = 250_000) -> List[str]:
    # Prefer openpyxl because it’s a common dependency and robust.
    try:
        import openpyxl  # type: ignore
    except Exception:
        return []

    values: List[str] = []
    try:
        wb = openpyxl.load_workbook(filename=str(path), read_only=True, data_only=True)
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                for v in row:
                    if v is None:
                        continue
                    s = str(v).strip()
                    if not s:
                        continue
                    values.append(s)
                    if len(values) >= max_cells:
                        return values
        return values
    except Exception:
        return values


def _pick_intent_sentence(text: str) -> str:
    # Very light heuristic: first sentence containing a domain keyword.
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    for s in sentences:
        ss = s.strip()
        if not ss:
            continue
        ls = ss.lower()
        if any(kw in ls for kws in _DOMAIN_KEYWORDS.values() for kw in kws):
            return ss[:240]
    return (sentences[0].strip()[:240]) if sentences else ""


def _policy_weight_for_filename(name: str) -> float:
    n = (name or "").lower()
    if "nep" in n:
        return 1.8
    if "policy" in n:
        return 1.6
    if "go" in n or "goms" in n:
        return 1.4
    return 1.2


_DATASETS_COLLECTION = "datasets_chunks"
_DATASETS_INDEXED = False
_POLICIES_COLLECTION = "dataset_policies"
_POLICY_RECORDS: List[Dict] = []


@lru_cache(maxsize=1)
def _embedded_chunks() -> List[Dict]:
    out: List[Dict] = []
    for f in iter_dataset_files():
        if f.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(f)
        else:
            text = "\n".join(extract_strings_from_excel(f))

        for idx, chunk in enumerate(_chunk_text(text, chunk_chars=900, overlap=150)):
            if not chunk.strip():
                continue
            out.append(
                {
                    "chunk_id": f"{f.stem}:{idx}",
                    "policy": f.stem,
                    "doc": f.name,
                    "text": chunk,
                    "vector": embed_text(chunk),
                }
            )
    return out


def _filter_semantic_tokens(tokens: Sequence[str], limit: int = 80) -> List[str]:
    clean = []
    for t in tokens:
        if t in _BANNED_TOKENS:
            continue
        if t.isdigit():
            continue
        clean.append(t)
    return clean[:limit]


def _policy_records() -> List[Dict]:
    if _POLICY_RECORDS:
        return _POLICY_RECORDS

    records: List[Dict] = []
    for f in iter_dataset_files():
        if f.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(f)
        else:
            text = "\n".join(extract_strings_from_excel(f))

        tokens = _tokenize(text)
        tokens = _filter_semantic_tokens(tokens)
        top_terms = _top_terms(tokens, 40)
        top_bi = _top_bigrams(tokens, 20)
        phrases = list(dict.fromkeys([*top_terms, *top_bi]))

        domains = _infer_domains(tokens)
        intent = _pick_intent_sentence(text)

        records.append(
            {
                "policy_id": f.stem,
                "policy_name": f.stem,
                "weight": _policy_weight_for_filename(f.name),
                "domains": domains,
                "intent": intent,
                "phrases": phrases,
                "doc": f.name,
            }
        )

    _POLICY_RECORDS.extend(records)
    return _POLICY_RECORDS


def _embedded_policies() -> List[Dict]:
    out: List[Dict] = []
    for r in _policy_records():
        text = " ".join(r.get("phrases") or [])
        vec = embed_text(text)
        out.append({**r, "vector": vec})
    return out


def synthetic_topics_from_policies(max_topics: int = 30) -> List[Dict]:
    """Generate research-style topic titles from policy content when academic APIs fail."""
    topics: List[Dict] = []
    
    # Domain-specific research templates
    templates = {
        "EdTech": [
            "AI-Powered Learning Analytics for {phrase}",
            "Digital Transformation in {phrase}",
            "Machine Learning Applications in {phrase}",
        ],
        "AgriTech": [
            "Smart {phrase} Using IoT and AI",
            "Precision Agriculture: {phrase} Optimization",
            "Machine Learning for {phrase} Prediction",
        ],
        "Clean Energy": [
            "{phrase} Integration in Smart Grids",
            "AI-Based {phrase} Forecasting Systems",
            "Sustainable {phrase} Management Using ML",
        ],
        "Innovation": [
            "Technology Innovation in {phrase}",
            "AI-Driven {phrase} Solutions",
            "Digital Solutions for {phrase}",
        ],
    }
    
    for r in _policy_records():
        policy_name = r.get("policy_name", "")
        domains = r.get("domains") or ["Other"]
        weight = float(r.get("weight", 1.0) or 1.0)
        phrases = r.get("phrases") or []
        
        # Filter to meaningful multi-word phrases (2-4 words, no numbers/artifacts)
        good_phrases = []
        for p in phrases:
            words = p.split()
            if 2 <= len(words) <= 4 and not any(c.isdigit() for c in p):
                if len(p) >= 8 and len(p) <= 60:  # reasonable length
                    good_phrases.append(p)
        
        if not good_phrases:
            continue
        
        # Pick appropriate templates for domain
        domain = domains[0]
        tmpl_list = templates.get(domain, templates["Innovation"])
        
        # Generate 2-3 topics per policy using best phrases
        for phrase in good_phrases[:2]:
            for tmpl in tmpl_list[:1]:  # Use 1 template per phrase
                title = tmpl.format(phrase=phrase.title())
                topics.append({
                    "title": title,
                    "domain": domain,
                    "policy_tags": [policy_name] if policy_name else [],
                    "citations": 10,  # synthetic baseline citations
                    "year": 2024,
                    "policy_weight_hint": weight,
                    "intent": f"Research aligned with {policy_name}" if policy_name else "",
                })
                if len(topics) >= max_topics:
                    return topics
    
    return topics


def index_datasets_into_qdrant(
    chunk_chars: int = 900,
    overlap: int = 150,
) -> bool:
    """Index ALL dataset content (PDF + XLSX) into Qdrant as chunk embeddings.

    This enables full-content retrieval for policy alignment.
    """

    global _DATASETS_INDEXED
    if _DATASETS_INDEXED:
        return True

    if not qdrant_enabled():
        return False

    vec_size = _vector_size()
    if vec_size <= 0:
        return False

    if not ensure_collection(vec_size, collection_name=_DATASETS_COLLECTION):
        return False

    points: List[Tuple[str, Sequence[float], Dict]] = []
    for c in _embedded_chunks():
        vec = c.get("vector")
        if not vec or len(vec) != vec_size:
            continue
        doc = c.get("doc", "")
        pid = c.get("chunk_id") or hashlib.sha1((doc + str(c.get("chunk_id", ""))).encode("utf-8", errors="ignore")).hexdigest()[:16]
        payload = {
            "doc_name": Path(doc).stem,
            "file_name": doc,
            "file_ext": Path(doc).suffix.lower(),
            "weight": float(_policy_weight_for_filename(doc)),
            "chunk_index": int(str(c.get("chunk_id", "")).split(":")[-1]) if ":" in str(c.get("chunk_id", "")) else 0,
            "text": c.get("text", ""),
            "policy": c.get("policy", ""),
        }
        points.append((pid, vec, payload))

    if not points:
        return False
    ok = upsert_topics(points, collection_name=_DATASETS_COLLECTION)
    if ok:
        _DATASETS_INDEXED = True
    return ok


def search_datasets_chunks(query_text: str, top_k: int = 8) -> List[Dict]:
    """Retrieve most relevant chunks from the indexed datasets."""

    qv = embed_text(query_text)

    # Try Qdrant first
    if qdrant_enabled():
        index_datasets_into_qdrant()
        hits = qdrant_search(qv, top_k=top_k, collection_name=_DATASETS_COLLECTION) or []
        out: List[Dict] = []
        for _id, score, payload in hits:
            p = dict(payload or {})
            p["_score"] = float(score)
            out.append(p)
        if out:
            return out

    # Fallback: in-memory search over embedded chunks
    out: List[Dict] = []
    for c in _embedded_chunks():
        vec = c.get("vector")
        if not vec:
            continue
        s = cosine_similarity(qv, vec)
        doc = c.get("doc", "")
        out.append({"_score": s, "policy": c.get("policy", ""), "doc": doc, "doc_name": Path(doc).stem, "text": c.get("text", "")})
    out.sort(key=lambda x: x["_score"], reverse=True)
    return out[: max(1, top_k)]


def index_policies_into_qdrant() -> bool:
    if not qdrant_enabled():
        return False
    vec_size = _vector_size()
    if vec_size <= 0:
        return False
    if not ensure_collection(vec_size, collection_name=_POLICIES_COLLECTION):
        return False

    points: List[Tuple[str, Sequence[float], Dict]] = []
    for r in _embedded_policies():
        vec = r.get("vector")
        if not vec or len(vec) != vec_size:
            continue
        payload = {
            "policy_name": r.get("policy_name", ""),
            "domains": r.get("domains", []),
            "weight": float(r.get("weight", 1.0) or 1.0),
            "intent": r.get("intent", ""),
            "doc": r.get("doc", ""),
        }
        points.append((r.get("policy_id", r.get("policy_name", "")), vec, payload))

    if not points:
        return False
    return upsert_topics(points, collection_name=_POLICIES_COLLECTION)


def search_policies(query: str, top_k: int = 5) -> List[Dict]:
    qv = embed_text(query)

    if qdrant_enabled() and index_policies_into_qdrant():
        hits = qdrant_search(qv, top_k=top_k, collection_name=_POLICIES_COLLECTION) or []
        out: List[Dict] = []
        for _id, score, payload in hits:
            p = dict(payload or {})
            p["_score"] = float(score)
            out.append(p)
        if out:
            return out

    # Fallback: in-memory similarity on semantic phrases
    out: List[Dict] = []
    for r in _embedded_policies():
        vec = r.get("vector")
        if not vec:
            continue
        s = cosine_similarity(qv, vec)
        out.append(
            {
                "policy_name": r.get("policy_name", ""),
                "domains": r.get("domains", []),
                "weight": float(r.get("weight", 1.0) or 1.0),
                "intent": r.get("intent", ""),
                "doc": r.get("doc", ""),
                "_score": s,
            }
        )
    out.sort(key=lambda x: x["_score"], reverse=True)
    return out[: max(1, top_k)]


@lru_cache(maxsize=1)
def build_policy_weight_table_from_datasets(
    max_keywords_per_doc: int = 40,
    max_bigrams_per_doc: int = 15,
) -> List[Dict]:
    base = _datasets_root()
    if not base.exists():
        return []

    table: List[Dict] = []

    for p in iter_dataset_files():
        if p.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(p)
            tokens = _tokenize(text)
            keywords = _top_terms(tokens, max_keywords_per_doc)
            keywords.extend(_top_bigrams(tokens, max_bigrams_per_doc))
            keywords = list(dict.fromkeys([k for k in keywords if k]))

            if keywords:
                table.append(
                    {
                        "policy": p.stem,
                        "weight": _policy_weight_for_filename(p.name),
                        "keywords": keywords,
                        "source": "Datasets/PDF",
                    }
                )

        elif p.suffix.lower() in {".xlsx", ".xls"}:
            strings = extract_strings_from_excel(p)
            joined = "\n".join(strings)
            tokens = _tokenize(joined)
            keywords = _top_terms(tokens, max_keywords_per_doc)
            keywords.extend(_top_bigrams(tokens, max_bigrams_per_doc))
            keywords = list(dict.fromkeys([k for k in keywords if k]))

            if keywords:
                table.append(
                    {
                        "policy": p.stem,
                        "weight": 1.25,
                        "keywords": keywords,
                        "source": "Datasets/Excel",
                    }
                )

    return table


def match_policy_tags(topic_text: str) -> List[str]:
    """Return policy tags using policy-level semantic retrieval."""

    try:
        import os

        hit_min = float(os.getenv("POLICY_HIT_MIN", "0.25") or 0.25)
    except Exception:
        hit_min = 0.25

    hits = search_policies(topic_text, top_k=5) or []
    tags: List[str] = []
    for h in hits:
        if float(h.get("_score", 0.0) or 0.0) < hit_min:
            continue
        name = str(h.get("policy_name", "")).strip()
        if name:
            tags.append(name)
    return list(dict.fromkeys(tags))
