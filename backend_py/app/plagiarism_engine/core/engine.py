from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import settings
from ..data_access.semantic_scholar import get_paper, get_paper_by_doi, search_papers
from ..data_access.openalex import search_works
from ..data_access.arxiv import search_arxiv
from ..data_access.tavily import search_tavily
from ..data_access.external_cache import load as cache_load, save as cache_save
from ..data_access.qdrant_store import (
    ensure_collection as qdrant_ensure_collection,
    qdrant_enabled,
    search as qdrant_search,
    to_point_id as qdrant_to_point_id,
    upsert_points as qdrant_upsert_points,
)
from ..data_access.source_loader import load_local_sources
from ..data_access.vector_store import build_docs, search as vector_search
from ..schemas.request import PlagiarismRequest
from ..schemas.response import MatchItem, MissingCitationItem
from .chunking import split_into_sentences
from .citation_checker import has_citation, is_source_cited_in_document
from .document_parser import extract_text_from_any, extract_text_from_upload
from .fingerprinting import fingerprint_set, jaccard
from .report_generator import build_report
from .scoring import compute_scores
from .semantic_similarity import batch_embed_text, cosine_similarity
from .text_similarity import token_jaccard


def _normalize_title_for_match(title: str) -> str:
    """Normalize titles for strict equality/containment checks.

    Purpose: avoid treating generic semantic similarity as a same-document match.
    """

    import re
    import unicodedata

    t = (title or "").strip().lower()
    if not t:
        return ""
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    # Keep alphanumerics and spaces only; remove punctuation differences.
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_strong_title_match(t1: str, t2: str) -> bool:
    """Return True only for exact/near-exact title matches.

    This intentionally rejects generic-but-related titles, e.g.
    "A Survey on Machine Learning" vs "A survey on machine learning for data fusion".
    """

    n1 = _normalize_title_for_match(t1)
    n2 = _normalize_title_for_match(t2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True

    # Allow small variations (punctuation, short suffix/prefix changes).
    if abs(len(n1) - len(n2)) < 10 and (n1 in n2 or n2 in n1):
        return True

    return False


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hash_text(t: str) -> str:
    return hashlib.sha256((t or "").encode("utf-8", errors="ignore")).hexdigest()


def _cache_path(user_id: str, key: str) -> Path:
    base = Path(settings.cache_dir)
    safe_user = "".join(c for c in (user_id or "user") if c.isalnum() or c in {"-", "_"}) or "user"
    p = base / safe_user
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{key}.json"


def _extract_keywords(text: str, *, max_terms: int = 12) -> list[str]:
    import re

    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    toks = [w for w in t.split() if 4 <= len(w) <= 18]
    stop = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "were",
        "been",
        "their",
        "there",
        "which",
        "these",
        "those",
        "into",
        "using",
        "used",
        "use",
        "such",
        "also",
        "between",
        "within",
        "study",
        "paper",
        "research",
        "results",
        "method",
        "methods",
        "analysis",
        "data",
    }
    freq: dict[str, int] = {}
    for w in toks:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in terms[: max_terms]]


def _first_n_words(text: str, *, n: int) -> str:
    toks = [t for t in (text or "").strip().split() if t]
    return " ".join(toks[:n])


def _normalize_pdf_text(text: str) -> str:
    """Normalize common PDF-extracted glyph quirks.

    This helps title extraction/search when PDFs contain ligatures like "ﬁ".
    """

    if not text:
        return ""
    # Common ligatures
    return (
        (text or "")
        .replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\ufb00", "ff")
        .replace("\ufb03", "ffi")
        .replace("\ufb04", "ffl")
        .replace("\u00ad", "")  # soft hyphen
    )


def _guess_title(text: str) -> str | None:
    """Best-effort title extraction from the start of the document.

    Goal: avoid grabbing body sentences like "X is ..." and avoid affiliations.
    """

    if not text:
        return None

    # Normalize literal "\\n" sequences if they leaked in.
    norm_text = _normalize_pdf_text((text or "").replace("\\r\\n", "\n").replace("\\n", "\n"))
    lines_raw = [ln.strip() for ln in norm_text.splitlines() if (ln or "").strip()]
    if not lines_raw:
        return None

    # Stop scanning once we reach common section headers.
    section_break_tokens = {"abstract", "introduction", "keywords"}
    skip_substrings = {
        "abstract",
        "introduction",
        "keywords",
        "references",
        "acknowledg",
        "copyright",
        "all rights reserved",
        "issn",
        "volume",
        "issue",
        "arxiv",
        "doi",
        "http://",
        "https://",
        "www.",
        "department",
        "university",
        "institute",
        "faculty",
        "school",
        "laboratory",
        "laboratories",
    }
    # If the 2nd token is a verb, it's very likely a body sentence (e.g., "BERT is ...").
    second_token_verbs = {
        "is",
        "are",
        "was",
        "were",
        "be",
        "has",
        "have",
        "had",
        "introduce",
        "introduces",
        "present",
        "presents",
        "propose",
        "proposes",
        "show",
        "shows",
        "demonstrate",
        "demonstrates",
    }

    def looks_like_title(line: str) -> bool:
        if not line:
            return False
        if "@" in line:
            return False

        lower = line.lower()
        if any(tok in lower for tok in skip_substrings):
            return False
        # Don't accept citation-ish lines.
        if "et al" in lower and ("(" in lower or ")" in lower):
            return False
        # Basic length bounds.
        if not (20 <= len(line) <= 180):
            return False
        # Reject obvious sentence endings.
        if line.endswith(".") or line.endswith("…"):
            return False
        # Reject all-lowercase lines.
        if line.islower():
            return False
        # Reject if it looks like a body sentence starting with pronouns.
        if lower.startswith(("we ", "this ", "our ", "in this ")):
            return False
        # Reject if 2nd token is a verb ("X is ...").
        parts = [p for p in lower.replace(":", " ").split() if p]
        if len(parts) >= 2 and parts[1] in second_token_verbs:
            return False
        # Require a decent alphabetic ratio.
        alpha = sum(ch.isalpha() for ch in line)
        if alpha < 0.55 * max(1, len(line)):
            return False
        return True

    # Scan top lines and pick the first strong title candidate.
    chosen: str | None = None
    for ln in lines_raw[:40]:
        lower = ln.lower()
        if any(t in lower for t in section_break_tokens):
            break
        ln = " ".join(ln.split())
        if looks_like_title(ln):
            chosen = ln
            break

    if not chosen:
        return None

    return chosen.strip(" -–—:;")


def _clean_title(title: str) -> str:
    """Normalize/cap a title string for search queries.

    Important: this function must not attempt to "re-guess" a title from body text.
    """

    import re

    max_chars = max(50, int(getattr(settings, "clean_query_max_chars", 150) or 150))
    t = (title or "").replace("\\r\\n", "\n").replace("\\n", "\n").strip()
    t = _normalize_pdf_text(t)
    t = " ".join(t.split())
    t = t.strip(" \t-–—:;,.\"")
    if not t:
        return ""

    def _looks_like_author_tail_tokens(tokens: list[str]) -> bool:
        if len(tokens) < 5:
            return False
        caps = 0
        alpha = 0
        for tok in tokens:
            tt = tok.strip(" ,.;:()[]{}")
            if not tt:
                continue
            if not any(ch.isalpha() for ch in tt):
                continue
            alpha += 1
            first_alpha = next((ch for ch in tt if ch.isalpha()), "")
            if first_alpha and first_alpha.isupper():
                caps += 1
        if alpha >= 5 and (caps / max(1, alpha)) >= 0.9:
            return True
        return False

    def _strip_author_tail(s: str) -> str:
        # If a PDF line got flattened like: "<Title> Jacob Devlin Ming-Wei Chang ...",
        # chop off the author list tail.
        parts = [p for p in (s or "").split() if p]
        if len(parts) < 10:
            return s
        # Search for a 5-token window that looks like person names.
        for i in range(6, min(len(parts) - 4, 30)):
            window = parts[i : i + 5]
            if _looks_like_author_tail_tokens(window):
                return " ".join(parts[:i]).strip()
        return s

    t = _strip_author_tail(t)

    # Remove trailing affiliation/author-ish fragments if they leaked into the title line.
    t = (
        re.split(
            r"\b(department|university|institute|faculty|school|laboratory|abstract|keywords|student|professor)\b",
            t,
            flags=re.IGNORECASE,
        )[0]
        .strip()
        or t
    )
    # Split on common author/designation prefixes.
    t = re.split(r"\b(mr\.?|mrs\.?|ms\.?|dr\.?|prof\.?|assistant professor|associate professor)\b", t, flags=re.IGNORECASE)[0].strip() or t

    def _smart_titlecase(s: str) -> str:
        acronyms = {"AI", "ML", "NLP", "XAI", "BERT", "CNN", "RNN", "LSTM", "GAN", "LLM"}
        out: list[str] = []
        for raw in (s or "").split():
            w = raw
            # Keep tokens that contain digits/symbols or are already acronym-ish.
            letters = "".join(ch for ch in w if ch.isalpha())
            if w.upper() in acronyms:
                out.append(w.upper())
                continue
            if any(ch.isdigit() for ch in w) or "-" in w or "/" in w:
                out.append(w)
                continue
            if letters.isupper() and len(letters) <= 3:
                out.append(w.upper())
                continue
            # Title-case word, but keep small prepositions lower unless first.
            base = w.lower().capitalize()
            out.append(base)
        # Lowercase common short words (except first word)
        small = {"and", "or", "of", "to", "in", "on", "for", "with", "toward", "towards"}
        if out:
            out = [out[0]] + [tok.lower() if tok.lower() in small else tok for tok in out[1:]]
        return " ".join(out)

    # If PDF extraction yields ALL CAPS titles, normalize for better retrieval.
    letters_only = "".join(ch for ch in t if ch.isalpha())
    if letters_only and letters_only.isupper():
        t = _smart_titlecase(t)
    return t[:max_chars]


def _extract_locked_title(document_text: str) -> str:
    """Extract and LOCK a single document title to drive doc-level search.

    This is the *only* title used for document-level retrieval. It must not be
    overwritten later by chunks, fallbacks, or heuristics.
    """

    if not document_text:
        return ""

    # IMPORTANT: lock a single title early and never override it.
    norm_text = _normalize_pdf_text((document_text or "").replace("\\r\\n", "\n").replace("\\n", "\n"))
    title_line = _guess_title(norm_text) or ""

    # If the title is split across multiple lines (common in PDFs), join continuation lines.
    # Example:
    #   "... Concepts, Taxonomies"
    #   "Opportunities and Challenges ..."
    title = title_line
    if title_line:
        lines = [" ".join(ln.strip().split()) for ln in norm_text.splitlines() if (ln or "").strip()]

        def looks_like_author_or_affiliation(line: str) -> bool:
            lower = (line or "").lower()
            if "@" in line:
                return True
            if any(
                t in lower
                for t in (
                    "university",
                    "department",
                    "institute",
                    "faculty",
                    "school",
                    "laboratory",
                    "assistant professor",
                    "associate professor",
                    "professor",
                    "student",
                )
            ):
                return True
            if lower.startswith(("mr ", "mr.", "mrs ", "mrs.", "ms ", "ms.", "dr ", "dr.", "prof ", "prof.")):
                return True
            # Likely author list: many capitalized name tokens.
            toks = [t.strip(" ,.;:()[]{}") for t in (line or "").split() if t.strip(" ,.;:()[]{}")]
            if len(toks) >= 5:
                alpha_toks = [t for t in toks if any(ch.isalpha() for ch in t)]
                if alpha_toks:
                    cap = 0
                    for t in alpha_toks:
                        ch0 = next((ch for ch in t if ch.isalpha()), "")
                        if ch0.isupper():
                            cap += 1
                    # Avoid treating titles with punctuation as author lists.
                    if (cap / max(1, len(alpha_toks))) >= 0.9 and all(p not in line for p in (":", "(", ")")):
                        return True
            # Likely author list: initials and commas/and.
            if "," in line and (" and " in lower or "." in line):
                return True
            if any(ch.isdigit() for ch in line) and len(line) < 80:
                # footnote markers / affiliations
                return True
            return False

        def looks_like_title_continuation(line: str) -> bool:
            if not line:
                return False
            lower = line.lower()
            if any(
                t in lower
                for t in (
                    "abstract",
                    "introduction",
                    "keywords",
                    "department",
                    "university",
                    "assistant professor",
                    "associate professor",
                    "professor",
                    "student",
                )
            ):
                return False
            if looks_like_author_or_affiliation(line):
                return False
            if line.islower():
                return False
            if len(line) < 5 or len(line) > 220:
                return False
            # Avoid obvious sentences.
            if line.endswith(".") and len(line.split()) > 12:
                return False
            alpha = sum(ch.isalpha() for ch in line)
            if alpha < 0.50 * max(1, len(line)):
                return False
            return True

        start_idx = None
        for i, ln in enumerate(lines[:40]):
            if ln == title_line:
                start_idx = i
                break
        if start_idx is not None:
            parts = [title_line]
            for ln in lines[start_idx + 1 : start_idx + 8]:
                if looks_like_title_continuation(ln):
                    parts.append(ln)
                else:
                    break
            title = " ".join(parts)

    # Safer fallback than "first non-empty line": choose the first decent line near the top.
    if not title:
        lines = [ln.strip() for ln in norm_text.splitlines() if (ln or "").strip()]
        for ln in lines[:40]:
            lower = ln.lower()
            if any(t in lower for t in ("abstract", "introduction", "keywords")):
                break
            ln = " ".join(ln.split())
            if 20 <= len(ln) <= 180 and "@" not in ln and not ln.islower():
                title = ln
                break

    return _clean_title(title)


def _clean_query(text: str) -> str:
    """Extract a short, search-friendly title query from document text."""

    import re

    max_chars = max(50, int(getattr(settings, "clean_query_max_chars", 150) or 150))
    # Normalize literal escape sequences (e.g., Swagger/clients may send "\\n" instead of newlines).
    norm_text = (text or "").replace("\\r\\n", "\n").replace("\\n", "\n")
    lines = [ln.strip() for ln in norm_text.splitlines() if (ln or "").strip()]
    head = lines[:10]

    # Heuristic: choose the longest "meaningful" line near the top, skipping authors/affiliations.
    skip_tokens = {
        "university",
        "department",
        "institute",
        "school",
        "faculty",
        "laboratory",
        "laboratories",
        "abstract",
        "keywords",
        "arxiv",
        "http://",
        "https://",
        "www.",
    }

    def is_title_candidate(line: str) -> bool:
        if not line:
            return False
        if "@" in line:
            return False
        lower = line.lower()
        if any(tok in lower for tok in skip_tokens):
            return False
        if not (20 < len(line) < 200):
            return False
        # Reject lines that look like a paragraph sentence.
        if lower.endswith(".") and "," in lower and len(lower.split()) > 18:
            return False
        # Prefer lines with a healthy alphabetic ratio.
        alpha = sum(ch.isalpha() for ch in line)
        if alpha < 0.55 * max(1, len(line)):
            return False
        return True

    best = ""
    for ln in head:
        if is_title_candidate(ln) and len(ln) > len(best):
            best = ln

    if not best:
        # Fall back to the previous title heuristic, then first non-empty line.
        best = (_guess_title(norm_text) or "")
        if not best and head:
            best = head[0]

    best = " ".join((best or "").split())
    if not best:
        return ""

    best = re.split(r"\b(department|university|institute|faculty|school|abstract)\b", best, flags=re.IGNORECASE)[0].strip() or best
    return best[:max_chars]


def _norm_title(s: str) -> str:
    import re

    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def _title_similarity(a: str, b: str) -> float:
    """Title similarity in [0,1] using a blend of sequence ratio + token overlap."""

    na = _norm_title(a)
    nb = _norm_title(b)
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(a=na, b=nb).ratio()
    jac = token_jaccard(na, nb)
    return float(max(seq, jac))


def _semantic_accept(*, cosine: float, threshold: float, lexical: float | None = None) -> bool:
    """Decide whether to accept a semantic match.

    - Always accept if cosine >= threshold
    - Also accept slightly-below-threshold cosine when lexical overlap is strong enough
      (helps avoid 0% reports from borderline vector scores).
    """

    try:
        c = float(cosine)
    except Exception:
        c = 0.0

    try:
        th = float(threshold)
    except Exception:
        th = 0.0

    if c >= th:
        return True

    if lexical is None:
        return False

    try:
        l = float(lexical)
    except Exception:
        l = 0.0

    # Allow up to 0.08 below threshold if lexical overlap is meaningful.
    return (c >= (th - 0.08)) and (l >= 0.12)


def _build_queries(text: str, *, chunks: list[str]) -> list[str]:
    """Generate search queries that are likely to return the exact source paper."""

    import re

    queries: list[str] = []

    def _sanitize_query(q: str) -> str:
        raw = (q or "").strip()
        if not raw:
            return ""

        lower = raw.lower()
        bad_patterns = ["www.", "issn", "volume", "issue", "doi", "©", "copyright"]
        if any(p in lower for p in bad_patterns):
            return ""

        qn = _normalize_pdf_text(raw.replace("\\r\\n", "\n").replace("\\n", "\n"))
        qn = " ".join(qn.split())
        qn = re.sub(r"[^A-Za-z0-9\s]", " ", qn)
        qn = re.sub(r"\s+", " ", qn).strip()
        return qn

    # Prefer the locked title over chunk prefixes (chunk prefixes tend to be noisy/paragraph-like).
    title = _extract_locked_title(text) or _clean_query(text)
    title = _sanitize_query(title)
    if title:
        queries.append(title)

    # Fall back to keyword-based queries only if needed.
    terms = _extract_keywords(text)
    if terms:
        # Build 1–2 concise keyword phrases.
        top_terms = [t for t in terms[:10] if t]
        if len(top_terms) >= 3:
            queries.append(" ".join(top_terms[:6]))
        if title and len(top_terms) >= 2:
            queries.append(" ".join([title] + top_terms[:4]))

    # De-dup while preserving order and cap length for URL/query sanity.
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        qn = _sanitize_query(q)
        if not qn:
            continue
        qn = qn[:220]
        key = qn.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(qn)
        if len(out) >= settings.external_max_queries:
            break
    return out


def run_check(*, req: PlagiarismRequest, file_bytes: bytes | None = None, filename: str | None = None) -> Any:
    report_id = str(uuid4())
    # Cache key must include a version salt because scoring/override logic changes frequently.
    # Otherwise old 0%-plagiarism reports can persist after code updates.
    cache_salt = "plag_report_v4"
    scoring_salt = f"doc_override={float(getattr(settings, 'document_title_override_threshold', 0.50) or 0.50):.3f}"
    raw_input_key = _hash_bytes(file_bytes) if file_bytes else _hash_text(req.text or req.doi or req.paper_id or "")
    raw_key = _hash_text(f"{cache_salt}|{scoring_salt}|{raw_input_key}")
    cache_fp = _cache_path(req.user_id, raw_key)
    if cache_fp.exists():
        try:
            import json

            return json.loads(cache_fp.read_text(encoding="utf-8"))
        except Exception:
            pass

    external_debug: dict[str, Any] = {}

    text = extract_text_from_any(text=req.text)
    if not text and file_bytes is not None:
        text = extract_text_from_upload(file_bytes, filename)
    if not text and (req.doi or req.paper_id):
        p = get_paper(req.paper_id) if req.paper_id else get_paper_by_doi(req.doi or "")
        text = (p or {}).get("abstract") or (p or {}).get("title") or ""

    if not text.strip():
        return build_report(
            report_id=report_id,
            status="error",
            errors=["No input content. Provide file, text, doi, or paper_id."],
            originality_score=0.0,
            plagiarism_percentage=0.0,
            exact_match_percentage=0,
            semantic_match_percentage=0.0,
            citation_coverage_percentage=0.0,
            matches=[],
            missing_citations=[],
            debug=None,
        ).model_dump(exclude_none=True)

    # LOCK title once; do not override later.
    locked_title = _extract_locked_title(text)

    chunks = split_into_sentences(text)
    if not chunks:
        chunks = [text.strip()]

    # STEP 5 — IGNORE TITLE / HEADER CHUNKS
    # (Keep it simple; this only removes obvious headers/noise before any matching.)
    def is_noise(chunk: str) -> bool:
        c = (chunk or "").strip()
        if len(c) < 50:
            return True
        low = c.lower()
        if "abstract" in low or "introduction" in low:
            return True
        return False

    filtered_chunks = [c for c in chunks if not is_noise(c)]
    if filtered_chunks:
        chunks = filtered_chunks

    # Guard: embedding every sentence of very long PDFs is expensive over HF API.
    # Merge sentences into larger chunks to cap compute.
    max_chunks = max(1, int(settings.max_chunks))
    if len(chunks) > max_chunks:
        group = max(2, len(chunks) // max_chunks)
        merged: list[str] = []
        for i in range(0, len(chunks), group):
            merged.append(" ".join(chunks[i : i + group]))
        chunks = merged[:max_chunks]

    local_sources = load_local_sources()
    sources: list[dict[str, Any]] = []
    sources.extend(local_sources)

    external_docs: list[dict[str, Any]] = []
    queries: list[str] = []
    external_query_stats: list[dict[str, Any]] = []
    semantic_chunk_limit_applied: int | None = None

    # Document-level match (title-level). Critical for detecting "same paper upload".
    # Use ONLY the locked title (never chunk text, never fallbacks).
    doc_title_query = locked_title
    doc_level_match: dict[str, Any] | None = None
    doc_level_match_sim: float = 0.0

    if settings.enable_external_checks:
        # 1) FAST / STRONG signal: title-level embedding match against OpenAlex.
        if doc_title_query and settings.enable_openalex:
            title_hits = search_works(
                doc_title_query,
                limit=max(1, int(getattr(settings, "document_title_search_limit", 10) or 10)),
                debug=external_debug,
            )

            cand_titles: list[str] = []
            cand_hits: list[dict[str, Any]] = []
            for h in title_hits:
                ht = str(h.get("title") or "").strip()
                if not ht:
                    continue
                cand_titles.append(ht)
                cand_hits.append(h)
                if len(cand_titles) >= 20:
                    break

            if cand_titles:
                # 1) Title-equality/near-equality should win over embeddings.
                best_title_sim = 0.0
                best_title_hit: dict[str, Any] | None = None
                for ht, h in zip(cand_titles, cand_hits):
                    ts = _title_similarity(doc_title_query, ht)
                    if ts > best_title_sim:
                        best_title_sim = float(ts)
                        best_title_hit = h

                if best_title_hit is not None and best_title_sim >= 0.98:
                    # Treat near-exact title match as same-document evidence.
                    doc_level_match = best_title_hit
                    doc_level_match_sim = 1.0
                else:
                    # 2) Otherwise fall back to embedding similarity.
                    vecs = batch_embed_text([doc_title_query] + cand_titles)
                    qv = vecs[0] if vecs else []
                    for ht, hv, h in zip(cand_titles, vecs[1:], cand_hits):
                        sim = cosine_similarity(qv, hv)
                        if sim > doc_level_match_sim:
                            doc_level_match_sim = float(sim)
                            doc_level_match = h

            th = float(getattr(settings, "document_title_match_threshold", 0.85) or 0.85)
            if (
                doc_level_match is not None
                and doc_level_match_sim >= th
                and is_strong_title_match(doc_title_query, str(doc_level_match.get("title") or ""))
            ):
                # NOTE: Do not early-exit here.
                # We now combine title similarity with actual content overlap (chunk-vs-abstract matches)
                # to decide the final plagiarism percentage.
                pass

        queries = _build_queries(text, chunks=chunks)
        external_query_stats: list[dict[str, Any]] = []
        seen_pid: set[str] = set()
        for q in queries:
            providers_stat = {"SemanticScholar": 0, "OpenAlex": 0, "arXiv": 0, "Tavily": 0}

            # If we've already hit the overall cap, stop early.
            if len(external_docs) >= settings.external_max_docs:
                break

            if settings.enable_semantic_scholar_search:
                cached = cache_load("semantic_scholar", q)
                hits = cached if cached is not None else search_papers(q, limit=settings.external_per_query_results, debug=external_debug)
                if cached is None and hits:
                    cache_save("semantic_scholar", q, hits)
                providers_stat["SemanticScholar"] = int(len(hits or []))
                for h in hits:
                    pid = str(h.get("paperId") or "")
                    if not pid or pid in seen_pid:
                        continue
                    seen_pid.add(pid)
                    external_docs.append(h)
                    if len(external_docs) >= settings.external_max_docs:
                        break

            if settings.enable_openalex and len(external_docs) < settings.external_max_docs:
                cached = cache_load("openalex", q)
                oa_hits = cached if cached is not None else search_works(q, limit=settings.external_per_query_results, debug=external_debug)
                if cached is None and oa_hits:
                    cache_save("openalex", q, oa_hits)
                providers_stat["OpenAlex"] = int(len(oa_hits or []))
                for h in oa_hits:
                    pid = str(h.get("paperId") or "")
                    if not pid or pid in seen_pid:
                        continue
                    seen_pid.add(pid)
                    external_docs.append(h)
                    if len(external_docs) >= settings.external_max_docs:
                        break

            if settings.enable_arxiv and len(external_docs) < settings.external_max_docs:
                cached = cache_load("arxiv", q)
                ax_hits = cached if cached is not None else search_arxiv(q, limit=settings.external_per_query_results, debug=external_debug)
                if cached is None and ax_hits:
                    cache_save("arxiv", q, ax_hits)
                providers_stat["arXiv"] = int(len(ax_hits or []))
                for h in ax_hits:
                    pid = str(h.get("paperId") or "")
                    if not pid or pid in seen_pid:
                        continue
                    seen_pid.add(pid)
                    external_docs.append(h)
                    if len(external_docs) >= settings.external_max_docs:
                        break

            if settings.enable_tavily and len(external_docs) < settings.external_max_docs:
                cached = cache_load("tavily", q)
                tv_hits = cached if cached is not None else search_tavily(q, limit=min(settings.external_per_query_results, 10), debug=external_debug)
                if cached is None and tv_hits:
                    cache_save("tavily", q, tv_hits)
                providers_stat["Tavily"] = int(len(tv_hits or []))
                for h in tv_hits:
                    pid = str(h.get("paperId") or "")
                    if not pid or pid in seen_pid:
                        continue
                    seen_pid.add(pid)
                    external_docs.append(h)
                    if len(external_docs) >= settings.external_max_docs:
                        break

            external_query_stats.append({"query": q, "providers": providers_stat})

            if len(external_docs) >= settings.external_max_docs:
                break

        # Fallback retry: if the first pass yields nothing, try a title-only query and a longer prefix.
        if not external_docs:
            fallback: list[str] = []
            if locked_title:
                fallback.append(locked_title)
            fallback.append(_first_n_words(text, n=25))
            for q in fallback:
                qn = " ".join((q or "").split())
                if not qn:
                    continue
                # Drop common PDF header/footer junk from fallback queries.
                low = qn.lower()
                if any(p in low for p in ["www.", "issn", "volume", "issue", "doi", "©", "copyright"]):
                    continue
                if qn.lower() in {qq.lower() for qq in queries}:
                    continue
                queries.append(qn)
                hits: list[dict[str, Any]] = []
                providers_stat = {"SemanticScholar": 0, "OpenAlex": 0, "arXiv": 0, "Tavily": 0}
                if settings.enable_semantic_scholar_search:
                    cached = cache_load("semantic_scholar", qn)
                    hits = cached if cached is not None else search_papers(qn, limit=settings.external_per_query_results, debug=external_debug)
                    if cached is None and hits:
                        cache_save("semantic_scholar", qn, hits)
                    providers_stat["SemanticScholar"] = int(len(hits or []))
                if not hits and settings.enable_openalex:
                    cached = cache_load("openalex", qn)
                    hits = cached if cached is not None else search_works(qn, limit=settings.external_per_query_results, debug=external_debug)
                    if cached is None and hits:
                        cache_save("openalex", qn, hits)
                    providers_stat["OpenAlex"] = int(len(hits or []))
                if not hits and settings.enable_arxiv:
                    cached = cache_load("arxiv", qn)
                    hits = cached if cached is not None else search_arxiv(qn, limit=settings.external_per_query_results, debug=external_debug)
                    if cached is None and hits:
                        cache_save("arxiv", qn, hits)
                    providers_stat["arXiv"] = int(len(hits or []))
                if not hits and settings.enable_tavily:
                    cached = cache_load("tavily", qn)
                    hits = cached if cached is not None else search_tavily(qn, limit=min(settings.external_per_query_results, 10), debug=external_debug)
                    if cached is None and hits:
                        cache_save("tavily", qn, hits)
                    providers_stat["Tavily"] = int(len(hits or []))

                try:
                    external_query_stats.append({"query": qn, "providers": providers_stat})
                except Exception:
                    pass
                for h in hits:
                    pid = str(h.get("paperId") or "")
                    if not pid or pid in seen_pid:
                        continue
                    seen_pid.add(pid)
                    external_docs.append(h)
                    if len(external_docs) >= settings.external_max_docs:
                        break
                if external_docs:
                    break

    # Merge document-level hits into external docs list and persist to Qdrant.
    if doc_level_match is not None and doc_title_query:
        th = float(getattr(settings, "document_title_match_threshold", 0.85) or 0.85)
        if doc_level_match_sim >= th:
            # Put best title match first for ranking.
            external_docs.insert(0, doc_level_match)

    # Persistent external DB in Qdrant (BIG upgrade): store OpenAlex/arXiv/etc abstracts so future
    # requests can search without re-fetching everything.
    external_collection: str | None = None
    external_qdrant_filter = None
    if (
        external_docs
        and qdrant_enabled()
        and bool(getattr(settings, "qdrant_external_enabled", True))
        and req.check_type in {"semantic", "full", "citation"}
    ):
        try:
            texts: list[str] = []
            payloads: list[dict[str, Any]] = []
            point_ids: list[str | int] = []
            for h in external_docs:
                s_text = (h.get("abstract") or h.get("text") or "").strip()
                if not s_text:
                    continue
                pid = str(h.get("paperId") or h.get("id") or "").strip() or "external"
                src = str(h.get("source") or "External").strip() or "External"
                title = str(h.get("title") or "").strip()
                ext_ids = h.get("externalIds")
                doi = (ext_ids or {}).get("DOI") if isinstance(ext_ids, dict) else h.get("doi")

                authors = None
                if isinstance(h.get("authors"), list):
                    authors = [
                        str(a.get("name") or "").strip()
                        for a in h.get("authors")
                        if isinstance(a, dict) and (a.get("name") or "").strip()
                    ]

                payload = {
                    "provider": src,
                    "source": src,
                    "paperId": pid,
                    "title": title or None,
                    "url": h.get("url"),
                    "doi": doi,
                    "venue": h.get("venue") or h.get("publicationVenue"),
                    "year": h.get("year"),
                    "authors": authors,
                    "is_external": True,
                }

                cap = max(200, int(getattr(settings, "qdrant_payload_text_max_chars", 2000) or 2000))
                combined = ((title + "\n") if title else "") + (s_text or "")
                payload["text"] = combined[:cap]

                texts.append(combined)
                payloads.append(payload)
                point_ids.append(qdrant_to_point_id(f"external:{pid}", namespace="plagiarism"))

            if texts:
                vecs = batch_embed_text(texts)
                vector_size = len(vecs[0]) if vecs and vecs[0] else 0
                if vector_size > 0:
                    external_collection = qdrant_ensure_collection(
                        vector_size,
                        base=str(getattr(settings, "qdrant_external_collection", "external_papers") or "external_papers"),
                    )
                if external_collection:
                    batch_size = int(getattr(settings, "qdrant_upsert_batch_size", 64) or 64)
                    qdrant_upsert_points(
                        external_collection,
                        [(pid, vec, payload) for pid, vec, payload in zip(point_ids, vecs, payloads)],
                        batch_size=batch_size,
                    )

                # Optional filter by source/provider (comma-separated) for external searches.
                src_filter = str(getattr(settings, "qdrant_external_source_filter", "") or "").strip()
                if src_filter:
                    from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

                    values = [v.strip() for v in src_filter.split(",") if v.strip()]
                    if values:
                        external_qdrant_filter = Filter(
                            should=[FieldCondition(key="source", match=MatchValue(value=v)) for v in values]
                        )
        except Exception:
            external_collection = None
            external_qdrant_filter = None

    sources.extend(external_docs)

    # Only build in-memory semantic docs for local sources; external papers are searched via Qdrant.
    docs = build_docs(local_sources)

    # Precompute source fingerprint sets for faster exact comparisons
    source_fps: list[tuple[dict[str, Any], set[int]]] = []
    if req.check_type in {"exact", "full"}:
        for s in sources[: min(len(sources), 50)]:
            s_text = (s.get("abstract") or s.get("text") or "").strip()
            if not s_text:
                continue
            source_fps.append((s, fingerprint_set(s_text)))

    sem_threshold = float(settings.semantic_threshold)
    external_accept_threshold = float(getattr(settings, "external_accept_threshold", 0.75) or 0.75)

    # Final safety: if config/env accidentally sets accept thresholds too high,
    # the system will report doc-level similarity but no chunk matches.
    effective_accept = 0.55
    try:
        sem_threshold = min(float(sem_threshold), float(effective_accept))
    except Exception:
        sem_threshold = float(effective_accept)
    try:
        external_accept_threshold = min(float(external_accept_threshold), float(effective_accept))
    except Exception:
        external_accept_threshold = float(effective_accept)

    # Performance guard: semantic checks require embeddings. Cap how many chunks
    # are embedded (particularly important when using remote HF API). When using
    # a local model, default to embedding all chunks (up to PLAG_MAX_CHUNKS).
    chunk_vecs: list[list[float]] = []
    semantic_chunks = chunks
    if req.check_type in {"semantic", "full", "citation"}:
        provider = (getattr(settings, "embed_provider", "hf_api") or "hf_api").strip().lower()
        if provider in {"local", "sentence_transformers"}:
            limit = max(1, int(getattr(settings, "max_chunks", len(chunks)) or len(chunks)))
        else:
            limit = max(1, int(getattr(settings, "embed_max_chunks", len(chunks)) or len(chunks)))
        semantic_chunk_limit_applied = min(len(chunks), limit)
        semantic_chunks = chunks[:semantic_chunk_limit_applied]
        chunk_vecs = batch_embed_text(semantic_chunks)

        # Ensure the persistent external DB exists for this embedding dimension, even if we didn't
        # fetch any new external docs during this request.
        if (
            external_collection is None
            and qdrant_enabled()
            and bool(getattr(settings, "qdrant_external_enabled", True))
        ):
            try:
                vector_size = len(chunk_vecs[0]) if chunk_vecs and chunk_vecs[0] else 0
                if vector_size > 0:
                    external_collection = qdrant_ensure_collection(
                        vector_size,
                        base=str(getattr(settings, "qdrant_external_collection", "external_papers") or "external_papers"),
                    )
            except Exception:
                external_collection = None

        # Optional filter by source/provider (comma-separated) for external searches.
        if external_collection and external_qdrant_filter is None:
            try:
                src_filter = str(getattr(settings, "qdrant_external_source_filter", "") or "").strip()
                if src_filter:
                    from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

                    values = [v.strip() for v in src_filter.split(",") if v.strip()]
                    if values:
                        external_qdrant_filter = Filter(
                            should=[FieldCondition(key="source", match=MatchValue(value=v)) for v in values]
                        )
            except Exception:
                external_qdrant_filter = None

    doc_fps = fingerprint_set(text) if req.check_type in {"exact", "full"} else set()

    matches_raw: list[dict[str, Any]] = []
    missing: list[MissingCitationItem] = []
    exact_hits = 0
    semantic_hits = 0
    matched_chunks = 0
    cited_chunks = 0

    # STEP 1 — ADD CHUNK vs ABSTRACT MATCH (quick win)
    # Pre-embed external abstracts once, then compare chunk vectors against them.
    abstract_docs: list[dict[str, Any]] = []
    abstract_texts: list[str] = []
    for d in (external_docs or []):
        a = str(d.get("abstract") or "").strip()
        if not a:
            continue
        abstract_docs.append(d)
        abstract_texts.append(a)
    abstract_vecs: list[list[float]] = batch_embed_text(abstract_texts) if abstract_texts else []
    abstract_matched_chunk_indices: set[int] = set()

    # Document-level (title) match indicators.
    # This is independent of chunk-level matching and should not be treated as a chunk match.
    doc_override_th = float(getattr(settings, "document_title_override_threshold", 0.50) or 0.50)
    doc_match_title = str((doc_level_match or {}).get("title") or "")
    doc_exact_title_match = bool(
        doc_title_query
        and doc_match_title
        and (
            _normalize_title_for_match(doc_title_query) == _normalize_title_for_match(doc_match_title)
            or _title_similarity(doc_title_query, doc_match_title) >= 0.98
        )
    )
    doc_strong_title_match = bool(
        doc_exact_title_match
        or (doc_title_query and doc_match_title and is_strong_title_match(doc_title_query, doc_match_title))
    )
    doc_has_match = bool(
        doc_level_match is not None
        and doc_title_query
        and (
            doc_exact_title_match
            or (doc_level_match_sim >= doc_override_th)
        )
    )

    # Internal check: repeated content within the document (semantic duplicates)
    internal_reuse = 0
    internal_reuse_samples: list[dict[str, Any]] = []
    if chunk_vecs:
        for i in range(len(chunk_vecs)):
            best = 0.0
            best_j = None
            # compare with a small rolling window to keep it fast
            for j in range(max(0, i - 40), i):
                sim = cosine_similarity(chunk_vecs[i], chunk_vecs[j])
                if sim > best:
                    best = sim
                    best_j = j
            if best_j is not None and best >= 0.92:
                internal_reuse += 1
                if len(internal_reuse_samples) < 5:
                    internal_reuse_samples.append(
                        {
                            "chunk_index": i,
                            "similar_to": best_j,
                            "similarity": float(best),
                            "text": semantic_chunks[i][:240],
                        }
                    )

    # Weighted plagiarism scoring inputs
    # Score even short-but-meaningful chunks; avoid filtering away everything.
    min_words_for_scoring = int(getattr(settings, "min_chunk_words_for_scoring", 3) or 3)
    eligible_chunks_for_scoring = sum(1 for ch in chunks if len((ch or "").split()) >= min_words_for_scoring)
    plagiarism_match_sims: list[float] = []

    for idx, ch in enumerate(chunks):
        cited_marker = has_citation(ch)

        best_match: dict[str, Any] | None = None
        best_sim = 0.0

        if req.check_type in {"exact", "full"} and ch and len(ch) >= 50:
            ch_fps = fingerprint_set(ch)
            intra = jaccard(ch_fps, doc_fps)
            if intra >= settings.exact_threshold:
                best_match = {
                    "text": ch,
                    "source": "Document (intra)",
                    "similarity": float(intra),
                    "algo": "exact",
                    "url": None,
                    "doi": None,
                    "title": None,
                    "authors": None,
                    "year": None,
                    "venue": None,
                }
                best_sim = intra

            for s, s_fps in source_fps:
                sim = jaccard(ch_fps, s_fps)
                if sim > best_sim and sim >= settings.exact_threshold:
                    best_sim = sim
                    ext_ids = s.get("externalIds")
                    doi = (ext_ids or {}).get("DOI") if isinstance(ext_ids, dict) else s.get("doi")
                    best_match = {
                        "text": ch,
                        "source": str(s.get("source") or "Semantic Scholar"),
                        "similarity": float(sim),
                        "algo": "exact",
                        "url": s.get("url"),
                        "doi": doi,
                        "title": s.get("title"),
                        "authors": None,
                        "year": s.get("year"),
                        "venue": s.get("venue") or s.get("publicationVenue"),
                    }

        if req.check_type in {"semantic", "full", "citation"} and idx < len(chunk_vecs):
            qv = chunk_vecs[idx]

            # External semantic search: Qdrant persistent external paper DB.
            if external_collection and qdrant_enabled() and bool(getattr(settings, "qdrant_external_enabled", True)):
                hits = qdrant_search(
                    external_collection,
                    qv,
                    top_k=max(1, int(getattr(settings, "qdrant_search_top_k", 5) or 5)),
                    qdrant_filter=external_qdrant_filter,
                )
                if hits:
                    alpha = float(getattr(settings, "hybrid_alpha", 0.75) or 0.75)
                    alpha = 0.0 if alpha < 0.0 else 1.0 if alpha > 1.0 else alpha
                    hybrid = bool(getattr(settings, "hybrid_enabled", True))

                    best_quality = -1.0
                    best_payload: dict[str, Any] | None = None
                    best_vec_sim = 0.0
                    best_fuzzy = 0.0
                    for _, vec_sim, payload in hits:
                        try:
                            vec_sim_f = float(vec_sim)
                        except Exception:
                            vec_sim_f = 0.0
                        fuzzy = token_jaccard(ch, str(payload.get("text") or "")) if hybrid else 0.0
                        quality = (alpha * vec_sim_f + (1.0 - alpha) * float(fuzzy)) if hybrid else vec_sim_f
                        if quality > best_quality:
                            best_quality = quality
                            best_payload = payload
                            best_vec_sim = vec_sim_f
                            best_fuzzy = float(fuzzy)

                    if best_payload is not None:
                        accept_th = external_accept_threshold if bool(best_payload.get("is_external", True)) else sem_threshold
                        if _semantic_accept(cosine=best_vec_sim, threshold=accept_th, lexical=best_fuzzy) and best_vec_sim > best_sim:
                            best_sim = best_vec_sim
                            best_match = {
                                "text": ch,
                                "source": str(best_payload.get("source") or "OpenAlex"),
                                "similarity": float(best_vec_sim),
                                "algo": "semantic",
                                "url": best_payload.get("url"),
                                "doi": best_payload.get("doi"),
                                "title": best_payload.get("title"),
                                "authors": best_payload.get("authors"),
                                "year": best_payload.get("year"),
                                "venue": best_payload.get("venue"),
                            }

            # Local semantic search (in-memory)
            if docs:
                top = vector_search(qv, docs, top_k=1)
                if top:
                    sim, doc = top[0]
                    accept_th = external_accept_threshold if bool(doc.meta.get("is_external")) else sem_threshold
                    hybrid = bool(getattr(settings, "hybrid_enabled", True))
                    fuzzy = token_jaccard(ch, str(getattr(doc, "text", "") or "")) if hybrid else 0.0
                    if _semantic_accept(cosine=float(sim), threshold=accept_th, lexical=float(fuzzy)) and sim > best_sim:
                        best_sim = sim
                        best_match = {
                            "text": ch,
                            "source": str(doc.meta.get("source") or "Local"),
                            "similarity": float(sim),
                            "algo": "semantic",
                            "url": doc.meta.get("url"),
                            "doi": doc.meta.get("doi"),
                            "title": doc.meta.get("title"),
                            "authors": doc.meta.get("authors"),
                            "year": doc.meta.get("year"),
                            "venue": doc.meta.get("venue"),
                        }

            # Abstract-only semantic overlap against external docs (cosine > 0.70)
            if abstract_vecs:
                best_abs_sim = 0.0
                best_abs_doc: dict[str, Any] | None = None
                for av, d in zip(abstract_vecs, abstract_docs):
                    sim = cosine_similarity(qv, av)
                    if sim > best_abs_sim:
                        best_abs_sim = float(sim)
                        best_abs_doc = d

                if best_abs_sim > 0.70:
                    abstract_matched_chunk_indices.add(idx)
                    if best_abs_doc is not None and best_abs_sim > best_sim:
                        ext_ids = best_abs_doc.get("externalIds")
                        doi = (ext_ids or {}).get("DOI") if isinstance(ext_ids, dict) else best_abs_doc.get("doi")
                        best_sim = best_abs_sim
                        best_match = {
                            "text": ch,
                            "source": str(best_abs_doc.get("source") or "External"),
                            "similarity": float(best_abs_sim),
                            "algo": "semantic",
                            "url": best_abs_doc.get("url"),
                            "doi": doi,
                            "title": best_abs_doc.get("title"),
                            "authors": best_abs_doc.get("authors"),
                            "year": best_abs_doc.get("year"),
                            "venue": best_abs_doc.get("venue") or best_abs_doc.get("publicationVenue"),
                        }

        if best_match is not None:
            matched_chunks += 1
            matches_raw.append(best_match)
            if best_match.get("algo") == "exact":
                exact_hits += 1
            else:
                semantic_hits += 1

            # Scoring: ignore title-only / tiny chunks.
            if len((ch or "").split()) >= min_words_for_scoring:
                try:
                    plagiarism_match_sims.append(float(best_match.get("similarity") or 0.0))
                except Exception:
                    plagiarism_match_sims.append(0.0)

            globally_cited = False
            try:
                globally_cited = is_source_cited_in_document(
                    document_text=text,
                    doi=best_match.get("doi"),
                    title=best_match.get("title"),
                    authors=best_match.get("authors"),
                    year=best_match.get("year"),
                    url=best_match.get("url"),
                )
            except Exception:
                globally_cited = False

            cited = bool(cited_marker or globally_cited)
            if cited:
                cited_chunks += 1

            # Only flag missing citations when matched content appears reused but has no citation evidence.
            if not cited:
                suggested = best_match.get("doi") or best_match.get("url") or (best_match.get("title") or best_match.get("source"))
                missing.append(
                    MissingCitationItem(text=ch, suggested_source=str(suggested), url=best_match.get("url"), doi=best_match.get("doi"))
                )

    scores = compute_scores(
        total_chunks=len(chunks),
        exact_hits=exact_hits,
        semantic_hits=semantic_hits,
        cited_chunks=cited_chunks,
        citation_total_chunks=matched_chunks,
        match_similarities=plagiarism_match_sims,
        eligible_chunks=eligible_chunks_for_scoring,
        internal_reuse_count=internal_reuse,
    )

    # STEP 2 — COUNT MATCHED CHUNKS (based on abstract semantic overlap)
    total_semantic_chunks = int(len(semantic_chunks) if semantic_chunks else len(chunks))
    abstract_matched_chunks = int(len(abstract_matched_chunk_indices))
    abstract_semantic_match_percentage = (
        (float(abstract_matched_chunks) / float(total_semantic_chunks)) * 100.0 if total_semantic_chunks > 0 else 0.0
    )

    # STEP 4 — ADD MINIMUM MATCH FILTER (VERY IMPORTANT)
    if abstract_matched_chunks < 3:
        abstract_semantic_match_percentage = 0.0

    # STEP 3 — COMBINE WITH TITLE LOGIC
    # Use title similarity (not embeddings) as the override gate.
    title_sim = float(_title_similarity(doc_title_query or "", doc_match_title or "")) if (doc_title_query and doc_match_title) else 0.0
    override_applied = False

    if title_sim >= 0.95:
        plagiarism_percentage = 100.0
        override_applied = True
    elif title_sim >= 0.60:
        plagiarism_percentage = float(max(abstract_semantic_match_percentage, title_sim * 100.0))
        override_applied = True
    else:
        plagiarism_percentage = float(abstract_semantic_match_percentage)

    plagiarism_percentage = float(round(max(0.0, min(100.0, plagiarism_percentage)) + 1e-12, 2))
    scores["semantic_match_percentage"] = float(round(max(0.0, min(100.0, float(abstract_semantic_match_percentage))) + 1e-12, 2))
    scores["plagiarism_percentage"] = plagiarism_percentage
    scores["originality_score"] = float(round(max(0.0, min(100.0, 100.0 - plagiarism_percentage)) + 1e-12, 2))
    scores["internal_reuse_count"] = internal_reuse

    debug: dict[str, Any] | None = None
    if settings.debug:
        import re

        extracted_sample = (text or "")[:1200]

        chunk_lens = [len(c or "") for c in (chunks or []) if (c or "").strip()]
        chunk_len_min = int(min(chunk_lens)) if chunk_lens else 0
        chunk_len_max = int(max(chunk_lens)) if chunk_lens else 0
        chunk_len_avg = (float(sum(chunk_lens)) / float(len(chunk_lens))) if chunk_lens else 0.0

        doi_re = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
        seed_dois: list[str] = []
        try:
            seed_dois = sorted({m.group(0) for m in doi_re.finditer(text or "")})[:25]
        except Exception:
            seed_dois = []

        # External docs stats
        external_docs_by_source: dict[str, int] = {}
        for d in external_docs:
            src = str(d.get("source") or "External")
            external_docs_by_source[src] = external_docs_by_source.get(src, 0) + 1

        external_doc_samples: list[dict[str, Any]] = []
        for d in (external_docs or [])[:5]:
            ext_ids = d.get("externalIds")
            doi = (ext_ids or {}).get("DOI") if isinstance(ext_ids, dict) else d.get("doi")
            txt = str(d.get("abstract") or d.get("text") or "")
            external_doc_samples.append(
                {
                    "title": str(d.get("title") or "") or None,
                    "source": str(d.get("source") or "External"),
                    "doi": doi,
                    "url": d.get("url"),
                    "text_chars": len(txt),
                }
            )

        similarity_samples: list[dict[str, Any]] = []
        for m in (matches_raw or [])[:4]:
            similarity_samples.append(
                {
                    "chunk_sample": str(m.get("text") or "")[:220],
                    "external_title": str(m.get("title") or "") or None,
                    "external_source": str(m.get("source") or ""),
                    "similarity": float(m.get("similarity") or 0.0),
                }
            )

        debug = {
            "extracted_text_chars": len(text),
            "extracted_text_sample": extracted_sample,
            "chunk_count": len(chunks),
            "chunk_samples": chunks[: min(5, len(chunks))],
            "chunk_len_min": chunk_len_min,
            "chunk_len_max": chunk_len_max,
            "chunk_len_avg": float(round(chunk_len_avg + 1e-12, 6)),
            "seed_dois": seed_dois,
            "external_enabled": bool(settings.enable_external_checks),
            "external_queries": queries,
            "external_query_stats": external_query_stats,
            "external_docs_count": len(external_docs),
            "external_docs_by_source": external_docs_by_source,
            "external_doc_samples": external_doc_samples,
            "embedding_provider": str(settings.embed_provider),
            "semantic_threshold": float(getattr(settings, "semantic_threshold", 0.0) or 0.0),
            "external_threshold": float(getattr(settings, "external_semantic_threshold", 0.0) or 0.0),
            "similarity_samples": similarity_samples,
            # Keep existing doc-level debug fields too (helps validate override behavior).
            "doc_title_query": doc_title_query or None,
            "doc_level_match_similarity": float(doc_level_match_sim or 0.0) if doc_level_match is not None else None,
            "doc_level_match_title": str((doc_level_match or {}).get("title") or "") or None,
            "doc_level_strong_title_match": bool(doc_strong_title_match) if doc_level_match is not None else None,
            "doc_level_exact_title_match": bool(doc_exact_title_match) if doc_level_match is not None else None,
            "doc_level_override_threshold": float(doc_override_th),
            "doc_level_override_applied": bool(override_applied) if doc_level_match is not None else None,
            "semantic_chunk_limit": semantic_chunk_limit_applied,
            "internal_reuse_samples": internal_reuse_samples,
        }

    api_matches: list[MatchItem] = []

    # Ensure doc-level evidence is visible to the API whenever doc match exists.
    if doc_level_match is not None and doc_title_query and (doc_exact_title_match or (doc_level_match_sim >= doc_override_th)):
        existing_doc_evidence = False
        for m in matches_raw:
            if (m.get("doi") and doc_level_match.get("doi") and str(m.get("doi")) == str(doc_level_match.get("doi"))):
                existing_doc_evidence = True
                break
            if (m.get("url") and doc_level_match.get("url") and str(m.get("url")) == str(doc_level_match.get("url"))):
                existing_doc_evidence = True
                break
            if (m.get("title") and doc_level_match.get("title") and str(m.get("title")) == str(doc_level_match.get("title"))):
                existing_doc_evidence = True
                break

        if not existing_doc_evidence:
            doc_source = str(doc_level_match.get("source") or "External")
            matches_raw.insert(
                0,
                {
                    "text": f"[Document-level match] {str(doc_level_match.get('title') or doc_title_query or '').strip()}".strip(),
                    "source": f"{doc_source} (doc-level)",
                    "similarity": float(doc_level_match_sim or 0.0),
                    "title": doc_level_match.get("title") or doc_title_query,
                    "authors": doc_level_match.get("authors"),
                    "year": doc_level_match.get("year"),
                    "venue": doc_level_match.get("venue"),
                    "doi": doc_level_match.get("doi"),
                    "url": doc_level_match.get("url"),
                },
            )

    for m in matches_raw:
        source = str(m.get("source") or "")
        url = m.get("url")
        doi = m.get("doi")
        title = m.get("title")
        match_type = "internal" if source == "Document (intra)" else "external"

        api_matches.append(
            MatchItem(
                text=str(m.get("text") or ""),
                source=source,
                similarity=float(m.get("similarity") or 0.0),
                type=match_type,  # type: ignore[arg-type]
                url=url,
                doi=doi,
            )
        )

    report = build_report(
        report_id=report_id,
        originality_score=float(scores.get("originality_score") or 0.0),
        plagiarism_percentage=float(scores.get("plagiarism_percentage") or 0.0),
        exact_match_percentage=int(scores.get("exact_match_percentage") or 0),
        semantic_match_percentage=float(scores.get("semantic_match_percentage") or 0.0),
        citation_coverage_percentage=float(scores.get("citation_coverage_percentage") or 0.0),
        matches=api_matches,
        missing_citations=missing,
        debug=debug,
    )
    out = report.model_dump(exclude_none=True)

    try:
        import json

        cache_fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return out
