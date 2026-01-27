from __future__ import annotations

import zlib
import os
from typing import Any, Dict, List, Optional

from academic_api import fetch_academic_topics
from ranking import score_and_rank
import re

from datasets_loader import (
    build_policy_weight_table_from_datasets,
    match_policy_tags,
    synthetic_topics_from_policies,
)
from topic_kb import Topic
from embedding import cosine_similarity, embed_text
from gemini_polish import polish_topics_inplace


_BAD_TOPIC_TOKENS = {
    "department",
    "dated",
    "ministry",
    "government",
    "orders",
    "order",
    "vide",
    "reference",
    "annexure",
    "chapter",
    "section",
    "wing",
    "goms",
    "ms",
    "no.",
    "dt",
}


_TOPIC_ACTION_TERMS = {
    "analysis",
    "predict",
    "prediction",
    "forecast",
    "forecasting",
    "optimiz",
    "optimization",
    "planning",
    "model",
    "modeling",
    "simulation",
    "assessment",
    "detect",
    "detection",
    "classification",
    "framework",
    "approach",
    "system",
    "design",
    "evaluation",
    "survey",
    "barriers",
    "adoption",
}


def _word_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z][A-Za-z\-]{1,}", (text or "").lower()) if t]


def _looks_like_research_topic(title: str, query: str) -> bool:
    """Heuristic topic-shape validator.

    Filters OCR/policy-fragment titles like "Technology Innovation in Department Dated".
    """

    t = (title or "").strip()
    if not t:
        return False

    words = [w for w in re.split(r"\s+", t) if w]
    if not (6 <= len(words) <= 20):
        return False

    lt = t.lower()
    if any(b in lt for b in _BAD_TOPIC_TOKENS):
        return False

    # Must contain at least one action/research term (or a strong domain term)
    toks = set(_word_tokens(t))
    if not any(any(tok.startswith(a) for a in _TOPIC_ACTION_TERMS) for tok in toks):
        # allow if it strongly matches the query keywords
        qk = {q for q in _word_tokens(query) if len(q) >= 4}
        if not (qk and len(toks.intersection(qk)) >= 2):
            return False

    # Prefer topics that share at least one meaningful keyword with the query
    qk = {q for q in _word_tokens(query) if len(q) >= 4}
    if qk and not toks.intersection(qk):
        # Special-case EV queries: allow EV/electric synonyms
        ql = (query or "").lower()
        if "ev" in ql or "electric" in ql:
            if not ({"ev", "electric", "vehicle", "charging", "battery", "mobility"} & toks):
                return False
        else:
            return False

    return True


def _infer_domain(text: str) -> str:
    t = (text or "").lower()
    toks = set(_word_tokens(t))
    if toks.intersection({"grid", "solar", "battery", "wind", "energy", "pv", "ev", "electric", "charging"}):
        return "Clean Energy"
    if toks.intersection({"crop", "irrig", "agri", "agriculture", "soil", "farm", "yield", "livestock", "farming"}):
        return "AgriTech"
    if toks.intersection({"education", "edtech", "learning", "curriculum", "teacher", "student", "university", "college", "higher", "teaching"}):
        return "EdTech"
    if toks.intersection({"health", "clinical", "hospital", "medical", "diagnosis", "diagnostic", "disease"}):
        return "HealthTech"
    return "Other"


def _topic_id_from_title(title: str, year: int) -> str:
    key = f"{title}|{year}".encode("utf-8", errors="ignore")
    return f"T{zlib.crc32(key) & 0xFFFFFFFF:08x}"


def _guess_year(text: str, default: int = 2022) -> int:
    m = re.search(r"\b(20\d{2})\b", text or "")
    if not m:
        return default
    y = int(m.group(1))
    if 2000 <= y <= 2100:
        return y
    return default


def generate_topics(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = (payload or {}).get("query")
    language = (payload or {}).get("language", "English")
    user_id: Optional[str] = (payload or {}).get("user_id")

    if not query:
        return {"error": "Missing required field: query"}

    # We intentionally do NOT use data/topic_kb.json.
    # Candidate topics come from the academic APIs; policy signals come from Datasets/ PDFs/XLSX.
    api_topics = fetch_academic_topics(query, limit=50)
    if not api_topics:
        # Offline / blocked network fallback: synthesize researchable titles from policy phrases.
        # IMPORTANT: do NOT rank raw policy document titles as "topics" (they are policy signals only).
        synthetic = synthetic_topics_from_policies(max_topics=80)
        if synthetic and len(synthetic) >= 3:
            api_topics = []
            for t in synthetic:
                api_topics.append(
                    {
                        "title": str(t.get("title", "")).strip(),
                        "domain": t.get("domain"),
                        "policy_tags": list(t.get("policy_tags") or []),
                        "citations": int(t.get("citations", 10) or 10),
                        "year": int(t.get("year", 2024) or 2024),
                        "source": "Datasets-synthetic",
                        "_keywords": [str(t.get("intent", "")).strip()] if t.get("intent") else [],
                        "_weight": float(t.get("policy_weight_hint", 1.2) or 1.2),
                    }
                )
        else:
            return {"error": "No candidates returned from academic sources and no synthetic topics could be generated"}

    # Add a few domain anchor topics when the query is clearly agricultural/remote-sensing.
    # This prevents "AI" queries from drifting into EdTech-heavy policy space.
    ql = (query or "").lower()
    if any(k in ql for k in ("crop", "yield", "farm", "farming", "agri", "agriculture", "satellite", "remote sensing", "geospatial", "ndvi")):
        api_topics = list(api_topics)
        api_topics.extend(
            [
                {
                    "title": "AI-Based Crop Yield Prediction Using Satellite Imagery",
                    "domain": "AgriTech",
                    "citations": 50,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["crop yield", "satellite imagery", "remote sensing", "precision agriculture"],
                },
                {
                    "title": "Deep Learning for Precision Agriculture and Crop Monitoring",
                    "domain": "AgriTech",
                    "citations": 40,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["precision agriculture", "crop monitoring", "deep learning"],
                },
                {
                    "title": "Geospatial AI for Crop Health Assessment Using Multispectral Data",
                    "domain": "AgriTech",
                    "citations": 35,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["multispectral", "crop health", "geospatial"],
                },
            ]
        )

    # Add EV / mobility anchors to prevent policy-fragment drift for EV queries.
    if any(k in ql for k in ("ev", "electric vehicle", "electric vehicles", "charging", "charger", "battery", "mobility", "adoption")):
        api_topics = list(api_topics)
        api_topics.extend(
            [
                {
                    "title": "AI-Based Analysis of Barriers to Electric Vehicle Adoption",
                    "domain": "Clean Energy",
                    "citations": 45,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["electric vehicle", "adoption", "barriers", "policy"],
                },
                {
                    "title": "Optimizing EV Charging Infrastructure Using Machine Learning",
                    "domain": "Clean Energy",
                    "citations": 55,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["EV", "charging infrastructure", "optimization", "demand forecasting"],
                },
                {
                    "title": "Policy-Aware Demand Forecasting for Electric Mobility",
                    "domain": "Clean Energy",
                    "citations": 30,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["electric mobility", "demand forecasting", "policy-aware"],
                },
            ]
        )

    qv = embed_text(query)
    # Hard semantic gate: prevents irrelevant candidates from being ranked/returned.
    try:
        sem_gate = float(os.getenv("SEMANTIC_MIN_HARD", "0.40") or 0.40)
    except Exception:
        sem_gate = 0.40

    topics_with_similarity = []
    for item in api_topics:
        title = str(item.get("title", "") or "").strip()
        if not title:
            continue

        year = int(item.get("year", 2022) or 2022)
        citations = int(item.get("citations", 0) or 0)

        domain = str(item.get("domain") or _infer_domain(title))
        policy_tags = list(item.get("policy_tags") or match_policy_tags(title))

        # If the candidate came from datasets, tag it with itself to preserve provenance
        src = str(item.get("source", "") or "")
        if src.lower().startswith("datasets"):
            policy_tags = list(dict.fromkeys([title, *policy_tags]))

        # Extract keywords from title (academic papers don't provide separate keywords)
        # Use meaningful words from title, filtering out common stopwords
        stopwords = {"the", "and", "for", "with", "using", "based", "from", "that", "this", "are", "was", "were", "through", "towards"}
        title_words = [w.strip().lower() for w in re.split(r'[\s:,-]+', title) if len(w.strip()) > 3]
        keywords = [w for w in title_words if w not in stopwords][:8]
        
        # Also check if there are dataset-provided keywords
        raw_keywords = list(item.get("_keywords") or [])
        for kw in raw_keywords:
            # Keep only short, clean keywords (not PDF text blobs)
            if isinstance(kw, str) and 2 <= len(kw) <= 50 and kw.count('\n') == 0 and kw.lower() not in keywords:
                keywords.append(kw.lower())
        
        # Ensure we have at least some keywords
        if not keywords and title:
            keywords = [title.lower()[:30]]
        
        # Limit to top 8 keywords
        keywords = keywords[:8]

        topic = Topic(
            topic_id=_topic_id_from_title(title, year),
            title=title,
            domain=domain,
            keywords=keywords,
            policy_tags=policy_tags,
            citations=citations,
            year=year,
        )

        sim = cosine_similarity(qv, embed_text(title))
        simf = float(sim)

        # If the topic is semantically relevant, drive domain from topic/query text.
        # (Policy domains are for explanations only, not for the topic's primary domain.)
        if simf >= 0.45:
            domain = _infer_domain(f"{title} {' '.join(keywords)} {query}")

        # Rebuild Topic with possibly-updated domain
        topic = Topic(
            topic_id=_topic_id_from_title(title, year),
            title=title,
            domain=domain,
            keywords=keywords,
            policy_tags=policy_tags,
            citations=citations,
            year=year,
        )

        # Filter low-quality / non-topic-shaped titles early.
        if not _looks_like_research_topic(title, query):
            continue

        # Enforce semantic gate, but allow seeds to pass if they are topic-shaped.
        src = str(item.get("source", "") or "")
        if src != "Seed" and simf < sem_gate:
            continue

        topics_with_similarity.append((topic, simf))

    topics_with_similarity.sort(key=lambda x: x[1], reverse=True)
    topics_with_similarity = topics_with_similarity[:50]

    if not topics_with_similarity:
        return {"error": "No valid candidate topics after filtering (semantic/topic-shape gate)"}

    ranked = score_and_rank(query=query, topics_with_similarity=topics_with_similarity, user_id=user_id, top_k=10)

    # Optional post-processing: Gemini language polishing (read-only)
    # This must NOT change the selected topics, ranking, or scores.
    try:
        polish_topics_inplace(ranked.get("recommended_topics") or [])
    except Exception:
        pass

    topics_simple = [
        {"title": t["title"], "score": t["final_score"]}
        for t in (ranked.get("recommended_topics") or [])
    ]

    return {
        "query": query,
        "language": language,
        "user_id": user_id,
        **ranked,
        # Spec-friendly alias
        "topics": topics_simple,
    }

