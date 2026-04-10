from __future__ import annotations

import json
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

import requests


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


_CYBER_SYNONYMS = {
    "security",
    "secure",
    "cyber",
    "malware",
    "phishing",
    "ransomware",
    "intrusion",
    "attacks",
    "attack",
    "threat",
    "threats",
    "adversarial",
    "botnet",
    "forensics",
    "anomaly",
    "anomalies",
    "detection",
    "ids",
    "ddos",
    "spoofing",
    "fraud",
    "cybersecurity",
    "infosec",
}


def _normalize_query(query: str) -> str:
    """Normalize noun-phrase queries into more research-style phrasing.

    This helps candidate retrieval + heuristic filters that rely on action/intent terms.
    """

    q = (query or "").strip()
    if not q:
        return q

    # Normalize common hyphen variants to ASCII '-' for consistent matching.
    q = (
        q.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )

    # Common surface-form rewrites.
    q = re.sub(r"\bai\s*[-]?based\b", "AI for", q, flags=re.IGNORECASE)
    q = re.sub(r"\bml\s*[-]?based\b", "Machine learning for", q, flags=re.IGNORECASE)
    q = re.sub(r"\bsystems\b", "applications", q, flags=re.IGNORECASE)

    # Expand common abbreviations when they clearly refer to a domain.
    # IMPORTANT: Only expand uppercase 'IT' (or 'I.T.') in sector/industry contexts
    # to avoid rewriting the pronoun "it".
    if re.search(r"\bI\.T\.\b", q):
        q = re.sub(r"\bI\.T\.\b", "information technology", q)
    if re.search(r"\bIT\b", q) and re.search(r"\b(sector|industry|services|companies|firms|organizations|organisation)\b", q, flags=re.IGNORECASE):
        q = re.sub(r"\bIT\b", "information technology", q)

    # Normalize common phrasing.
    q = re.sub(r"\btech\s+sector\b", "technology sector", q, flags=re.IGNORECASE)
    q = re.sub(r"\bit\s+sector\b", "information technology sector", q, flags=re.IGNORECASE)

    return q.strip()


_IT_SECTOR_MARKERS = (
    "information technology",
    "it sector",
    "it industry",
    "technology sector",
    "tech sector",
    "it services",
    "enterprise",
    "software industry",
    "software services",
    "digital transformation",
)


def _it_sector_topics(query: str) -> List[str]:
    # Query-aware anchors to keep AI+IT-sector queries on-domain even when
    # semantic similarity is low (e.g., hashing embeddings).
    _ = query
    return [
        "Impact of Artificial Intelligence on Productivity and Employment in the IT Sector",
        "AI Adoption in IT Services: Drivers, Barriers, and Organizational Readiness",
        "AI-Driven Automation in Software Development Lifecycles: Opportunities and Risks",
        "AIOps in Enterprise IT: Incident Prediction, Root Cause Analysis, and Service Reliability",
        "Governance, Ethics, and Compliance Frameworks for Enterprise AI in IT Organizations",
        "Measuring Business Value of AI in IT Organizations: KPIs, ROI, and Operational Impact",
        "AI for Cybersecurity in Enterprise IT Environments: Threat Detection and Response",
    ]


_TOURISM_MARKERS = (
    "tourism",
    "travel",
    "hospitality",
    "heritage",
    "eco tourism",
    "eco-tourism",
    "ecotourism",
)


_EDUCATION_MARKERS = (
    "education",
    "school",
    "schools",
    "schooling",
    "teacher",
    "teachers",
    "student",
    "students",
    "classroom",
    "curriculum",
    "higher education",
    "university",
    "college",
    "literacy",
)


_AI_MARKERS = (
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "nlp",
    "computer vision",
)


def _detect_domain(query: str) -> str:
    """Detect coarse domain for domain-specific fallback generation.

    This is intentionally simple: it only needs to prevent non-AI domains from
    being forced into AI/ML templates.
    """

    q = f" {(query or '').lower()} "

    # If the user explicitly asks for AI/ML, honor that.
    if any(m in q for m in _AI_MARKERS):
        return "ai"

    if any(m in q for m in _TOURISM_MARKERS):
        return "tourism"

    if any(m in q for m in _EDUCATION_MARKERS):
        return "education"

    return "general"


def _education_location(query: str) -> Optional[str]:
    ql = f" {(query or '').lower()} "
    if " india " in ql:
        return "India"
    return None


def _education_topics(query: str) -> List[str]:
    loc = _education_location(query)
    suffix = f" in {loc}" if loc else ""
    return [
        f"School infrastructure development and learning outcomes{suffix}",
        f"Government policies for improving school education quality{suffix}",
        f"Teacher training and capacity building for school development{suffix}",
        f"Digital education adoption and implementation challenges in schools{suffix}",
        f"Equity and access in school education: rural-urban and gender gaps{suffix}",
    ]


def _tourism_location(query: str) -> Optional[str]:
    ql = f" {(query or '').lower()} "

    # Common, explicit cases we see in user queries.
    if " india " in ql:
        return "India"
    if " andhra pradesh " in ql or " ap " in ql:
        return "Andhra Pradesh"

    # Try to parse "tourism in <location>".
    m = re.search(r"\btourism\s+in\s+([A-Za-z][A-Za-z\s\-&,]{2,})\b", query or "", flags=re.IGNORECASE)
    if m:
        loc = (m.group(1) or "").strip(" ,.-")
        return loc.title() if loc else None

    return None


def _tourism_topics(query: str) -> List[str]:
    loc = _tourism_location(query)
    suffix = f" in {loc}" if loc else ""
    return [
        f"Impact of tourism on economic development{suffix}",
        f"Sustainable tourism practices{suffix}",
        f"Role of government policies in promoting tourism{suffix}",
        f"Cultural tourism and heritage management{suffix}",
        f"Challenges and opportunities in eco-tourism{suffix}",
    ]


def _ai_topics(_: str) -> List[str]:
    # NOTE: This is a last-resort offline fallback for AI/ML queries.
    return [
        "Artificial Intelligence Adoption in Industry: Challenges and Opportunities",
        "AI Governance and Responsible AI in Organizations: Frameworks and Best Practices",
        "AI-Driven Automation and Workforce Transformation: Impact Assessment",
        "AIOps for IT Operations: Predictive Monitoring and Incident Management",
        "AI in Cybersecurity Threat Detection and Response",
    ]


def _generate_basic_topics(query: str) -> List[str]:
    q = (query or "").strip()
    if not q:
        return []

    domain = _detect_domain(q)
    if domain == "tourism":
        return _tourism_topics(q)
    if domain == "ai":
        return _ai_topics(q)
    if domain == "education":
        return _education_topics(q)

    # General (non-AI) research-ready templates.
    return [
        f"Challenges and opportunities in {q}",
        f"Policy and governance issues in {q}",
        f"Socio-economic impacts of {q}",
        f"Sustainability and environmental considerations in {q}",
        f"Future directions and research gaps in {q}",
    ]


def _detect_intent(query: str) -> str:
    """Coarse intent detection.

    This engine primarily returns research-style topics (paper titles). For user queries
    asking for project ideas, return practical project topics instead of research templates.
    """

    q = f" {(query or '').lower()} "
    project_markers = (
        " project ",
        " projects ",
        " project idea ",
        " project ideas ",
        " mini project ",
        " capstone ",
        " final year ",
        " assignment ",
        " demo ",
        " prototype ",
        " for students ",
        " student project ",
        " students ",
        " ideas ",
    )
    return "project" if any(m in q for m in project_markers) else "research"


def _project_level(query: str) -> str:
    q = f" {(query or '').lower()} "
    if any(m in q for m in (" easy ", " beginner ", " simple ", " basic ", " starter ")):
        return "beginner"
    return "general"


def _generate_project_topics(query: str) -> List[str]:
    """Generate practical, implementable project topics.

    This intentionally prefers concrete, buildable ideas over paper-title templates.
    """

    ql = (query or "").lower()
    level = _project_level(query)

    def _dedupe_keep_order(items: List[str]) -> List[str]:
        return list(dict.fromkeys([i for i in items if (i or "").strip()]))

    ml_markers = (
        "machine learning",
        "ml ",
        " ml",
        "deep learning",
        "neural network",
        "cnn",
        "computer vision",
        "nlp",
    )

    quantum_markers = (
        "quantum",
        "qubit",
        "qiskit",
        "quantum computing",
        "quantum-computing",
        "quantum circuit",
        "quantum-circuit",
        "vqe",
        "qaoa",
    )

    edu_markers = ("student", "assessment", "grading", "exam", "quiz", "education", "edtech", "learning")

    # Quantum computing project ideas.
    if any(m in ql for m in quantum_markers):
        beginner = [
            "Simulation of Quantum Computing Circuits Using Qiskit",
            "Quantum Computing Algorithms for Grover Search Implementation",
            "Quantum Computing Algorithms for Shor Factorization Simulation",
            "Quantum Circuit Optimization for Reduced Gate Count",
            "Benchmarking Quantum Computing Noise Models Using Simulation",
            "Quantum Computing Error Mitigation Techniques Evaluation",
            "Variational Quantum Eigensolver for Quantum Computing Chemistry Problems",
            "Quantum Approximate Optimization Algorithm for Quantum Computing Scheduling",
        ]
        general = [
            "Deep Learning for Quantum Circuit Optimization and Compilation",
            "Machine Learning Approaches for Quantum Computing Error Mitigation",
            "Quantum Computing Noise Characterization and Anomaly Detection",
            "Hybrid Quantum Computing and Classical Optimization Framework Design",
            "Quantum Computing Hardware-Aware Circuit Mapping and Scheduling",
            "Evaluation of Quantum Computing Error Correction Codes",
            "Quantum Computing Reinforcement Learning for Circuit Synthesis",
            "Survey of Quantum Computing Applications in Machine Learning",
        ]

        topics = beginner if level == "beginner" else general
        return _dedupe_keep_order(topics)[:10]

    # If the user explicitly asks for ML projects (even if they mention students),
    # prioritize broad, practical ML ideas across NLP/CV/recsys + include a couple
    # student-focused ML ideas.
    if any(m in ql for m in ml_markers):
        base_beginner = [
            "Spam Email Detection Using Machine Learning Classification",
            "Movie Recommendation System Using Machine Learning Collaborative Filtering",
            "Fake News Detection Using Machine Learning and NLP Classification",
            "Handwritten Digit Recognition Using Deep Learning Classification",
            "Credit Card Fraud Detection Using Machine Learning Classification",
            "Customer Churn Prediction Using Machine Learning Models",
            "House Price Prediction Using Machine Learning Regression",
            "Image Classification Using Deep Learning Convolutional Neural Networks",
        ]
        base_general = [
            "Sentiment Analysis Using Machine Learning and NLP Classification",
            "Face Recognition Attendance System Using Deep Learning Models",
            "Resume Screening and Skill Extraction Using Machine Learning and NLP",
            "Network Intrusion Detection Using Machine Learning Anomaly Detection",
            "Traffic Sign Recognition Using Deep Learning Convolutional Neural Networks",
            "Topic Modeling and Document Clustering Using Machine Learning",
            "Recommendation System for E-Learning Platforms Using Machine Learning",
            "Chatbot for Student Support Using NLP and Machine Learning",
        ]

        student_specific = []
        if any(k in ql for k in edu_markers):
            student_specific = [
                "Student Performance Prediction Using Machine Learning Models",
                "Automated Short-Answer Grading Using NLP Classification",
                "Plagiarism Detection for Student Submissions Using Text Similarity",
                "Sentiment Analysis of Student Feedback Using NLP",
            ]

        # If the query is student/education oriented, lead with student-specific ideas.
        core = base_beginner if level == "beginner" else base_general
        topics = (student_specific + core) if any(k in ql for k in edu_markers) else (core + student_specific)
        return _dedupe_keep_order(topics)[:10]

    # Education / assessment focused projects.
    if any(k in ql for k in edu_markers):
        if level == "beginner":
            return [
                "Student Performance Prediction Using Machine Learning",
                "Automated Quiz Recommendation Based on Student Learning Gaps",
                "Sentiment Analysis of Student Feedback Using NLP",
                "Plagiarism Detection for Student Submissions Using Text Similarity",
                "Attendance Tracking and Analytics Dashboard for Classrooms",
            ]
        return [
            "Adaptive Learning Recommendation System for Personalized Study Plans",
            "AI-Based Automated Short-Answer Grading Using NLP",
            "Early Warning System for Student Dropout Risk Prediction",
            "Question Difficulty Estimation and Adaptive Quiz Generation",
            "Proctoring Anomaly Detection Using Computer Vision",
        ]

    # Generic ML/AI project ideas.
    if level == "beginner":
        return [
            "Spam Email Detection Using Machine Learning",
            "Movie Recommendation System Using Collaborative Filtering",
            "Fake News Detection Using NLP",
            "Handwritten Digit Recognition Using a Simple Neural Network",
            "Credit Card Fraud Detection Using Supervised Learning",
        ]

    return [
        "Real-Time Sentiment Analysis for Social Media Streams",
        "Face Recognition Attendance System Using Deep Learning",
        "Resume Screening and Skill Extraction Using NLP",
        "Network Intrusion Detection Using Machine Learning",
        "Image-Based Plant Disease Detection Using Computer Vision",
    ]


def _filter_project_topics(topics: List[str], query: str) -> List[str]:
    """Filter project-topic suggestions for relevance to the query.

    Goal: keep a diverse set of practical ML/AI ideas, but avoid obviously unrelated
    domains (e.g., energy) unless the query mentions them.
    """

    ql = f" {(query or '').lower()} "
    allow_energy = any(k in ql for k in ("energy", "solar", "wind", "grid", "ev", "battery", "charging"))
    allow_agri = any(k in ql for k in ("agri", "agriculture", "crop", "farm", "irrig", "soil"))

    filtered: List[str] = []
    for t in topics or []:
        tl = f" {(t or '').lower()} "

        # Drop energy-specific topics unless the query is energy-related.
        if not allow_energy and any(k in tl for k in (" energy ", " solar ", " wind ", " grid ", " ev ", " battery ", " charging ")):
            continue

        # Drop agriculture-specific topics unless the query is agriculture-related.
        if not allow_agri and any(k in tl for k in (" crop ", " farm ", " farming ", " agriculture ", " irrigation ", " plant disease ", " soil ")):
            continue

        filtered.append(t)

    return list(dict.fromkeys([x for x in filtered if (x or '').strip()]))


def _word_tokens(text: str) -> List[str]:
    toks = [t for t in re.findall(r"[A-Za-z][A-Za-z\-]{1,}", (text or "").lower()) if t]
    expanded: List[str] = []
    for t in toks:
        expanded.append(t)

        # Split hyphenated compounds (e.g., cyber-security -> cyber, security).
        if "-" in t:
            parts = [p for p in t.split("-") if p]
            expanded.extend(parts)
            expanded.append(t.replace("-", ""))

        # Common concatenations in paper titles.
        if t == "cybersecurity":
            expanded.extend(["cyber", "security"])
        if t == "infosec":
            expanded.extend(["information", "security"])

    # Deduplicate but keep deterministic order.
    return list(dict.fromkeys(expanded))


def _looks_like_research_topic(title: str, query: str) -> bool:
    """Heuristic topic-shape validator.

    Filters OCR/policy-fragment titles like "Technology Innovation in Department Dated".
    """

    t = (title or "").strip()
    if not t:
        return False

    words = [w for w in re.split(r"\s+", t) if w]
    # Academic paper titles are often short; keep this permissive.
    if not (4 <= len(words) <= 24):
        return False

    lt = t.lower()
    if any(b in lt for b in _BAD_TOPIC_TOKENS):
        return False

    # Must contain at least one action/research term (or a strong query match)
    toks = set(_word_tokens(t))
    if not any(any(tok.startswith(a) for a in _TOPIC_ACTION_TERMS) for tok in toks):
        # allow if it strongly matches the query keywords
        qk = {q for q in _word_tokens(query) if len(q) >= 4}
        # If the query has only 1-2 meaningful tokens (e.g., "quantum computing"),
        # requiring 2 overlaps rejects many legitimate titles.
        required = 2 if len(qk) >= 3 else 1
        if not (qk and len(toks.intersection(qk)) >= required):
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
            # Cybersecurity queries often use different surface forms than paper titles
            # (e.g., "intrusion detection" vs "cyber security"). Allow common synonyms.
            ql = (query or "").lower()
            if "cyber" in ql or "security" in ql:
                if not (toks & _CYBER_SYNONYMS):
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
    if toks.intersection({"marine", "maritime", "coastal", "coast", "ocean", "sea", "port", "harbor", "harbour", "shipping", "ship", "fisheries", "fishery", "aquaculture", "mangrove", "estuaries", "estuary"}):
        return "Marine"
    if toks.intersection({"tourism", "travel", "hospitality", "heritage", "ecotourism", "eco-tourism", "ecotour"}):
        return "Tourism"
    # NOTE: Do not treat generic "machine learning" as an education signal.
    if toks.intersection({"education", "edtech", "curriculum", "teacher", "student", "university", "college", "higher", "teaching", "classroom", "assessment", "grading", "exam", "quiz"}):
        return "EdTech"
    if toks.intersection({"health", "clinical", "hospital", "medical", "diagnosis", "diagnostic", "disease"}):
        return "HealthTech"
    return "Other"


_LOCATION_TOKENS = {
    # Common location-only tokens that should not count as topical relevance.
    "andhra",
    "pradesh",
    "india",
    "indian",
    "state",
    "district",
    "region",
    "coastal",
}


_GENERIC_QUERY_TOKENS = {
    # Words that commonly cause cross-domain drift when used alone.
    "smart",
    "system",
    "systems",
    "framework",
    "model",
    "models",
    "technology",
    "technologies",
    "innovation",
    "integrated",
    "integration",
    "implementation",
    "development",
    "approach",
    "application",
    "applications",
}


def _query_core_tokens(query: str) -> List[str]:
    toks = [t for t in _word_tokens(query) if t and len(t) >= 4]
    ql = f" {(query or '').lower()} "
    # Keep 'technology' when the user is explicitly talking about "information technology".
    has_infotech = " information technology " in ql
    out: List[str] = []
    for t in toks:
        if t in _LOCATION_TOKENS:
            continue
        if t in _GENERIC_QUERY_TOKENS:
            if has_infotech and t == "technology":
                out.append(t)
                continue
            continue
        out.append(t)
    # Deduplicate, keep order.
    return list(dict.fromkeys(out))


def _expand_queries(query: str, domain_hint: str) -> List[str]:
    q = (query or "").strip()
    if not q:
        return []

    enabled = (os.getenv("QUERY_EXPANSION_ENABLED", "true") or "true").strip().lower() in {"1", "true", "yes", "y"}
    if not enabled:
        return [q]

    variants: List[str] = [q]
    if domain_hint == "education":
        variants.extend([f"{q} edtech", f"{q} digital learning schools"])
    elif domain_hint == "tourism":
        variants.append(f"{q} sustainable tourism")
    elif domain_hint == "ai":
        ql = f" {(q or '').lower()} "
        # If the query is about the IT sector / enterprise, add a couple targeted variants
        # to improve retrieval coverage from academic APIs.
        if any(m in ql for m in _IT_SECTOR_MARKERS):
            variants.extend([f"{q} enterprise", f"{q} IT services", f"{q} digital transformation"])

    # Deduplicate + cap to keep API calls bounded.
    uniq: List[str] = []
    seen: set[str] = set()
    for v in variants:
        vv = " ".join((v or "").split())
        key = vv.lower()
        if not vv or key in seen:
            continue
        seen.add(key)
        uniq.append(vv)

    try:
        cap = int(os.getenv("QUERY_EXPANSION_MAX", "3") or 3)
    except Exception:
        cap = 3
    return uniq[: max(1, min(cap, 5))]


def _groq_summarize(query: str, language: str, recommended_topics: List[Dict[str, Any]]) -> Optional[str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    model = (os.getenv("GROQ_MODEL", "llama-3.1-8b-instant") or "llama-3.1-8b-instant").strip()
    try:
        max_items = int(os.getenv("GROQ_SUMMARY_MAX_TOPICS", "5") or 5)
    except Exception:
        max_items = 5

    items: List[Dict[str, Any]] = []
    for t in (recommended_topics or [])[: max(1, max_items)]:
        if not isinstance(t, dict):
            continue
        items.append(
            {
                "title": t.get("title"),
                "domain": t.get("domain"),
                "final_score_100": t.get("final_score_100"),
                "reasons": (t.get("reasons") or [])[:5],
            }
        )

    prompt = {"query": query, "language": language, "recommended_topics": items}

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0.2,
                "max_tokens": 250,
                "messages": [
                    {
                        "role": "system",
                        "content": "Summarize the recommended topics concisely. Emphasize relevance to the query and flag any off-domain topic in one short clause.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Summarize this JSON in 3-6 sentences. Keep it practical.\n\n"
                            + json.dumps(prompt, ensure_ascii=False)
                        ),
                    },
                ],
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        content = (((data.get("choices") or [])[0] or {}).get("message") or {}).get("content")
        return (str(content).strip() if content else None) or None
    except Exception:
        return None


def _groq_generate_topics(query: str, language: str, domain_hint: str, *, n: int = 10) -> List[Dict[str, Any]]:
    """Generate research-style topics via Groq when all other candidate generation fails."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return []

    model = (os.getenv("GROQ_MODEL", "llama-3.1-8b-instant") or "llama-3.1-8b-instant").strip()
    try:
        n = int(n)
    except Exception:
        n = 10
    n = max(5, min(n, 12))

    system = (
        "You generate research paper topic titles. Output ONLY valid JSON."
    )

    user = {
        "query": query,
        "language": language,
        "domain_hint": domain_hint,
        "requirements": {
            "count": n,
            "title_style": "research paper / thesis topic",
            "avoid": ["smart grid", "renewable energy", "solar", "wind"] if domain_hint == "education" else [],
            "output_format": {
                "type": "array",
                "items": {"title": "string", "domain": "string", "keywords": "array[string]"},
            },
        },
    }

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0.3,
                "max_tokens": 700,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            "Generate topic titles as JSON only. Return an array with exactly 'count' items. "
                            "No markdown, no extra keys.\n\n" + json.dumps(user, ensure_ascii=False)
                        ),
                    },
                ],
            },
            timeout=25,
        )
        r.raise_for_status()
        data = r.json()
        content = (((data.get("choices") or [])[0] or {}).get("message") or {}).get("content")
        raw = (str(content).strip() if content else "").strip()
        if not raw:
            return []

        # Try to parse JSON strictly; if the model wrapped it, attempt to extract the first JSON array.
        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"\[[\s\S]*\]", raw)
            if not m:
                return []
            parsed = json.loads(m.group(0))

        if not isinstance(parsed, list):
            return []

        out: List[Dict[str, Any]] = []
        for it in parsed:
            if not isinstance(it, dict):
                continue
            title = str(it.get("title") or "").strip()
            if not title:
                continue
            domain = str(it.get("domain") or "Other").strip() or "Other"
            kws = it.get("keywords") or []
            if not isinstance(kws, list):
                kws = []
            keywords = [str(x).strip().lower() for x in kws if str(x).strip()][:8]
            out.append({"title": title, "domain": domain, "keywords": keywords})

        return out[:n]
    except Exception:
        return []


def _groq_polish_titles(query: str, language: str, titles: List[str]) -> List[str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return titles

    enabled = (os.getenv("GROQ_POLISH_ENABLED", "true") or "true").strip().lower() in {"1", "true", "yes", "y"}
    if not enabled:
        return titles

    model = (os.getenv("GROQ_MODEL", "llama-3.1-8b-instant") or "llama-3.1-8b-instant").strip()
    payload = {"query": query, "language": language, "titles": titles}

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0.2,
                "max_tokens": 600,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You rewrite research topic titles to better match the user's query. "
                            "Output ONLY JSON: an array of strings with the same length and order as input. "
                            "Keep titles concise, academic, and on-domain."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Rewrite these titles (same count/order) as JSON only:\n\n" + json.dumps(payload, ensure_ascii=False),
                    },
                ],
            },
            timeout=25,
        )
        r.raise_for_status()
        data = r.json()
        content = (((data.get("choices") or [])[0] or {}).get("message") or {}).get("content")
        raw = (str(content).strip() if content else "").strip()
        if not raw:
            return titles

        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"\[[\s\S]*\]", raw)
            if not m:
                return titles
            parsed = json.loads(m.group(0))

        if not isinstance(parsed, list) or len(parsed) != len(titles):
            return titles

        out: List[str] = []
        for i, it in enumerate(parsed):
            s = str(it or "").strip()
            out.append(s if s else titles[i])
        return out
    except Exception:
        return titles


_MARINE_QUERY_MARKERS = (
    "marine",
    "maritime",
    "coastal",
    "coast",
    "ocean",
    "sea",
    "port",
    "harbor",
    "harbour",
    "shipping",
    "fisheries",
    "fishery",
    "aquaculture",
    "blue economy",
    "mangrove",
)


def _synthetic_title_relevant_to_query(title: str, query: str) -> bool:
    """Heuristic relevance filter for offline synthetic topics.

    Purpose: avoid returning generic policy-derived titles that only match the
    location words in the query (e.g., "Andhra Pradesh").
    """

    if not (title or "").strip() or not (query or "").strip():
        return False

    ql = f" {(query or '').lower()} "
    tl = f" {(title or '').lower()} "

    qt = set(_word_tokens(query))
    tt = set(_word_tokens(title))
    overlap = qt.intersection(tt)
    if not overlap:
        return False
    if overlap.issubset(_LOCATION_TOKENS):
        return False

    # If the query is marine/coastal oriented, require marine signal in the title.
    if any(m in ql for m in _MARINE_QUERY_MARKERS):
        marine_tokens = {
            "marine",
            "maritime",
            "coastal",
            "coast",
            "ocean",
            "sea",
            "port",
            "harbor",
            "harbour",
            "shipping",
            "ship",
            "fisheries",
            "fishery",
            "aquaculture",
            "mangrove",
            "estuary",
            "estuaries",
        }
        return bool(tt.intersection(marine_tokens))

    # Otherwise, accept as long as there is at least one non-location overlap.
    return True


def _merge_topic_dicts(primary: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two topic dictionaries (same title/topic_id) into one."""

    def _merge_list(a: Any, b: Any) -> List[Any]:
        out: List[Any] = []
        for src in (a or [], b or []):
            if isinstance(src, (list, tuple)):
                for x in src:
                    if x not in out:
                        out.append(x)
        return out

    merged = dict(primary)

    merged["keywords"] = _merge_list(primary.get("keywords"), other.get("keywords"))
    merged["policy_tags"] = _merge_list(primary.get("policy_tags"), other.get("policy_tags"))
    merged["reasons"] = _merge_list(primary.get("reasons"), other.get("reasons"))

    # Merge policy_meta fields when present.
    pm_a = primary.get("policy_meta") if isinstance(primary.get("policy_meta"), dict) else {}
    pm_b = other.get("policy_meta") if isinstance(other.get("policy_meta"), dict) else {}
    policy_meta: Dict[str, Any] = dict(pm_a)
    for k in set(pm_a.keys()).union(pm_b.keys()):
        if isinstance(pm_a.get(k), list) or isinstance(pm_b.get(k), list):
            policy_meta[k] = _merge_list(pm_a.get(k), pm_b.get(k))
        else:
            policy_meta[k] = pm_a.get(k, pm_b.get(k))
    if policy_meta:
        merged["policy_meta"] = policy_meta

    # Prefer the variant with the higher final_score.
    try:
        fs_a = float(primary.get("final_score", 0.0) or 0.0)
    except Exception:
        fs_a = 0.0
    try:
        fs_b = float(other.get("final_score", 0.0) or 0.0)
    except Exception:
        fs_b = 0.0

    if fs_b > fs_a:
        # Keep the higher-score record as the base, but preserve merged lists/meta.
        merged = dict(other)
        merged["keywords"] = _merge_list(primary.get("keywords"), other.get("keywords"))
        merged["policy_tags"] = _merge_list(primary.get("policy_tags"), other.get("policy_tags"))
        merged["reasons"] = _merge_list(primary.get("reasons"), other.get("reasons"))
        if policy_meta:
            merged["policy_meta"] = policy_meta

    # Ensure score_100 matches the kept final_score.
    try:
        merged["final_score_100"] = round(float(merged.get("final_score", 0.0) or 0.0) * 100.0, 2)
    except Exception:
        pass

    return merged


def _dedupe_merge_recommended_topics(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate recommended topics by topic_id/title while merging metadata."""

    merged_by_key: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for it in items or []:
        if not isinstance(it, dict):
            continue
        topic_id = str(it.get("topic_id") or "").strip()
        title = str(it.get("title") or "").strip()
        key = topic_id or title.lower()
        if not key:
            continue
        if key not in merged_by_key:
            merged_by_key[key] = it
            order.append(key)
        else:
            merged_by_key[key] = _merge_topic_dicts(merged_by_key[key], it)

    out = [merged_by_key[k] for k in order if k in merged_by_key]
    # Keep stable ordering, but ensure highest scores float up naturally.
    try:
        out.sort(key=lambda x: float(x.get("final_score", 0.0) or 0.0), reverse=True)
    except Exception:
        pass
    return out


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

    # Normalize to improve retrieval/filters for noun-phrase inputs.
    # Keep the original query for the response.
    query_norm = _normalize_query(str(query))

    intent = _detect_intent(str(query))

    # Project-mode: provide practical project ideas rather than paper-title templates.
    # We still pass them through the same ranking output format for consistency.
    if intent == "project":
        used_fallback = True
        api_topics = []
        project_titles = _filter_project_topics(_generate_project_topics(str(query)), str(query))
        for t in project_titles:
            api_topics.append(
                {
                    "title": t,
                    "domain": _infer_domain(f"{t} {str(query)}"),
                    "citations": 10,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": [],
                }
            )
    else:
        used_fallback = False

    # We intentionally do NOT use data/topic_kb.json.
    # Candidate topics come from the academic APIs; policy signals come from Datasets/ PDFs/XLSX.
    if intent != "project":
        domain_hint = _detect_domain(query_norm or str(query))
        q_variants = _expand_queries(query_norm or str(query), domain_hint)
        api_topics = []
        seen_titles: set[str] = set()
        for qx in q_variants:
            for it in fetch_academic_topics(qx, limit=50) or []:
                if not isinstance(it, dict):
                    continue
                title = str(it.get("title") or "").strip()
                if not title:
                    continue
                k = title.lower()
                if k in seen_titles:
                    continue
                seen_titles.add(k)
                api_topics.append(it)
    if not api_topics:
        # Offline / blocked network fallback: synthesize researchable titles from policy phrases.
        # IMPORTANT: do NOT rank raw policy document titles as "topics" (they are policy signals only).
        synthetic = synthetic_topics_from_policies(max_topics=80)
        if synthetic and len(synthetic) >= 3:
            api_topics = []
            for t in synthetic:
                title = str(t.get("title", "")).strip()
                if not title:
                    continue
                if not _synthetic_title_relevant_to_query(title, query_norm or str(query)):
                    continue
                api_topics.append(
                    {
                        "title": title,
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
    ql = (query_norm or str(query) or "").lower()
    q_tokens = set(_word_tokens(ql))

    # Add tourism anchors for tourism/travel queries (avoids AI-biased fallback titles).
    if any(k in ql for k in _TOURISM_MARKERS):
        api_topics = list(api_topics)
        for t in _tourism_topics(str(query)):
            api_topics.append(
                {
                    "title": t,
                    "domain": "Tourism",
                    "citations": 35,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["tourism", "policy", "sustainability", "heritage"],
                }
            )

    # Add education/schools anchors for school-development queries.
    if any(k in ql for k in _EDUCATION_MARKERS):
        api_topics = list(api_topics)
        for t in _education_topics(str(query)):
            api_topics.append(
                {
                    "title": t,
                    "domain": "EdTech",
                    "citations": 35,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["education", "schools", "policy", "development"],
                }
            )
    if any(k in ql for k in ("crop", "yield", "farm", "farming", "agri", "agriculture", "irrig", "irrigation", "soil moisture", "water", "satellite", "remote sensing", "geospatial", "ndvi")):
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
                {
                    "title": "AI-Based Irrigation Scheduling Using Soil Moisture and Weather Forecasting",
                    "domain": "AgriTech",
                    "citations": 45,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["irrigation scheduling", "soil moisture", "weather forecasting"],
                },
                {
                    "title": "Machine Learning for Smart Irrigation and Water Use Efficiency in Agriculture",
                    "domain": "AgriTech",
                    "citations": 40,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["smart irrigation", "water use efficiency", "machine learning"],
                },
            ]
        )

    # Add healthcare anchors for common "AI in healthcare" queries.
    if any(k in ql for k in ("health", "healthcare", "medical", "clinical", "hospital", "diagnosis", "diagnostic", "disease", "patient", "radiology", "ehr", "electronic health", "medical imaging")):
        api_topics = list(api_topics)
        api_topics.extend(
            [
                {
                    "title": "Artificial Intelligence in Healthcare: A Systematic Review of Methods and Applications",
                    "domain": "HealthTech",
                    "citations": 90,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["healthcare", "systematic review", "AI"],
                },
                {
                    "title": "Machine Learning for Clinical Decision Support Using Electronic Health Records",
                    "domain": "HealthTech",
                    "citations": 85,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["clinical decision support", "EHR", "machine learning"],
                },
                {
                    "title": "Deep Learning for Medical Imaging Diagnosis and Prognosis",
                    "domain": "HealthTech",
                    "citations": 95,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["medical imaging", "diagnosis", "deep learning"],
                },
                {
                    "title": "Privacy-Preserving AI for Healthcare Data Sharing and Federated Learning",
                    "domain": "HealthTech",
                    "citations": 70,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["privacy", "federated learning", "healthcare"],
                },
            ]
        )

    # Add marine/coastal anchors for "marine development" queries.
    if any(k in ql for k in ("marine", "maritime", "coastal", "coast", "ocean", "sea", "port", "harbor", "harbour", "shipping", "fisheries", "fishery", "aquaculture", "blue economy", "mangrove")):
        api_topics = list(api_topics)
        api_topics.extend(
            [
                {
                    "title": "Marine Spatial Planning and Coastal Zone Management in Andhra Pradesh",
                    "domain": "Marine",
                    "citations": 40,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["marine spatial planning", "coastal zone", "andhra pradesh"],
                },
                {
                    "title": "Sustainable Fisheries Management Using Data-Driven Stock Assessment and Forecasting",
                    "domain": "Marine",
                    "citations": 55,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["fisheries", "stock assessment", "forecasting"],
                },
                {
                    "title": "Aquaculture Disease Detection Using Computer Vision and Water Quality Monitoring",
                    "domain": "Marine",
                    "citations": 45,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["aquaculture", "disease detection", "water quality"],
                },
                {
                    "title": "Port Infrastructure Development and Maritime Logistics Optimization Using AI",
                    "domain": "Marine",
                    "citations": 35,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["port development", "maritime logistics", "optimization"],
                },
                {
                    "title": "Coastal Erosion Monitoring and Shoreline Change Detection Using Satellite Imagery",
                    "domain": "Marine",
                    "citations": 60,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["coastal erosion", "shoreline", "satellite imagery"],
                },
                {
                    "title": "Mangrove Ecosystem Monitoring and Conservation Planning Using Remote Sensing",
                    "domain": "Marine",
                    "citations": 50,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["mangroves", "remote sensing", "conservation"],
                },
            ]
        )

    # Add EV / mobility anchors to prevent policy-fragment drift for EV queries.
    # IMPORTANT: do not match substring "ev" inside unrelated words like "development".
    is_ev_query = (
        ("ev" in q_tokens)
        or ("electric" in q_tokens and "vehicle" in q_tokens)
        or ("electric" in q_tokens and "vehicles" in q_tokens)
        or any(k in ql for k in ("electric vehicle", "electric vehicles"))
        or any(t in q_tokens for t in ("charging", "charger", "battery", "mobility"))
    )
    if is_ev_query:
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

    # Add quantum-computing anchors to keep results non-empty with strict semantic gates
    # (especially when embeddings fall back to hashing vectors).
    if any(k in ql for k in ("quantum", "qubit", "quantum computing", "quantum-computing", "quantum computer")):
        api_topics = list(api_topics)
        api_topics.extend(
            [
                {
                    "title": "Machine Learning for Quantum Computing: Algorithms and Applications",
                    "domain": "Other",
                    "citations": 60,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["quantum computing", "machine learning", "algorithms"],
                },
                {
                    "title": "AI-Assisted Quantum Error Mitigation and Noise Characterization",
                    "domain": "Other",
                    "citations": 45,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["quantum error mitigation", "noise", "AI"],
                },
                {
                    "title": "Deep Learning for Quantum Circuit Optimization and Compilation",
                    "domain": "Other",
                    "citations": 50,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["quantum circuits", "optimization", "compilation"],
                },
            ]
        )

    # Add cybersecurity anchors for common "AI for cyber security" / forensics queries.
    if any(k in ql for k in ("cyber", "security", "cybersecurity", "infosec", "malware", "phishing", "ransomware", "intrusion", "threat", "forensics", "digital forensics")):
        api_topics = list(api_topics)
        api_topics.extend(
            [
                {
                    "title": "Machine Learning for Intrusion Detection in Network Traffic",
                    "domain": "Other",
                    "citations": 70,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["intrusion detection", "network security", "machine learning"],
                },
                {
                    "title": "AI-Based Malware Detection Using Behavioral Analysis and Graph Features",
                    "domain": "Other",
                    "citations": 65,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["malware", "behavioral analysis", "graph learning"],
                },
                {
                    "title": "Detecting Phishing Attacks with NLP and Transformer Models",
                    "domain": "Other",
                    "citations": 55,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["phishing", "NLP", "transformers"],
                },
                {
                    "title": "Digital Forensics with Machine Learning: Evidence Triage and Artifact Extraction",
                    "domain": "Other",
                    "citations": 50,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["digital forensics", "evidence triage", "artifact extraction"],
                },
                {
                    "title": "Automated Memory and File-System Forensics Using AI-Assisted Feature Extraction",
                    "domain": "Other",
                    "citations": 40,
                    "year": 2023,
                    "source": "Seed",
                    "_keywords": ["memory forensics", "file system", "feature extraction"],
                },
            ]
        )

    # Add IT/enterprise anchors for "AI in IT sector" / enterprise AI queries.
    if any(k in ql for k in _AI_MARKERS) and any(m in ql for m in _IT_SECTOR_MARKERS):
        api_topics = list(api_topics)
        for t in _it_sector_topics(str(query)):
            api_topics.append(
                {
                    "title": t,
                    "domain": "Other",
                    "citations": 40,
                    "year": 2024,
                    "source": "Seed",
                    "_keywords": ["enterprise", "information technology", "industry", "adoption", "impact"],
                }
            )

    qv = embed_text(query_norm or str(query))
    # Hard semantic gate: prevents irrelevant candidates from being ranked/returned.
    # When we fall back to the hashing BoW embedding (256-d), cosine similarities are
    # typically much lower than transformer embeddings. Use a safer default unless
    # the user explicitly overrides SEMANTIC_MIN_HARD.
    try:
        sem_gate = float(os.getenv("SEMANTIC_MIN_HARD", "0.40") or 0.40)
    except Exception:
        sem_gate = 0.40

    if "SEMANTIC_MIN_HARD" not in os.environ and isinstance(qv, list) and len(qv) == 256:
        # Hashing-BoW similarities are much lower; keep the gate low enough to not
        # eliminate everything for synonym-heavy queries.
        sem_gate = 0.05

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

        # If the topic is semantically relevant, infer domain from the TOPIC itself.
        # Do NOT include query text here: it can incorrectly force domains.
        if simf >= 0.45:
            domain = _infer_domain(f"{title} {' '.join(keywords)}")

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
        if not _looks_like_research_topic(title, query_norm or str(query)):
            continue

        # Enforce semantic gate, but allow seeds / synthetic candidates to pass if they are topic-shaped.
        src = str(item.get("source", "") or "")
        if src not in {"Seed", "Datasets-synthetic"} and simf < sem_gate:
            continue

        topics_with_similarity.append((topic, simf))

    # Domain + noise filtering (keep non-empty). Goal: stop generic-token drift.
    domain_hint = _detect_domain(query_norm or str(query))
    q_core = set(_query_core_tokens(query_norm or str(query)))
    # Avoid over-filtering on broad "meta" words (impact/sector/etc.).
    meta_q = {
        "impact",
        "impacts",
        "effect",
        "effects",
        "sector",
        "sectors",
        "industry",
        "industries",
        "role",
        "future",
        "trend",
        "trends",
        "overview",
        "survey",
        "review",
        "study",
        "studies",
    }
    q_core_strict = {t for t in q_core if t not in meta_q}

    def _topic_tokens(t: Topic) -> set[str]:
        return set(_word_tokens(f"{t.title} {' '.join(t.keywords or [])}"))

    allow_energy = any(k in (query_norm or str(query) or "").lower() for k in ("energy", "solar", "wind", "grid", "ev", "battery", "power", "charging"))
    filtered: List[tuple[Topic, float]] = []
    for t, simf in topics_with_similarity:
        tt = _topic_tokens(t)

        if domain_hint == "education":
            edu_tokens = set(_word_tokens(" ".join(_EDUCATION_MARKERS)))
            if t.domain != "EdTech" and not (tt & edu_tokens):
                continue
            if not allow_energy and any(x in (t.title or "").lower() for x in ("grid", "solar", "wind", "battery", "energy", "power")):
                continue

        if domain_hint == "tourism":
            if t.domain != "Tourism" and not (tt & {"tourism", "travel", "hospitality", "heritage", "ecotourism", "eco"}):
                continue

        if q_core_strict and not (tt & q_core_strict):
            continue

        filtered.append((t, simf))

    if filtered:
        topics_with_similarity = filtered

    topics_with_similarity.sort(key=lambda x: x[1], reverse=True)
    topics_with_similarity = topics_with_similarity[:50]

    if not topics_with_similarity:
        # Last-resort offline fallback: always return something topic-shaped.
        used_fallback = True
        fallback_titles = _generate_basic_topics(str(query)) if intent != "project" else _generate_project_topics(str(query))

        for ft in fallback_titles:
            title = str(ft).strip()
            if not title:
                continue
            if not _looks_like_research_topic(title, query_norm or str(query)):
                continue

            year = 2024
            citations = 25
            domain = _infer_domain(f"{title} {query_norm or str(query)}")
            policy_tags = list(match_policy_tags(title))

            title_words = [w.strip().lower() for w in re.split(r"[\s:,-]+", title) if len(w.strip()) > 3]
            stopwords = {"the", "and", "for", "with", "using", "based", "from", "that", "this", "are", "was", "were", "through", "towards"}
            keywords = [w for w in title_words if w not in stopwords][:8]
            if not keywords:
                keywords = [title.lower()[:30]]

            topic = Topic(
                topic_id=_topic_id_from_title(title, year),
                title=title,
                domain=domain,
                keywords=keywords,
                policy_tags=policy_tags,
                citations=citations,
                year=year,
            )
            simf = float(cosine_similarity(qv, embed_text(title)))
            topics_with_similarity.append((topic, simf))

        topics_with_similarity.sort(key=lambda x: x[1], reverse=True)
        topics_with_similarity = topics_with_similarity[:50]

    if not topics_with_similarity and intent != "project":
        # Final fallback: ask Groq to generate topics when strict filtering eliminates everything.
        domain_hint = _detect_domain(query_norm or str(query))
        gen = _groq_generate_topics(str(query), str(language), domain_hint, n=10)
        if gen:
            api_topics = []
            for g in gen:
                title = str(g.get("title") or "").strip()
                if not title:
                    continue
                api_topics.append(
                    {
                        "title": title,
                        "domain": str(g.get("domain") or _infer_domain(title)),
                        "citations": 20,
                        "year": 2024,
                        "source": "Seed",
                        "_keywords": list(g.get("keywords") or []),
                    }
                )

            # Re-run the minimal candidate build pass.
            qv = embed_text(query_norm or str(query))
            topics_with_similarity = []
            for item in api_topics:
                title = str(item.get("title", "") or "").strip()
                if not title:
                    continue
                year = int(item.get("year", 2024) or 2024)
                citations = int(item.get("citations", 20) or 20)
                domain = str(item.get("domain") or _infer_domain(title))
                policy_tags = list(item.get("policy_tags") or match_policy_tags(title))

                stopwords = {"the", "and", "for", "with", "using", "based", "from", "that", "this", "are", "was", "were", "through", "towards"}
                title_words = [w.strip().lower() for w in re.split(r"[\s:,-]+", title) if len(w.strip()) > 3]
                keywords = [w for w in title_words if w not in stopwords][:8]
                raw_keywords = list(item.get("_keywords") or [])
                for kw in raw_keywords:
                    if isinstance(kw, str) and 2 <= len(kw) <= 50 and kw.count("\n") == 0 and kw.lower() not in keywords:
                        keywords.append(kw.lower())
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
                simf = float(cosine_similarity(qv, embed_text(title)))
                if not _looks_like_research_topic(title, query_norm or str(query)):
                    continue
                topics_with_similarity.append((topic, simf))

            topics_with_similarity.sort(key=lambda x: x[1], reverse=True)
            topics_with_similarity = topics_with_similarity[:50]

    if not topics_with_similarity:
        # Should be unreachable due to fallback, but keep as a final guard.
        # If we still have nothing, return basic topics rather than an empty list.
        fallback_titles = _generate_basic_topics(str(query))
        topics_simple = [{"title": t, "score": 0.0} for t in fallback_titles[:10]]
        return {"query": query, "language": language, "user_id": user_id, "topics": topics_simple, "recommended_topics": [], "warning": "Fallback used (no valid candidates after filtering)"}

    # Ensure we have enough candidates to return at least N topics.
    try:
        min_return = int(os.getenv("TOPIC_MIN_RETURN", "5") or 5)
    except Exception:
        min_return = 5
    min_return = max(3, min(min_return, 10))

    if intent != "project" and len(topics_with_similarity) < min_return:
        domain_hint = _detect_domain(query_norm or str(query))
        gen = _groq_generate_topics(str(query), str(language), domain_hint, n=max(min_return, 8))
        if gen:
            existing_titles = {t.title.strip().lower() for t, _ in topics_with_similarity if getattr(t, "title", None)}
            qv = embed_text(query_norm or str(query))
            for g in gen:
                title = str(g.get("title") or "").strip()
                if not title:
                    continue
                if title.lower() in existing_titles:
                    continue
                if not _looks_like_research_topic(title, query_norm or str(query)):
                    continue

                year = 2024
                citations = 20
                domain = str(g.get("domain") or _infer_domain(title))
                policy_tags = list(match_policy_tags(title))

                stopwords = {"the", "and", "for", "with", "using", "based", "from", "that", "this", "are", "was", "were", "through", "towards"}
                title_words = [w.strip().lower() for w in re.split(r"[\s:,-]+", title) if len(w.strip()) > 3]
                keywords = [w for w in title_words if w not in stopwords][:8]
                kws = g.get("keywords") or []
                if isinstance(kws, list):
                    for kw in kws:
                        if isinstance(kw, str) and 2 <= len(kw) <= 50 and kw.count("\n") == 0 and kw.lower() not in keywords:
                            keywords.append(kw.lower())
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
                simf = float(cosine_similarity(qv, embed_text(title)))
                topics_with_similarity.append((topic, simf))
                existing_titles.add(title.lower())

            topics_with_similarity.sort(key=lambda x: x[1], reverse=True)
            topics_with_similarity = topics_with_similarity[:50]

    ranked = score_and_rank(
        query=str(query),
        topics_with_similarity=topics_with_similarity,
        user_id=user_id,
        top_k=10,
        # For project intent, always return multiple practical ideas even when
        # semantic similarities are low (common with hashing embeddings).
        min_semantic=0.0 if intent == "project" else None,
        # Project-mode should be fast and not depend on policy/dataset indexing.
        use_policy=False if intent == "project" else True,
    )

    # Optional post-processing: LLM title polishing (Groq preferred, Gemini fallback).
    # This may rewrite titles; keep order and scores stable.
    try:
        rec = list(ranked.get("recommended_topics") or [])
        titles = [str(x.get("title") or "").strip() for x in rec]
        if any(titles):
            polished = _groq_polish_titles(str(query), str(language), titles)
            if polished and len(polished) == len(rec):
                for i, new_title in enumerate(polished):
                    nt = (new_title or "").strip()
                    if not nt:
                        continue
                    rec[i]["title"] = nt
                    # Keep IDs consistent with displayed title.
                    try:
                        yr = int(rec[i].get("year", 2024) or 2024)
                        rec[i]["topic_id"] = _topic_id_from_title(nt, yr)
                    except Exception:
                        pass
                ranked["recommended_topics"] = rec
            else:
                polish_topics_inplace(ranked.get("recommended_topics") or [])
    except Exception:
        try:
            polish_topics_inplace(ranked.get("recommended_topics") or [])
        except Exception:
            pass

    # Collapse duplicates produced by policy-matching expansions.
    ranked_topics = _dedupe_merge_recommended_topics(list(ranked.get("recommended_topics") or []))
    ranked["recommended_topics"] = ranked_topics

    # Ensure we return a minimum number of topics for every query.
    # If strict semantic filtering leaves fewer than N, pad with Groq-generated research topics.
    if intent != "project" and len(ranked_topics) < min_return:
        domain_hint = _detect_domain(query_norm or str(query))
        gen = _groq_generate_topics(str(query), str(language), domain_hint, n=max(min_return, 8))

        # If Groq isn't available/returns too little, fall back to offline generated titles.
        if not gen or len(gen) < min_return:
            fallback = _generate_basic_topics(str(query))
            for t in fallback:
                gen.append({"title": t, "domain": _infer_domain(t), "keywords": []})

        existing_titles = {str(x.get("title", "")).strip().lower() for x in ranked_topics}
        existing_ids = {str(x.get("topic_id", "")).strip() for x in ranked_topics}

        for g in gen:
            if len(ranked_topics) >= min_return:
                break

            title = str(g.get("title") or "").strip()
            if not title:
                continue
            if title.lower() in existing_titles:
                continue
            if not _looks_like_research_topic(title, query_norm or str(query)):
                continue

            domain = str(g.get("domain") or _infer_domain(title))
            kws = g.get("keywords") or []
            if not isinstance(kws, list):
                kws = []
            keywords = [str(x).strip().lower() for x in kws if str(x).strip()][:8]
            if not keywords:
                title_words = [w.strip().lower() for w in re.split(r"[\s:,-]+", title) if len(w.strip()) > 3]
                stopwords = {"the", "and", "for", "with", "using", "based", "from", "that", "this", "are", "was", "were", "through", "towards"}
                keywords = [w for w in title_words if w not in stopwords][:8]

            year = 2024
            topic_id = _topic_id_from_title(title, year)
            if topic_id in existing_ids:
                continue

            ranked_topics.append(
                {
                    "topic_id": topic_id,
                    "title": title,
                    "domain": domain,
                    "keywords": keywords,
                    "policy_tags": [],
                    "citations": 20,
                    "year": year,
                    "semantic_similarity": 0.0,
                    "policy_weight": 1.0,
                    "trend_score": 0.0,
                    "keyword_score": 0.0,
                    "policy_meta": {"policies": [], "policy_ids": [], "domains": [], "intents": []},
                    "final_score": 0.0,
                    "final_score_100": 0.0,
                    "reasons": ["Generated fallback topic"],
                }
            )
            existing_titles.add(title.lower())
            existing_ids.add(topic_id)

        # Polish padded titles too (optional).
        try:
            titles = [str(x.get("title") or "").strip() for x in ranked_topics]
            polished = _groq_polish_titles(str(query), str(language), titles)
            if polished and len(polished) == len(ranked_topics):
                for i, nt in enumerate(polished):
                    new_title = (nt or "").strip()
                    if not new_title:
                        continue
                    ranked_topics[i]["title"] = new_title
                    try:
                        yr = int(ranked_topics[i].get("year", 2024) or 2024)
                        ranked_topics[i]["topic_id"] = _topic_id_from_title(new_title, yr)
                    except Exception:
                        pass
        except Exception:
            pass

        ranked["recommended_topics"] = ranked_topics

    topics_simple = [
        {"title": str(t.get("title") or "").strip(), "score": float(t.get("final_score") or 0.0)}
        for t in ranked_topics
        if str(t.get("title") or "").strip()
    ]

    # Always return at least N non-empty topic titles.
    if len(topics_simple) < min_return:
        existing = {str(x.get("title", "")).strip().lower() for x in topics_simple}
        domain_hint = _detect_domain(query_norm or str(query))
        gen = _groq_generate_topics(str(query), str(language), domain_hint, n=max(min_return, 8))
        if not gen:
            fallback = _generate_basic_topics(str(query))
            gen = [{"title": t, "domain": _infer_domain(t), "keywords": []} for t in fallback]

        for g in gen:
            if len(topics_simple) >= min_return:
                break
            title = str(g.get("title") or "").strip()
            if not title:
                continue
            if title.lower() in existing:
                continue
            topics_simple.append({"title": title, "score": 0.0})
            existing.add(title.lower())

    # Keep the lightweight alias in sync with recommended_topics.
    # (No blank titles are ever appended.)

    # Ensure project-mode always returns at least 5 ideas.
    if intent == "project" and len(topics_simple) < 5:
        existing = {str(x.get("title", "")).strip().lower() for x in topics_simple}
        extras = _filter_project_topics(_generate_project_topics(str(query)), str(query))
        for et in extras:
            if len(topics_simple) >= 5:
                break
            if (et or "").strip().lower() in existing:
                continue
            # Only pad with ideas that would pass the same topic-shape validator.
            if not _looks_like_research_topic(str(et), query_norm or str(query)):
                continue
            topics_simple.append({"title": et, "score": 0.0})
            existing.add((et or "").strip().lower())

    resp: Dict[str, Any] = {
        "query": query,
        "language": language,
        "user_id": user_id,
        **ranked,
        # Spec-friendly alias
        "topics": topics_simple,
    }

    # Optional: summarize output using Groq (OpenAI-compatible endpoint).
    try:
        enabled = (os.getenv("GROQ_SUMMARY_ENABLED", "true") or "true").strip().lower() in {"1", "true", "yes", "y"}
        if enabled:
            summary = _groq_summarize(str(query), str(language), list(resp.get("recommended_topics") or []))
            if summary:
                resp["summary"] = summary
    except Exception:
        pass

    if used_fallback and intent != "project":
        resp["warning"] = "Fallback used (query normalization / basic topic generator)"

    return resp

