from __future__ import annotations

import re
from typing import Dict, List, Tuple

import os

from .datasets_loader import build_policy_weight_table_from_datasets, search_policies


def _clean_intent_text(text: str, max_length: int = 120) -> str:
    """Clean PDF artifacts from policy intent text for display."""
    if not text or not isinstance(text, str):
        return ""
    # Remove non-printable chars
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    # Drop common PDF artifact runs (====, :::::, etc.)
    cleaned = re.sub(r"[=:_\-]{3,}", " ", cleaned)
    # Remove typical boilerplate headings/noise
    cleaned = re.sub(
        r"\b(orders?|order|vide|reference|dated|dt\.?|g\.o\.?|goms|ms\.?|annexure|chapter|section)\b\s*[:=]*",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Trim leading bullets/punctuation
    cleaned = cleaned.lstrip("-–—•:;,. ")
    # Truncate long PDF chunks
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    return cleaned


POLICY_PRIORITY_WEIGHTS = {
    "nep": 1.6,
    "national education policy": 1.6,
    "ap innovation": 1.4,
    "innovation": 1.4,
    "clean energy": 1.3,
    "energy": 1.2,
}


# Optional demo polish: map raw dataset filenames/ids to human-friendly names.
# Used only in response text/metadata; matching/scoring still uses raw IDs.
POLICY_DISPLAY_NAMES: Dict[str, str] = {
    "GOMs-No-2-dated-24-02-2025": "AP Innovation Government Order (2025)",
    "Compendium_of_Datasets_and_Registries_in_India_2024": "National Open Data & Research Registry (2024)",
}


def _display_policy_name(raw: str) -> str:
    r = (raw or "").strip()
    if not r:
        return ""
    if r in POLICY_DISPLAY_NAMES:
        return POLICY_DISPLAY_NAMES[r]

    # Remove common file extensions
    r = re.sub(r"\.(pdf|xlsx|xls)$", "", r, flags=re.IGNORECASE)

    # Prefer the part before a noisy "_" suffix (often contains order numbers/dates)
    if "_" in r:
        head, tail = r.split("_", 1)
        if re.search(r"\b(go|goms|ms|dt|dated|no\.|order)\b", tail, flags=re.IGNORECASE) or re.search(r"\d{4}", tail):
            r = head

    # Humanize separators
    r = r.replace("_", " ")
    r = r.replace("-", " ")
    r = re.sub(r"\s+", " ", r).strip()

    # Drop trailing long numeric IDs
    r = re.sub(r"\b\d{7,}\b", "", r).strip()
    r = re.sub(r"\s+", " ", r).strip()
    return r


def load_policy_weight_table() -> List[Dict]:
    # Source of truth is the Datasets/ folder (PDFs + XLSX).
    # This intentionally avoids using data/*.json files.
    return build_policy_weight_table_from_datasets()


def _is_education_query(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ("education", "learning", "student", "higher", "university", "college", "teaching", "curriculum", "accredit", "dropout", "edtech"))


def _policy_allowed_for_context(topic_text: str, policy_name: str, domains: List[str]) -> bool:
    """Heuristic filter to reduce noisy policy matches.

    For education topics, block Energy/EV/Drone/Aerospace policies.
    """

    if not policy_name and not domains:
        return True

    t = (topic_text or "").lower()
    pname = (policy_name or "").lower()
    dset = {str(d) for d in (domains or [])}

    if _is_education_query(t):
        blocked_name_tokens = ("drone", "ev", "electric", "energy", "power", "battery", "charging")
        blocked_domains = {"Clean Energy", "Aerospace"}
        if any(b in pname for b in blocked_name_tokens):
            return False
        if dset.intersection(blocked_domains):
            return False

        # Allowlist-style preference for education/research/innovation policies
        allowed_name_tokens = ("nep", "education", "research", "innovation", "startup", "skill")
        allowed_domains = {"EdTech", "Innovation", "Other"}
        if any(a in pname for a in allowed_name_tokens):
            return True
        if dset.intersection(allowed_domains):
            return True
        return False

    return True


def policy_alignment(
    topic_text: str,
    topic_policy_tags: List[str],
    query_text: str = "",
) -> Tuple[float, List[str], Dict[str, List[str]]]:
    """Return (policy_weight, reasons, metadata)."""

    # Use query_text for context filtering if provided.
    text = ((query_text or "") + " " + (topic_text or "")).strip()
    matched_policies: List[str] = []
    matched_domains: List[str] = []
    intents: List[str] = []

    # Preferred path: semantic retrieval over policy-level embeddings
    hit_min = float(os.getenv("POLICY_HIT_MIN", "0.25") or 0.25)
    hits = search_policies(text, top_k=5)
    hits = [h for h in (hits or []) if float(h.get("_score", 0.0) or 0.0) >= hit_min]
    if hits:
        best_weight = 1.0
        reasons: List[str] = []

        # Keep explanations concise: only describe the strongest couple hits.
        for i, h in enumerate(hits):
            policy_name = str(h.get("policy_name", "")).strip()
            base_weight = float(h.get("weight", 1.0) or 1.0)
            sim = float(h.get("_score", 0.0) or 0.0)
            domains = h.get("domains") or []
            intent = str(h.get("intent", "")).strip()

            if not _policy_allowed_for_context(text, policy_name, list(domains or [])):
                continue

            priority = 1.0
            lname = policy_name.lower()
            for key, mult in POLICY_PRIORITY_WEIGHTS.items():
                if key in lname:
                    priority = max(priority, mult)
                    break

            sim01 = max(0.0, min(1.0, sim))
            effective_weight = base_weight * priority * (1.0 + sim01)
            best_weight = max(best_weight, effective_weight)

            if policy_name:
                matched_policies.append(policy_name)
                if i < 2:
                    reasons.append(f"Aligned with {_display_policy_name(policy_name) or policy_name}")
            if domains:
                matched_domains.extend(domains)
                if i < 1:
                    reasons.append(f"Supports domains: {', '.join(domains[:2])}")
            if intent:
                clean_intent = _clean_intent_text(intent)
                if clean_intent:
                    intents.append(clean_intent)

        # Only include user/topic-provided tags if they look like human policy names.
        for t in (topic_policy_tags or [])[:5]:
            tt = str(t or "").strip()
            if not tt:
                continue
            if len(tt) > 60:
                continue
            if re.search(r"\d{3,}", tt):
                continue
            if tt.count("_") >= 3:
                continue
            matched_policies.append(tt)
            if len(reasons) < 5:
                reasons.append(f"Aligned with {_display_policy_name(tt) or tt}")

        uniq_raw = list(dict.fromkeys(matched_policies))
        uniq_display = [(_display_policy_name(p) or p) for p in uniq_raw]
        return best_weight, list(dict.fromkeys(reasons))[:5], {
            "policies": uniq_display,
            "policy_ids": uniq_raw,
            "domains": list(dict.fromkeys(matched_domains)),
            "intents": intents[:2],
        }

    # Fallback: keyword table derived from datasets (no Qdrant)
    policies = load_policy_weight_table()
    lowered = text.lower()
    best_weight = 1.0
    reasons: List[str] = []
    for p in policies:
        policy_name = str(p.get("policy", ""))
        weight = float(p.get("weight", 1.0) or 1.0)
        keywords = [str(x) for x in (p.get("keywords", []) or [])]
        matched = False
        if policy_name and any(policy_name.lower() == (t or "").lower() for t in (topic_policy_tags or [])):
            matched = True
        if not matched:
            for kw in keywords[:100]:
                if kw and kw.lower() in lowered:
                    matched = True
                    break
        if matched:
            matched_policies.append(policy_name)
            best_weight = max(best_weight, weight)
            reasons.append(f"Aligned with {policy_name}")

    uniq_raw = list(dict.fromkeys(matched_policies))
    uniq_display = [(_display_policy_name(p) or p) for p in uniq_raw]
    return best_weight, list(dict.fromkeys(reasons))[:5], {
        "policies": uniq_display,
        "policy_ids": uniq_raw,
        "domains": [],
        "intents": [],
    }

