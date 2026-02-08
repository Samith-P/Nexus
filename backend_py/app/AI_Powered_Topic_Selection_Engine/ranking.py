from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import os

from .cf import cf_scores
from .embedding import cosine_similarity, embed_text
from .policy import policy_alignment
from .topic_kb import Topic, trend_score


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def final_score(semantic_similarity: float, trend_score_value: float, policy_weight: float) -> float:
    """Weighted scoring where semantic relevance dominates.

    Output is normalized to ~0..1.
    """

    # Normalize components to 0..1
    sem = _clamp01(semantic_similarity)
    # Trend score can exceed 1 for highly cited recent work; compress into 0..1
    trend_norm = _clamp01(float(trend_score_value) / 2.0)
    # Policy weight is >= 1.0; map 1..3 -> 0..1 (and clamp)
    policy_norm = _clamp01((float(policy_weight) - 1.0) / 2.0)

    # Weights: semantics > trend > policy
    return float(0.55 * sem + 0.25 * trend_norm + 0.20 * policy_norm)


def score_and_rank(
    query: str,
    topics_with_similarity: List[Tuple[Topic, float]],
    user_id: Optional[str] = None,
    now_year: int = 2026,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Core engine scoring + ranking + explainability.

    Cold-start handling (required): if no user history, CF is skipped.
    """

    candidate_ids = [t.topic_id for t, _ in topics_with_similarity]
    cf_map, cold_start = cf_scores(user_id, candidate_ids)

    trend_raw = [trend_score(t.citations, t.year, now_year=now_year) for t, _ in topics_with_similarity]

    query_vec = embed_text(query)
    out: List[Dict[str, Any]] = []

    sem_min = float(os.getenv("SEMANTIC_MIN", "0.40") or 0.40)
    policy_max = float(os.getenv("POLICY_WEIGHT_MAX", "2.5") or 2.5)

    for idx, (topic, retrieval_sim) in enumerate(topics_with_similarity):
        sem = float(retrieval_sim)
        if sem <= 0:
            sem = cosine_similarity(query_vec, embed_text(topic.text_for_embedding()))
        sem = max(0.0, min(1.0, sem))

        policy_weight, policy_reasons, policy_meta = policy_alignment(
            topic_text=f"{topic.title} {' '.join(topic.keywords)}",
            topic_policy_tags=topic.policy_tags,
            query_text=query,
        )
        policy_weight = max(1.0, float(policy_weight))
        # Prevent policy from overpowering semantics
        policy_weight = min(policy_weight, policy_max)
        trend_score_value = float(trend_raw[idx])

        cf_val = float(cf_map.get(topic.topic_id, 0.0)) if (cf_map and not cold_start) else 0.0

        score = final_score(
            semantic_similarity=sem,
            trend_score_value=trend_score_value,
            policy_weight=policy_weight,
        )

        # Hard penalty: old + zero-citation topics should not dominate unless very relevant.
        if int(topic.citations or 0) == 0 and int(topic.year or 0) < 2022 and sem < 0.80:
            score *= 0.6

        reasons: List[str] = []
        reasons.extend(policy_reasons)
        if sem > 0.6:
            reasons.append("Matches your query semantically")
        elif sem < sem_min:
            reasons.append(f"Filtered candidate: low semantic match (< {sem_min:.2f})")
        if trend_score_value > 0.6:
            reasons.append("High citation growth / momentum")
        if not cold_start and cf_val > 0.2:
            reasons.append("Recommended by similar researchers")
        if cold_start:
            reasons.append("Cold-start: ranked using similarity + policy + trends")

        out.append(
            {
                "topic_id": topic.topic_id,
                "title": topic.title,
                "domain": topic.domain,
                "keywords": topic.keywords,
                "policy_tags": topic.policy_tags,
                "citations": topic.citations,
                "year": topic.year,
                "semantic_similarity": round(float(sem), 4),
                "policy_weight": round(float(policy_weight), 4),
                "trend_score": round(float(trend_score_value), 4),
                "policy_meta": policy_meta,
                "final_score": round(float(score), 4),
                "final_score_100": round(float(score) * 100.0, 2),
                "reasons": list(dict.fromkeys(reasons)),
            }
        )

    out.sort(key=lambda x: x["final_score"], reverse=True)

    # Enforce semantic relevance threshold, but never return an empty list.
    filtered = [x for x in out if float(x.get("semantic_similarity", 0.0)) >= sem_min]
    selected = filtered if filtered else out
    return {"cold_start": cold_start, "recommended_topics": selected[: max(1, top_k)]}

