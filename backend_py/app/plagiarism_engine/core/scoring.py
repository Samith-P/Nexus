from __future__ import annotations


def clamp_int(x: float, lo: int = 0, hi: int = 100) -> int:
    try:
        v = int(round(float(x)))
    except Exception:
        v = 0
    return max(lo, min(hi, v))


def clamp_pct(x: float, lo: float = 0.0, hi: float = 100.0, *, decimals: int = 2) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    # Avoid tiny float artifacts.
    return float(round(v + 1e-12, decimals))


def _weight_from_similarity(sim: float) -> float:
    """Map a [0,1] similarity into a plagiarism weight in [0,1]."""

    try:
        s = float(sim)
    except Exception:
        s = 0.0

    if s >= 0.75:
        return 1.0
    if s >= 0.65:
        return 0.7
    if s >= 0.55:
        return 0.4
    return 0.0


def compute_scores(
    *,
    total_chunks: int,
    exact_hits: int,
    semantic_hits: int,
    cited_chunks: int,
    citation_total_chunks: int | None = None,
    match_similarities: list[float] | None = None,
    eligible_chunks: int | None = None,
    internal_reuse_count: int = 0,
) -> dict:
    total = max(1, int(total_chunks))
    exact_pct = 100.0 * (exact_hits / total)
    sem_hit_pct = 100.0 * (semantic_hits / total)

    # Default behavior: simple blend of hit rates.
    plagiarism_pct = float(min(100.0, (0.65 * exact_pct) + (0.35 * sem_hit_pct)))

    # Improved behavior (when similarities are provided): weight matches by strength.
    if match_similarities is not None:
        denom = max(1, int(eligible_chunks) if eligible_chunks is not None else total)
        units = 0.0
        for sim in match_similarities:
            units += _weight_from_similarity(sim)

        # Internal duplicate chunks are a weak plagiarism signal.
        try:
            units += 0.5 * max(0, int(internal_reuse_count))
        except Exception:
            pass

        plagiarism_pct = float(min(100.0, 100.0 * (units / denom)))

    originality = 100.0 - float(plagiarism_pct)

    citation_total = int(citation_total_chunks) if citation_total_chunks is not None else total
    citation_total = max(1, citation_total)
    citation_coverage = 100.0 * (cited_chunks / citation_total)
    return {
        "total_chunks": total_chunks,
        "exact_match_percentage": clamp_int(exact_pct),
        # When weighted similarities are used, `plagiarism_pct` already reflects semantic strength.
        "semantic_match_percentage": clamp_pct(plagiarism_pct if match_similarities is not None else sem_hit_pct),
        "citation_coverage_percentage": clamp_pct(citation_coverage),
        "plagiarism_percentage": clamp_pct(plagiarism_pct),
        "originality_score": clamp_pct(originality),
    }
