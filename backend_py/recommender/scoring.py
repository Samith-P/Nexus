"""
scoring.py

Purpose:
- Implement metric-aware scoring formula
- Combine semantic similarity with journal impact metrics
- Normalize and handle missing values safely

Scoring formula:
final_score = 0.55 * semantic_similarity +
              0.25 * normalized_sjr +
              0.20 * normalized_citations_per_doc
"""

from typing import Optional


def normalize_value(value: Optional[float], min_val: float = 0, max_val: float = 2.0) -> float:
    """
    Normalize a value to 0-1 range.
    
    Args:
        value: Value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value
    
    Returns:
        float: Normalized value (0-1), or 0 if value is None/invalid
    """
    if value is None:
        return 0.0
    
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    
    # Clamp to range and normalize
    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 1.0
    
    return (value - min_val) / (max_val - min_val)


def compute_final_score(
    semantic_similarity: float,
    sjr: Optional[float] = None,
    citations_per_doc_2y: Optional[float] = None
) -> float:
    """
    Compute final recommendation score with metric-aware weighting.
    
    Args:
        semantic_similarity: Cosine similarity (0-1)
        sjr: Journal SJR (typically 0-3, default None)
        citations_per_doc_2y: Citations per document (typically 0-50, default None)
    
    Returns:
        float: Final score (0-1)
    """
    if not (0 <= semantic_similarity <= 1):
        semantic_similarity = max(0, min(1, semantic_similarity))
    
    # Normalize metrics
    sjr_normalized = normalize_value(sjr, min_val=0, max_val=2.0)
    citations_normalized = normalize_value(
        citations_per_doc_2y,
        min_val=0,
        max_val=50.0
    )
    
    # Weighted formula
    final_score = (
        0.55 * semantic_similarity +
        0.25 * sjr_normalized +
        0.20 * citations_normalized
    )
    
    return float(final_score)


def score_journals(
    journals_with_similarity: list
) -> list:
    """
    Score a list of journals with (journal_dict, similarity) tuples.
    
    Args:
        journals_with_similarity: List of (journal, similarity_score) tuples
    
    Returns:
        list: List of journal dicts with added 'final_score' and 'semantic_score' fields
    """
    scored_journals = []
    
    for journal, semantic_sim in journals_with_similarity:
        final_score = compute_final_score(
            semantic_similarity=semantic_sim,
            sjr=journal.get("sjr"),
            citations_per_doc_2y=journal.get("citations_per_doc_2y")
        )
        
        # Add scores to journal dict
        journal_result = journal.copy()
        journal_result["semantic_score"] = float(semantic_sim)
        journal_result["final_score"] = float(final_score)
        
        scored_journals.append(journal_result)
    
    # Sort by final score (descending)
    scored_journals.sort(key=lambda x: x["final_score"], reverse=True)
    
    return scored_journals


if __name__ == "__main__":
    # Test example
    test_cases = [
        {
            "title": "High Semantic, High SJR",
            "semantic_sim": 0.85,
            "sjr": 1.5,
            "citations": 35.0
        },
        {
            "title": "High Semantic, Low SJR",
            "semantic_sim": 0.85,
            "sjr": 0.3,
            "citations": 5.0
        },
        {
            "title": "Low Semantic, High SJR",
            "semantic_sim": 0.45,
            "sjr": 2.0,
            "citations": 45.0
        },
    ]
    
    print("Scoring examples:\n")
    for case in test_cases:
        score = compute_final_score(
            case["semantic_sim"],
            case["sjr"],
            case["citations"]
        )
        print(f"{case['title']}")
        print(f"  Semantic: {case['semantic_sim']:.3f}, SJR: {case['sjr']}, Citations: {case['citations']}")
        print(f"  Final Score: {score:.4f}\n")
