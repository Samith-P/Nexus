"""
explainer.py

Purpose:
- Generate deterministic, human-readable explanations
- Reference semantic alignment, journal impact metrics
- NO LLM used here - pure template-based logic
"""

from typing import Optional


def generate_explanation(
    journal: dict,
    semantic_score: float,
    final_score: float,
    rank: int
) -> str:
    """
    Generate a human-readable explanation for why this journal is recommended.
    
    Args:
        journal: Journal dictionary
        semantic_score: Cosine similarity score (0-1)
        final_score: Final weighted score (0-1)
        rank: Ranking position (1-indexed)
    
    Returns:
        str: Explanation text
    """
    title = journal.get("title", "Unknown Journal")
    journal_type = journal.get("type", "journal").capitalize()
    sjr = journal.get("sjr")
    quartile = journal.get("quartile", "N/A")
    citations = journal.get("citations_per_doc_2y")
    h_index = journal.get("h_index")
    
    explanation_parts = []
    
    # Semantic alignment
    if semantic_score >= 0.75:
        alignment = "excellent semantic alignment"
    elif semantic_score >= 0.60:
        alignment = "strong semantic alignment"
    elif semantic_score >= 0.45:
        alignment = "moderate semantic alignment"
    else:
        alignment = "some semantic alignment"
    
    explanation_parts.append(
        f"#{rank}: {title} has {alignment} with your research."
    )
    
    # Journal impact metrics
    impact_factors = []
    
    if sjr is not None and sjr > 0:
        if sjr >= 1.5:
            impact_factors.append(f"high SJR ({sjr:.2f}, {quartile})")
        elif sjr >= 0.75:
            impact_factors.append(f"solid SJR ({sjr:.2f}, {quartile})")
        else:
            impact_factors.append(f"SJR {sjr:.2f} ({quartile})")
    
    if citations is not None and citations > 0:
        if citations >= 30:
            impact_factors.append(f"highly cited ({citations:.1f} citations/doc)")
        elif citations >= 15:
            impact_factors.append(f"well-cited ({citations:.1f} citations/doc)")
        else:
            impact_factors.append(f"{citations:.1f} citations/doc")
    
    if h_index is not None and h_index > 0:
        if h_index >= 100:
            impact_factors.append(f"h-index {h_index}")
    
    if impact_factors:
        impact_text = ", ".join(impact_factors)
        explanation_parts.append(
            f"This {journal_type} is {impact_text}, indicating strong research impact."
        )
    
    # Research areas
    domain_text = journal.get("domain_text", "")
    if domain_text:
        explanation_parts.append(f"Focus area: {domain_text[:100]}...")
    
    # Confidence indicator
    if final_score >= 0.75:
        confidence = "highly recommended"
    elif final_score >= 0.60:
        confidence = "recommended"
    else:
        confidence = "worth considering"
    
    explanation_parts.append(f"Overall: {confidence} (score: {final_score:.3f})")
    
    return " ".join(explanation_parts)


def generate_explanations_batch(
    journals: list
) -> list:
    """
    Generate explanations for a batch of scored journals.
    
    Args:
        journals: List of journal dicts (must have semantic_score, final_score)
    
    Returns:
        list: Same journals with added 'explanation' field
    """
    result = []
    
    for rank, journal in enumerate(journals, 1):
        journal_with_explanation = journal.copy()
        
        explanation = generate_explanation(
            journal=journal,
            semantic_score=journal.get("semantic_score", 0),
            final_score=journal.get("final_score", 0),
            rank=rank
        )
        
        journal_with_explanation["explanation"] = explanation
        result.append(journal_with_explanation)
    
    return result


if __name__ == "__main__":
    # Test example
    test_journal = {
        "title": "Nature Machine Intelligence",
        "type": "journal",
        "sjr": 1.0,
        "quartile": "Q1",
        "h_index": 94,
        "citations_per_doc_2y": 18.44,
        "domain_text": "This journal publishes research in artificial intelligence and machine learning."
    }
    
    explanation = generate_explanation(
        test_journal,
        semantic_score=0.82,
        final_score=0.71,
        rank=1
    )
    
    print("Generated Explanation:")
    print(explanation)
