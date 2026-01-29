"""
query_builder.py

Purpose:
- Convert Gemini JSON output into a clean, semantic query text
- Build natural-language text optimized for embedding
- Use primary_research_area, methods, application_domains, key_concepts

Input: Gemini structured JSON
Output: Single query string
"""


def build_query_from_gemini(gemini_output: dict) -> str:
    """
    Convert Gemini JSON output into a semantic query string.
    
    Args:
        gemini_output: Dictionary with Gemini-formatted fields:
            - primary_research_area
            - secondary_areas (list)
            - methods (list)
            - application_domains (list)
            - key_concepts (list)
            - condensed_summary
    
    Returns:
        String: Natural-language query optimized for embedding
    """
    if not gemini_output or not isinstance(gemini_output, dict):
        raise ValueError("gemini_output must be a non-empty dictionary")
    
    # Extract fields with defaults
    primary_area = gemini_output.get("primary_research_area", "").strip()
    methods = gemini_output.get("methods", [])
    app_domains = gemini_output.get("application_domains", [])
    key_concepts = gemini_output.get("key_concepts", [])
    
    # Build natural language query
    query_parts = []
    
    # Primary research area
    if primary_area:
        query_parts.append(f"Research in {primary_area}")
    
    # Methods
    if methods and isinstance(methods, list):
        methods_text = ", ".join(str(m).strip() for m in methods if m)
        if methods_text:
            query_parts.append(f"using methods like {methods_text}")
    
    # Application domains
    if app_domains and isinstance(app_domains, list):
        domains_text = ", ".join(str(d).strip() for d in app_domains if d)
        if domains_text:
            query_parts.append(f"with applications in {domains_text}")
    
    # Key concepts
    if key_concepts and isinstance(key_concepts, list):
        concepts_text = ", ".join(str(c).strip() for c in key_concepts if c)
        if concepts_text:
            query_parts.append(f"involving {concepts_text}")
    
    # Combine into single sentence
    if query_parts:
        query = ". ".join(query_parts) + "."
    else:
        # Fallback to condensed summary if available
        query = gemini_output.get("condensed_summary", "")
    
    if not query:
        raise ValueError("Could not build query from gemini_output")
    
    return query


if __name__ == "__main__":
    # Test example
    test_gemini = {
        "primary_research_area": "Machine Learning",
        "secondary_areas": ["AI", "Deep Learning"],
        "methods": ["neural networks", "gradient descent"],
        "application_domains": ["computer vision", "natural language processing"],
        "key_concepts": ["optimization", "feature extraction"],
        "condensed_summary": "Research on deep learning methods"
    }
    
    query = build_query_from_gemini(test_gemini)
    print("Generated query:")
    print(query)
