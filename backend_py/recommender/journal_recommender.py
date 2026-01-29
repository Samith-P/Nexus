"""
journal_recommender.py

Purpose:
- Orchestrate the full recommendation pipeline
- Accept Gemini output
- Return top-K recommendations with scores and explanations
"""

from typing import List, Dict, Optional
from .query_builder import build_query_from_gemini
from .query_embedder import embed_query
from .semantic_search import semantic_search
from .scoring import score_journals
from .explainer import generate_explanations_batch


class JournalRecommender:
    """
    Main recommendation engine.
    
    Pipeline:
    Gemini output → query_builder → query_embedder → semantic_search → 
    scoring → explainer → final results
    """
    
    def __init__(self):
        """Initialize the recommender."""
        self.initialized = True
    
    def recommend(
        self,
        gemini_output: Dict,
        top_k: int = 10,
        search_depth: int = 50
    ) -> List[Dict]:
        """
        Generate journal recommendations from Gemini output.
        
        Args:
            gemini_output: Gemini-formatted JSON with:
                - primary_research_area
                - secondary_areas (list)
                - methods (list)
                - application_domains (list)
                - key_concepts (list)
                - condensed_summary
            
            top_k: Number of top recommendations to return (default: 10)
            search_depth: Number of candidates to consider before scoring (default: 50)
        
        Returns:
            list: Top-K journal recommendations with:
                - title, type, sjr, quartile, h_index, citations_per_doc_2y
                - domain_text, embedding (removed for API response)
                - semantic_score (0-1)
                - final_score (0-1)
                - explanation (human-readable text)
        
        Raises:
            ValueError: If gemini_output is invalid
        """
        if not gemini_output:
            raise ValueError("gemini_output cannot be empty")
        
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        
        if search_depth < top_k:
            search_depth = top_k * 2  # Ensure we have enough candidates
        
        # Step 1: Build semantic query from Gemini output
        query_text = build_query_from_gemini(gemini_output)
        
        # Step 2: Embed the query
        query_embedding = embed_query(query_text)
        
        # Step 3: Semantic search for top candidates
        candidates = semantic_search(query_embedding, top_n=search_depth)
        
        # Step 4: Score candidates with metrics
        scored_journals = score_journals(candidates)
        
        # Step 5: Generate explanations
        final_results = generate_explanations_batch(scored_journals[:top_k])
        
        # Step 6: Clean up sensitive fields and prepare response
        response = []
        for journal in final_results:
            # Remove embedding and other sensitive fields
            clean_journal = {
                "title": journal.get("title"),
                "type": journal.get("type"),
                "sjr": journal.get("sjr"),
                "quartile": journal.get("quartile"),
                "h_index": journal.get("h_index"),
                "citations_per_doc_2y": journal.get("citations_per_doc_2y"),
                "publisher": journal.get("publisher"),
                "open_access": journal.get("open_access"),
                "country": journal.get("country"),
                "semantic_score": journal.get("semantic_score"),
                "final_score": journal.get("final_score"),
                "explanation": journal.get("explanation")
            }
            response.append(clean_journal)
        
        return response
    
    def get_recommendation_info(self) -> Dict:
        """
        Get information about the recommendation system.
        
        Returns:
            dict: System configuration and metadata
        """
        return {
            "status": "active",
            "model": "all-MiniLM-L6-v2",
            "journals_indexed": "13946",
            "scoring_formula": "0.55*semantic + 0.25*sjr + 0.20*citations",
            "explanation_type": "deterministic"
        }


if __name__ == "__main__":
    # Test example
    test_gemini_output = {
        "primary_research_area": "Machine Learning",
        "secondary_areas": ["Artificial Intelligence", "Deep Learning"],
        "methods": ["neural networks", "gradient descent", "convolutional networks"],
        "application_domains": ["computer vision", "image classification"],
        "key_concepts": ["optimization", "feature learning", "backpropagation"],
        "condensed_summary": "Research on deep learning for computer vision applications"
    }
    
    recommender = JournalRecommender()
    
    print("Testing JournalRecommender...")
    print(f"System info: {recommender.get_recommendation_info()}\n")
    
    try:
        recommendations = recommender.recommend(test_gemini_output, top_k=5)
        
        print(f"Top 5 recommendations:\n")
        for i, journal in enumerate(recommendations, 1):
            print(f"{i}. {journal['title']}")
            print(f"   Score: {journal['final_score']:.4f} (semantic: {journal['semantic_score']:.4f})")
            print(f"   {journal['explanation'][:150]}...\n")
    
    except Exception as e:
        print(f"Error: {e}")
