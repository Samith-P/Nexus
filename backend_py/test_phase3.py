"""
test_phase3.py

Quick test of Phase 3 recommendation engine
"""

from recommender import JournalRecommender

def main():
    recommender = JournalRecommender()
    
    test_gemini = {
        "primary_research_area": "Machine Learning",
        "secondary_areas": ["Artificial Intelligence", "Deep Learning"],
        "methods": ["neural networks", "gradient descent", "convolutional networks"],
        "application_domains": ["computer vision", "image classification"],
        "key_concepts": ["optimization", "feature learning", "backpropagation"],
        "condensed_summary": "Research on deep learning for computer vision"
    }
    
    print("[TEST] Generating recommendations...\n")
    recommendations = recommender.recommend(test_gemini, top_k=3)
    
    for i, journal in enumerate(recommendations, 1):
        print(f"{i}. {journal['title']}")
        print(f"   SJR: {journal['sjr']}, Quartile: {journal['quartile']}")
        print(f"   Scores - Semantic: {journal['semantic_score']:.4f}, Final: {journal['final_score']:.4f}")
        print(f"   {journal['explanation'][:120]}...\n")
    
    print("[OK] Phase 3 recommendation engine is working!")

if __name__ == "__main__":
    main()
