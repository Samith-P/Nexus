"""
semantic_search.py

Purpose:
- Load journals_master.json lazily
- Compute cosine similarity between query and journal embeddings
- Return top-N journals by semantic relevance
"""

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Global journals index (lazy loaded)
_journals_index = None


def load_journals_index():
    """
    Load journals_master.json lazily.
    
    Returns:
        list: List of journal dictionaries with embeddings
    """
    global _journals_index
    if _journals_index is None:
        master_file = Path(__file__).parent.parent / "journals_master.json"
        
        if not master_file.exists():
            raise FileNotFoundError(f"journals_master.json not found at {master_file}")
        
        with open(master_file, "r", encoding="utf-8") as f:
            _journals_index = json.load(f)
        
        print(f"[INFO] Loaded {len(_journals_index)} journals from index")
    
    return _journals_index


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector (list or numpy array)
        vec2: Second vector (list or numpy array)
    
    Returns:
        float: Cosine similarity score (0-1)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def semantic_search(query_embedding: list, top_n: int = 50) -> List[Tuple[dict, float]]:
    """
    Search journals by semantic similarity to query embedding.
    
    Args:
        query_embedding: Query vector (384-dimensional)
        top_n: Number of top results to return (default: 50)
    
    Returns:
        list: List of tuples (journal_dict, similarity_score) sorted by score DESC
    """
    if not query_embedding:
        raise ValueError("query_embedding cannot be empty")
    
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    
    # Load journals index
    journals = load_journals_index()
    
    # Compute similarity scores for all journals
    similarities = []
    
    for journal in journals:
        if "embedding" not in journal:
            continue  # Skip journals without embeddings
        
        journal_embedding = journal["embedding"]
        similarity = cosine_similarity(query_embedding, journal_embedding)
        similarities.append((journal, similarity))
    
    # Sort by similarity (descending) and return top-N
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]


if __name__ == "__main__":
    # Test example
    from query_embedder import embed_query
    
    test_query = "Research in machine learning using neural networks"
    embedding = embed_query(test_query)
    
    results = semantic_search(embedding, top_n=5)
    print(f"Top 5 results for: {test_query}\n")
    for journal, score in results:
        print(f"{journal['title']}")
        print(f"  Similarity: {score:.4f}\n")
