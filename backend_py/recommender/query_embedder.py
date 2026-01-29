"""
query_embedder.py

Purpose:
- Load sentence-transformers model
- Embed query text into a vector
- Model MUST match the one used for journal embeddings (all-MiniLM-L6-v2)
"""

import os
from sentence_transformers import SentenceTransformer

# Avoid GPU probing delays on Windows
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Global model instance (lazy loaded)
_model = None


def get_embedding_model():
    """
    Get or initialize the sentence-transformers model.
    Uses lazy loading to avoid loading if not needed.
    
    Returns:
        SentenceTransformer: Model instance for all-MiniLM-L6-v2
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_query(query_text: str) -> list:
    """
    Embed a query string into a vector.
    
    Args:
        query_text: String query to embed
    
    Returns:
        list: 384-dimensional embedding vector
    """
    if not query_text or not isinstance(query_text, str):
        raise ValueError("query_text must be a non-empty string")
    
    model = get_embedding_model()
    
    # Embed the query
    embedding = model.encode(
        query_text,
        convert_to_tensor=False,
        show_progress_bar=False
    )
    
    return embedding.tolist()


if __name__ == "__main__":
    # Test example
    test_query = "Research in machine learning using neural networks with applications in computer vision"
    embedding = embed_query(test_query)
    print(f"Query: {test_query}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
