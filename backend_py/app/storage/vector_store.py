# app/storage/vector_store.py
# FAISS-based vector store for chunk embeddings

import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed — vector store will use numpy fallback.")


class VectorStore:
    """Store and search chunk embeddings using FAISS (or numpy fallback)."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.chunks: list[dict] = []  # {text, paper_title, section, index}

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dimension)
            logger.info(f"FAISS index created (dim={dimension}).")
        else:
            self._embeddings = []
            logger.info(f"Numpy fallback store created (dim={dimension}).")

    def add(self, embeddings: list[np.ndarray], metadata: list[dict]):
        """Add embeddings with metadata to the store.

        Args:
            embeddings: List of numpy arrays (one per chunk).
            metadata: List of dicts with keys like text, paper_title, section.
        """
        if not embeddings:
            return

        matrix = np.array(embeddings, dtype=np.float32)

        if FAISS_AVAILABLE:
            self.index.add(matrix)
        else:
            self._embeddings.append(matrix)

        self.chunks.extend(metadata)
        logger.info(f"Added {len(embeddings)} vectors. Total: {len(self.chunks)}.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Find most similar chunks to a query embedding.

        Returns:
            List of metadata dicts for the top_k most similar chunks,
            each augmented with a 'score' key.
        """
        if not self.chunks:
            return []

        query = np.array([query_embedding], dtype=np.float32)

        if FAISS_AVAILABLE:
            distances, indices = self.index.search(query, min(top_k, len(self.chunks)))
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunks):
                    item = dict(self.chunks[idx])
                    item["score"] = float(dist)
                    results.append(item)
            return results
        else:
            # Numpy fallback: brute-force cosine similarity
            all_emb = np.vstack(self._embeddings)
            # L2 distances
            dists = np.linalg.norm(all_emb - query, axis=1)
            top_indices = np.argsort(dists)[:top_k]
            results = []
            for idx in top_indices:
                item = dict(self.chunks[idx])
                item["score"] = float(dists[idx])
                results.append(item)
            return results

    def clear(self):
        """Reset the store."""
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self._embeddings = []
        self.chunks = []

    @property
    def size(self) -> int:
        return len(self.chunks)
