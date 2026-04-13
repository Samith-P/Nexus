# app/pipeline/embedder.py
# Stage 4b — Text embedding (WIRED UP — will connect to FAISS vector store)

from sentence_transformers import SentenceTransformer
import numpy as np

from literature_review.utils.logger import get_logger

logger = get_logger(__name__)

# Default model — will be swapped to IndicBERT/multilingual in Phase 2
DEFAULT_MODEL = "all-MiniLM-L6-v2"
MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:

    def __init__(self, model_name: str = None, multilingual: bool = False):
        if model_name:
            selected = model_name
        elif multilingual:
            selected = MULTILINGUAL_MODEL
        else:
            selected = DEFAULT_MODEL

        logger.info(f"Loading embedding model: {selected}")
        self.model = SentenceTransformer(selected)
        self.model_name = selected

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for a single text."""
        if not text or not text.strip():
            logger.warning("Embedder received empty text.")
            return np.array([])
        return self.model.encode(text)

    def embed_chunks(self, chunks: list[str]) -> list[np.ndarray]:
        """Get embedding vectors for a list of chunks."""
        if not chunks:
            logger.warning("Embedder received empty chunk list.")
            return []
        logger.info(f"Embedding {len(chunks)} chunks with {self.model_name}...")
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        return list(embeddings)
