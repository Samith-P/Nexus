# app/pipeline/chunker.py
# Stage 4a — Sentence-aware chunking (IMPROVED — no more mid-sentence splits)

import nltk
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class TextChunker:

    def __init__(self, max_words: int = 400, overlap_sentences: int = 2):
        self.max_words = max_words
        self.overlap_sentences = overlap_sentences

    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks at sentence boundaries.

        Args:
            text: The text to chunk.

        Returns:
            List of text chunks, each roughly max_words long,
            with overlap_sentences carried over between chunks.
        """
        if not text or not text.strip():
            logger.warning("Chunker received empty text.")
            return []

        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        current_sentences = []
        current_word_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())

            if current_word_count + word_count > self.max_words and current_sentences:
                # Save current chunk
                chunks.append(" ".join(current_sentences))

                # Overlap: carry last N sentences into next chunk
                overlap = current_sentences[-self.overlap_sentences:]
                current_sentences = list(overlap)
                current_word_count = sum(len(s.split()) for s in current_sentences)

            current_sentences.append(sentence)
            current_word_count += word_count

        # Don't forget the last chunk
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        logger.info(f"Created {len(chunks)} sentence-aware chunks (max ~{self.max_words} words each).")
        return chunks
