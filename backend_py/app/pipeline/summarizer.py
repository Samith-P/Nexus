# app/pipeline/summarizer.py
# Stage 5 — Hierarchical summarization
# Fixed: truncates section text to fit model's 1024 token limit.
# CPU-optimized for Ryzen 5 5600H / 14GB RAM.

import os
import requests
from app.pipeline.hf_token_pool import HFTokenPool
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"
MAX_INPUT_WORDS = 400  # distilbart-cnn-12-6 has 1024 token limit (~400 words safe)


class Summarizer:

    def __init__(self, model_name: str = None):
        self.model_name = model_name or DEFAULT_MODEL
        logger.info(f"Using Hugging Face Inference API for model: {self.model_name}")
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"
        self.token_pool = HFTokenPool()
        if not self.token_pool.has_tokens():
            logger.warning("No Hugging Face token found for summarizer.")

    def _safe_truncate(self, text: str, max_words: int = MAX_INPUT_WORDS) -> str:
        """Truncate text to fit within model token limits."""
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return text

    def summarize_text(self, text: str) -> str:
        """Summarize a single text chunk (safely truncated)."""
        if not text or not text.strip():
            logger.warning("Summarizer received empty text.")
            return ""

        # Truncate to safe length
        text = self._safe_truncate(text)
        input_len = len(text.split())

        if input_len < 20:
            logger.warning("Text too short to summarize, returning as-is.")
            return text

        max_len = min(150, input_len // 2 + 20)
        min_len = min(40, input_len // 4)
        
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": max_len,
                "min_length": min_len,
                "do_sample": False
            }
        }

        tokens = self.token_pool.get_primary_tokens() + self.token_pool.get_fallback_tokens()
        if not tokens:
            return ""

        last_error = None
        for idx, token in enumerate(tokens, start=1):
            headers = {"Authorization": f"Bearer {token}"}
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                self.token_pool.mark_result(token, response.status_code)
                if response.status_code in {401, 402, 403, 429, 500, 502, 503, 504}:
                    last_error = RuntimeError(f"status={response.status_code}")
                    continue
                response.raise_for_status()
                result = response.json()
                if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
                    return result[0]["summary_text"]
                logger.error(f"Summarization API returned unexpected format: {result}")
                return ""
            except Exception as e:
                last_error = e
                logger.warning("Summarization request failed token %s/%s: %s", idx, len(tokens), e)

        logger.error(f"Summarization API failed after token retries: {last_error}")
        return ""

    def hierarchical_summarize(self, chunks: list[str]) -> str:
        """Summarize chunks individually, then combine and re-summarize."""
        if not chunks:
            logger.warning("No chunks to summarize.")
            return ""

        logger.info(f"Hierarchical summarization of {len(chunks)} chunks...")

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  Summarizing chunk {i + 1}/{len(chunks)}")
            summary = self.summarize_text(chunk)
            if summary:
                chunk_summaries.append(summary)

        if not chunk_summaries:
            logger.warning("All chunk summaries were empty.")
            return ""

        combined = " ".join(chunk_summaries)
        combined = self._safe_truncate(combined)

        logger.info("Generating final combined summary...")
        return self.summarize_text(combined)

    def summarize_sections(self, sections: dict) -> dict:
        """Summarize each section individually (safely truncated)."""
        section_summaries = {}
        for name, content in sections.items():
            if content and content.strip() and name not in ("title", "others"):
                logger.info(f"Summarizing section: {name}")
                # Truncate long sections to fit model limit
                truncated = self._safe_truncate(content)
                summary = self.summarize_text(truncated)
                if summary:
                    section_summaries[name] = summary
        return section_summaries
