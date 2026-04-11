# app/pipeline/multilingual.py
# Multilingual support layer using IndicBERT + IndicBART
# IndicBERT: ai4bharat/indic-bert — ALBERT-based multilingual embeddings
# IndicBART: ai4bharat/IndicBARTSS — multilingual summarization/generation
#
# CPU-optimized: models loaded lazily, only when non-English language is requested.
# Supported languages: English (en), Hindi (hi), Telugu (te), Urdu (ur), Sanskrit (sa)

import torch
import numpy as np
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ur": "Urdu",
    "sa": "Sanskrit",
    "bn": "Bengali",
    "ta": "Tamil",
    "ml": "Malayalam",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
}

# IndicBART language codes (used in tokenizer)
INDICBART_LANG_CODES = {
    "en": "<2en>",
    "hi": "<2hi>",
    "te": "<2te>",
    "ur": "<2ur>",
    "sa": "<2sa>",
    "bn": "<2bn>",
    "ta": "<2ta>",
    "ml": "<2ml>",
    "kn": "<2kn>",
    "mr": "<2mr>",
    "gu": "<2gu>",
    "pa": "<2pa>",
    "or": "<2or>",
}


class IndicBERTEmbedder:
    """Multilingual embedder using IndicBERT (ai4bharat/indic-bert).

    Generates embeddings for text in Indian languages + English.
    Falls back to mean-pooling of token embeddings.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoModel, AutoTokenizer

            model_name = "ai4bharat/indic-bert"
            logger.info(f"Loading IndicBERT: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self._loaded = True
            logger.info("IndicBERT loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load IndicBERT: {e}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text using IndicBERT."""
        self._load()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling over token embeddings (exclude [CLS], [SEP], [PAD])
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).squeeze().numpy()

        return embedding

    def embed_chunks(self, chunks: list[str]) -> list[np.ndarray]:
        """Get embeddings for multiple text chunks."""
        self._load()
        logger.info(f"Embedding {len(chunks)} chunks with IndicBERT...")
        return [self.get_embedding(chunk) for chunk in chunks]

    @property
    def dimension(self) -> int:
        """IndicBERT (ALBERT-base) embedding dimension."""
        return 768


class IndicBARTSummarizer:
    """Multilingual summarizer using IndicBART (ai4bharat/IndicBARTSS).

    Generates summaries in Indian languages.
    CPU-optimized: uses beam search with limited beams.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_name = "ai4bharat/IndicBARTSS"
            logger.info(f"Loading IndicBART: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.eval()
            self._loaded = True
            logger.info("IndicBART loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load IndicBART: {e}")
            raise

    def summarize(
        self,
        text: str,
        target_lang: str = "hi",
        max_length: int = 150,
    ) -> str:
        """Summarize/translate text to the target language.

        Args:
            text: Input text (any supported language).
            target_lang: Target language code (hi, te, ur, sa, etc.).
            max_length: Maximum output length in tokens.

        Returns:
            Summarized/translated text in target language.
        """
        self._load()

        lang_code = INDICBART_LANG_CODES.get(target_lang, "<2hi>")

        try:
            # Prepend language token
            input_text = f"{lang_code} {text}"

            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=2,  # CPU-friendly beam count
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"IndicBART summarization complete ({target_lang}): {len(result)} chars")
            return result

        except Exception as e:
            logger.error(f"IndicBART summarization failed: {e}")
            return ""

    def translate(self, text: str, target_lang: str = "hi") -> str:
        """Translate text to target language using IndicBART."""
        return self.summarize(text, target_lang=target_lang, max_length=256)
