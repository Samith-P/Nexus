# app/pipeline/insight_extractor.py
# Stage 6 — Insight extraction (CLEANED — removed BERT-specific boost_methods)
# Uses FLAN-T5 with improved, generic prompts.
# Supports shared model instance to save RAM on CPU-only systems.

import os
import requests
import re

from app.utils.logger import get_logger

logger = get_logger(__name__)


class InsightExtractor:

    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B-Instruct", shared_model=None):
        """Initialize API-based insight extractor."""
        self.model_name = model_name
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.tokens = self._load_hf_tokens()
        if not self.tokens:
            logger.warning("No Hugging Face token found. Set HF_TOKEN_1..HF_TOKEN_5 or HF_TOKEN.")

    def _load_hf_tokens(self) -> list[str]:
        tokens: list[str] = []

        for idx in range(1, 6):
            token = os.getenv(f"HF_TOKEN_{idx}", "").strip()
            if token:
                tokens.append(token)

        legacy_token = os.getenv("HF_TOKEN", "").strip()
        if legacy_token:
            tokens.append(legacy_token)

        csv_tokens = os.getenv("HF_TOKENS", "").strip()
        if csv_tokens:
            tokens.extend([item.strip() for item in csv_tokens.split(",") if item.strip()])

        return list(dict.fromkeys(tokens))

    def _build_prompt(self, text: str, task: str) -> str:
        prompts = {
            "contributions": (
                "Read the following research text and extract the key contributions "
                "of this work. Return 2-4 short bullet points.\n\nText:\n"
            ),
            "methods": (
                "Read the following research text and list only the specific methods, "
                "techniques, or algorithms used. Return 2-4 short names/phrases.\n\nText:\n"
            ),
            "results": (
                "Read the following research text and extract the key results with "
                "any numbers or metrics mentioned. Return 2-4 short points.\n\nText:\n"
            ),
        }
        prefix = prompts.get(task, "Extract key points from:\n\nText:\n")
        return prefix + text

    def _clean_output(self, raw: str) -> list[str]:
        """Parse model output into clean list of insights."""
        lines = re.split(r"\n|•|-|;\s|\.\s", raw)
        cleaned = []
        for line in lines:
            line = line.strip()
            if len(line) < 5 or len(line) > 120:
                continue
            line = re.sub(r"\s+", " ", line)
            cleaned.append(line)
        return list(dict.fromkeys(cleaned))

    def extract(self, text: str, task: str) -> list[str]:
        """Extract insights for a given task (contributions/methods/results)."""
        if not text or not text.strip():
            logger.warning(f"Empty text for task '{task}', skipping.")
            return []

        safe_text = text[:1200]
        prompt = self._build_prompt(safe_text, task)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }

        if not self.tokens:
            return []

        last_error = None
        for idx, token in enumerate(self.tokens, start=1):
            headers = {"Authorization": f"Bearer {token}"}
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                if response.status_code in {401, 403}:
                    logger.warning(
                        "Insight extraction auth failed for %s with token %s/%s; trying next token.",
                        task,
                        idx,
                        len(self.tokens),
                    )
                    last_error = RuntimeError(f"unauthorized status={response.status_code}")
                    continue

                response.raise_for_status()
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    raw = result["choices"][0]["message"]["content"]
                    results = self._clean_output(raw)
                    logger.info(f"Extracted {len(results)} {task} insights.")
                    return results[:5]

                logger.error(f"Insight extraction API returned unexpected format: {result}")
                return []
            except Exception as e:
                last_error = e
                logger.warning(
                    "Insight extraction request failed for %s with token %s/%s: %s",
                    task,
                    idx,
                    len(self.tokens),
                    e,
                )

        logger.error(f"Insight extraction API failed for {task} after {len(self.tokens)} tokens: {last_error}")
        return []

    def extract_all(self, sections: dict) -> dict:
        """Extract contributions, methods, and results from paper sections."""
        abstract = sections.get("abstract", "")
        methodology = sections.get("methodology", "")
        results = sections.get("results", "")

        insights = {
            "contributions": self.extract(abstract, "contributions"),
            "methods": self.extract(methodology, "methods"),
            "results": self.extract(results, "results"),
        }

        logger.info(
            f"Total insights: {sum(len(v) for v in insights.values())} "
            f"(contributions={len(insights['contributions'])}, "
            f"methods={len(insights['methods'])}, "
            f"results={len(insights['results'])})"
        )
        return insights
