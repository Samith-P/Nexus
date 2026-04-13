# app/pipeline/insight_extractor.py
# Stage 6 — Insight extraction (CLEANED — removed BERT-specific boost_methods)
# Uses FLAN-T5 with improved, generic prompts.
# Supports shared model instance to save RAM on CPU-only systems.

from transformers import pipeline as hf_pipeline
import re

from literature_review.utils.logger import get_logger

logger = get_logger(__name__)


class InsightExtractor:

    def __init__(self, model_name: str = "google/flan-t5-base", shared_model=None):
        """Initialize insight extractor.

        Args:
            model_name: HuggingFace model ID.
            shared_model: Optional pre-loaded pipeline to share with GapDetector (saves RAM).
        """
        if shared_model is not None:
            logger.info("Using shared FLAN-T5 model for insight extraction.")
            self.model = shared_model
        else:
            logger.info(f"Loading insight extraction model: {model_name}")
            self.model = hf_pipeline("text2text-generation", model=model_name)

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

        try:
            output = self.model(prompt, max_length=150, do_sample=False)
            raw = output[0]["generated_text"]
            results = self._clean_output(raw)
            logger.info(f"Extracted {len(results)} {task} insights.")
            return results[:5]
        except Exception as e:
            logger.error(f"Insight extraction failed for {task}: {e}")
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
