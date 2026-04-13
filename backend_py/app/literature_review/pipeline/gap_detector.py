# app/pipeline/gap_detector.py
# Stage 7 — Research gap detection (CLEANED — no hardcoded generic gaps)
# Supports shared model instance to save RAM on CPU-only systems.

from transformers import pipeline as hf_pipeline
import re

from literature_review.utils.logger import get_logger

logger = get_logger(__name__)


class GapDetector:

    def __init__(self, model_name: str = "google/flan-t5-base", shared_model=None):
        """Initialize gap detector.

        Args:
            model_name: HuggingFace model ID.
            shared_model: Optional pre-loaded pipeline to share with InsightExtractor.
        """
        if shared_model is not None:
            logger.info("Using shared FLAN-T5 model for gap detection.")
            self.model = shared_model
        else:
            logger.info(f"Loading gap detection model: {model_name}")
            self.model = hf_pipeline("text2text-generation", model=model_name)

    def _context_aware_rules(self, insights: dict, context: str) -> list[str]:
        """Generate gaps ONLY if there is evidence in the text."""
        gaps = []
        text_lower = context.lower()
        methods_text = " ".join(insights.get("methods", [])).lower()

        if any(kw in methods_text for kw in ["training", "pre-training", "train"]):
            gaps.append("High computational cost due to training process — may limit reproducibility")

        if any(kw in methods_text for kw in ["fine-tuning", "fine-tune"]):
            gaps.append("Requires large labeled datasets for effective fine-tuning")

        if any(kw in methods_text for kw in ["transformer", "attention"]):
            if "long" in text_lower or "sequence length" in text_lower:
                gaps.append("Limited performance on very long sequences due to quadratic attention complexity")

        if "english" in text_lower and not any(
            lang in text_lower for lang in ["multilingual", "hindi", "telugu", "cross-lingual"]
        ):
            gaps.append("Evaluation limited to English — cross-lingual generalization untested")

        if not any(kw in text_lower for kw in ["real-time", "latency", "inference speed", "deployment"]):
            if any(kw in methods_text for kw in ["deep", "neural", "transformer", "bert"]):
                gaps.append("No analysis of real-time inference or deployment constraints")

        return gaps

    def _llm_gaps(self, insights: dict, context: str) -> list[str]:
        """Use LLM to find additional research gaps."""
        combined = ""
        for key, values in insights.items():
            if values:
                combined += f"{key}: " + ", ".join(values) + "\n"
        combined += "\nContext: " + context[:400]

        prompt = (
            "You are an expert research reviewer. Based on the following research findings, "
            "identify 2-3 specific limitations or research gaps. "
            "Do NOT repeat the input. Return short bullet points only.\n\n"
            f"{combined[:600]}"
        )

        try:
            output = self.model(prompt, max_length=120, do_sample=False)
            raw = output[0]["generated_text"].strip()

            lines = re.split(r"\n|•|-", raw)
            gaps = []
            for line in lines:
                line = line.strip()
                if 10 < len(line) < 150:
                    gaps.append(line)
            return gaps[:3]
        except Exception as e:
            logger.error(f"LLM gap detection failed: {e}")
            return []

    def detect_gaps(self, insights: dict, context_text: str) -> list[str]:
        """Detect research gaps using hybrid approach (context-aware rules + LLM)."""
        logger.info("Detecting research gaps (hybrid approach)...")

        rule_gaps = self._context_aware_rules(insights, context_text)
        llm_gaps = self._llm_gaps(insights, context_text)

        all_gaps = rule_gaps + llm_gaps
        all_gaps = list(dict.fromkeys(all_gaps))

        logger.info(f"Found {len(all_gaps)} gaps (rules={len(rule_gaps)}, llm={len(llm_gaps)}).")
        return all_gaps[:6]
