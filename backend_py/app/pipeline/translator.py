# app/pipeline/translator.py
# Translation output layer — converts analysis results to user's preferred language.
# Uses IndicBART for Indian languages, passes through for English.
# Translates: summary, section_summaries, insights, gaps

from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy import to avoid loading IndicBART until needed
_indicbart = None


def _get_indicbart():
    global _indicbart
    if _indicbart is None:
        from app.pipeline.multilingual import IndicBARTSummarizer
        _indicbart = IndicBARTSummarizer()
    return _indicbart


class TranslationOutputLayer:
    """Translates analysis output to the user's preferred language.

    For English: pass-through (no translation needed).
    For Indian languages: uses IndicBART to translate each field.
    """

    def __init__(self, target_lang: str = "en"):
        self.target_lang = target_lang
        self.translator = None

        if target_lang != "en":
            logger.info(f"Translation output layer active: target={target_lang}")
            self.translator = _get_indicbart()
        else:
            logger.info("Translation output layer: English (pass-through)")

    def _translate_text(self, text: str) -> str:
        """Translate a single text string."""
        if not text or not text.strip():
            return text
        if self.target_lang == "en" or self.translator is None:
            return text

        try:
            return self.translator.translate(text, target_lang=self.target_lang)
        except Exception as e:
            logger.warning(f"Translation failed, returning original: {e}")
            return text

    def _translate_list(self, items: list[str]) -> list[str]:
        """Translate a list of strings."""
        return [self._translate_text(item) for item in items]

    def translate_analysis(self, analysis) -> None:
        """Translate a PaperAnalysis object in-place.

        Translates: summary, section_summaries, insights, gaps.
        Does NOT translate: title, sections (raw text), evidence_spans.
        """
        if self.target_lang == "en":
            return  # No-op for English

        logger.info(f"Translating analysis to {self.target_lang}...")

        # Translate summary
        if analysis.summary:
            analysis.summary = self._translate_text(analysis.summary)

        # Translate section summaries
        translated_summaries = {}
        for section, summary in analysis.section_summaries.items():
            translated_summaries[section] = self._translate_text(summary)
        analysis.section_summaries = translated_summaries

        # Translate insights
        analysis.insights.contributions = self._translate_list(
            analysis.insights.contributions
        )
        analysis.insights.methods = self._translate_list(
            analysis.insights.methods
        )
        analysis.insights.results = self._translate_list(
            analysis.insights.results
        )

        # Translate gaps
        analysis.gaps = self._translate_list(analysis.gaps)

        logger.info(f"Translation to {self.target_lang} complete.")

    def translate_review_result(self, result) -> None:
        """Translate a full LiteratureReviewResult in-place."""
        if self.target_lang == "en":
            return

        logger.info(f"Translating full review result to {self.target_lang}...")

        for paper in result.papers:
            self.translate_analysis(paper)

        result.common_themes = self._translate_list(result.common_themes)
        result.research_gaps = self._translate_list(result.research_gaps)

        logger.info("Full review translation complete.")
