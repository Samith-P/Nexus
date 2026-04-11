# tests/test_pipeline.py
# Unit tests for the Literature Review Engine pipeline stages.
# Run: python -m pytest tests/test_pipeline.py -v

import os
import pytest


# ============================================================
# Stage 2: TextCleaner
# ============================================================
class TestTextCleaner:

    def test_remove_emails(self):
        from app.pipeline.text_cleaner import TextCleaner
        c = TextCleaner("Contact us at test@example.com for more info.")
        c.remove_emails()
        assert "test@example.com" not in c.text

    def test_remove_urls(self):
        from app.pipeline.text_cleaner import TextCleaner
        c = TextCleaner("Visit https://arxiv.org/abs/1234 for the paper.")
        c.remove_urls()
        assert "https://arxiv.org" not in c.text

    def test_remove_dois(self):
        from app.pipeline.text_cleaner import TextCleaner
        c = TextCleaner("doi: 10.1234/test.5678 is the reference.")
        c.remove_dois()
        assert "10.1234" not in c.text

    def test_fix_hyphenated_words(self):
        from app.pipeline.text_cleaner import TextCleaner
        c = TextCleaner("This is a hyph-\nenated word.")
        c.fix_hyphenated_words()
        assert "hyphenated" in c.text

    def test_clean_returns_string(self):
        from app.pipeline.text_cleaner import TextCleaner
        c = TextCleaner("Some text with email@test.com and https://url.com")
        result = c.clean()
        assert isinstance(result, str)
        assert "email@test.com" not in result

    def test_empty_input(self):
        from app.pipeline.text_cleaner import TextCleaner
        c = TextCleaner("")
        result = c.clean()
        assert result == ""


# ============================================================
# Stage 3: SectionDetector
# ============================================================
class TestSectionDetector:

    def test_detect_numbered_headings(self):
        from app.pipeline.section_detector import SectionDetector
        text = """Title of Paper\n\nAbstract\nThis is the abstract.\n\n1 Introduction\nIntro text here.\n\n2 Methodology\nMethod text.\n\n3 Results\nResults here.\n\n4 Conclusion\nConclusion text.\n\nReferences\n[1] Ref."""
        d = SectionDetector(text)
        sections = d.detect_sections()
        assert sections["abstract"] != ""
        assert sections["introduction"] != ""
        assert sections["methodology"] != ""
        assert sections["conclusion"] != ""

    def test_detect_caps_headings(self):
        from app.pipeline.section_detector import SectionDetector
        text = """Title\n\nABSTRACT\nAbstract content.\n\nINTRODUCTION\nIntro content.\n\nMETHODOLOGY\nMethod content.\n\nCONCLUSION\nConclusion content."""
        d = SectionDetector(text)
        sections = d.detect_sections()
        assert sections["abstract"] != ""
        assert sections["introduction"] != ""

    def test_stops_at_references(self):
        from app.pipeline.section_detector import SectionDetector
        text = """Abstract\nAbstract text.\n\n1 Introduction\nIntro.\n\n6 Conclusion\nConclusion text only.\n\nReferences\n[1] Long reference list that should not be in conclusion."""
        d = SectionDetector(text)
        sections = d.detect_sections()
        # Conclusion should NOT contain references
        assert "[1] Long reference" not in sections.get("conclusion", "")

    def test_empty_input(self):
        from app.pipeline.section_detector import SectionDetector
        d = SectionDetector("")
        sections = d.detect_sections()
        assert isinstance(sections, dict)

    def test_no_headings_fallback(self):
        from app.pipeline.section_detector import SectionDetector
        d = SectionDetector("This is just plain text with no headings at all.")
        sections = d.detect_sections()
        assert sections["others"] != ""


# ============================================================
# Stage 4: TextChunker
# ============================================================
class TestChunker:

    def test_basic_chunking(self):
        from app.pipeline.chunker import TextChunker
        chunker = TextChunker(max_words=50)
        text = "This is sentence one. " * 20 + "This is sentence two. " * 20
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0

    def test_empty_input(self):
        from app.pipeline.chunker import TextChunker
        chunker = TextChunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        from app.pipeline.chunker import TextChunker
        chunker = TextChunker(max_words=400)
        text = "Short text. Only two sentences."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1


# ============================================================
# Schemas
# ============================================================
class TestSchemas:

    def test_paper_analysis_defaults(self):
        from app.models.schemas import PaperAnalysis
        pa = PaperAnalysis()
        assert pa.title == ""
        assert pa.gaps == []
        assert pa.insights.contributions == []

    def test_literature_review_result(self):
        from app.models.schemas import LiteratureReviewResult
        r = LiteratureReviewResult()
        assert r.papers == []
        assert r.processing_time_seconds == 0.0
        assert r.timestamp != ""

    def test_review_status_response(self):
        from app.models.schemas import ReviewStatusResponse
        r = ReviewStatusResponse(task_id="abc123", status="queued")
        assert r.task_id == "abc123"
        assert r.result is None


# ============================================================
# Vector Store
# ============================================================
class TestVectorStore:

    def test_add_and_search(self):
        import numpy as np
        from app.storage.vector_store import VectorStore
        store = VectorStore(dimension=4)
        embeddings = [np.array([1, 0, 0, 0], dtype=np.float32), np.array([0, 1, 0, 0], dtype=np.float32)]
        metadata = [{"text": "chunk1"}, {"text": "chunk2"}]
        store.add(embeddings, metadata)
        assert store.size == 2

        results = store.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "chunk1"

    def test_empty_store(self):
        import numpy as np
        from app.storage.vector_store import VectorStore
        store = VectorStore(dimension=4)
        results = store.search(np.array([1, 0, 0, 0], dtype=np.float32))
        assert results == []

    def test_clear(self):
        import numpy as np
        from app.storage.vector_store import VectorStore
        store = VectorStore(dimension=4)
        store.add([np.array([1, 0, 0, 0], dtype=np.float32)], [{"text": "a"}])
        assert store.size == 1
        store.clear()
        assert store.size == 0


# ============================================================
# Integrations
# ============================================================
class TestCrossRef:

    def test_extract_dois(self):
        from app.integrations.crossref import CrossRefClient
        text = "As shown by Smith (doi: 10.1234/test.5678) and Jones (10.5555/abc.123)."
        dois = CrossRefClient.extract_dois_from_text(text)
        assert len(dois) == 2
        assert "10.1234/test.5678" in dois


class TestMultilingual:

    def test_supported_languages(self):
        from app.pipeline.multilingual import SUPPORTED_LANGUAGES
        assert "en" in SUPPORTED_LANGUAGES
        assert "hi" in SUPPORTED_LANGUAGES
        assert "te" in SUPPORTED_LANGUAGES
        assert "ur" in SUPPORTED_LANGUAGES
        assert "sa" in SUPPORTED_LANGUAGES

    def test_indicbart_lang_codes(self):
        from app.pipeline.multilingual import INDICBART_LANG_CODES
        assert INDICBART_LANG_CODES["hi"] == "<2hi>"
        assert INDICBART_LANG_CODES["te"] == "<2te>"


class TestTranslator:

    def test_english_passthrough(self):
        from app.pipeline.translator import TranslationOutputLayer
        t = TranslationOutputLayer(target_lang="en")
        assert t.translator is None  # No model loaded for English
