# app/pipeline/orchestrator.py
# Pipeline orchestrator — single entry point that chains all stages.
# Optimized for CPU-only (Ryzen 5 5600H, 14GB RAM, no GPU):
#   - Shares FLAN-T5 model between InsightExtractor and GapDetector
#   - Lazy loads heavy models only when needed

import time
from typing import Optional
from transformers import pipeline as hf_pipeline

from app.pipeline.pdf_parser import PDFParser
from app.pipeline.text_cleaner import TextCleaner
from app.pipeline.section_detector import SectionDetector
from app.pipeline.chunker import TextChunker
from app.pipeline.embedder import Embedder
from app.pipeline.summarizer import Summarizer
from app.pipeline.insight_extractor import InsightExtractor
from app.pipeline.gap_detector import GapDetector
from app.storage.vector_store import VectorStore
from app.integrations.semantic_scholar import SemanticScholarClient
from app.integrations.crossref import CrossRefClient
from app.models.schemas import (
    PaperAnalysis,
    InsightResult,
    EvidenceSpan,
    LiteratureReviewResult,
    ComparisonMatrix,
    ComparisonEntry,
    RelatedWork,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LiteratureReviewOrchestrator:
    """Orchestrates the full literature review pipeline.

    CPU-optimized: shares FLAN-T5 across insight_extractor and gap_detector.
    """

    def __init__(self):
        logger.info("Initializing pipeline components (CPU-optimized)...")

        # Lightweight — load immediately
        self.chunker = TextChunker(max_words=400, overlap_sentences=2)
        self.embedder = Embedder()
        self.summarizer = Summarizer()

        # Share one FLAN-T5 model between insight extractor and gap detector
        # Saves ~250MB RAM on CPU-only systems
        logger.info("Loading shared FLAN-T5 model for insights + gaps...")
        shared_flan = hf_pipeline("text2text-generation", model="google/flan-t5-base")
        self.insight_extractor = InsightExtractor(shared_model=shared_flan)
        self.gap_detector = GapDetector(shared_model=shared_flan)

        self.vector_store = VectorStore(dimension=384)
        self.scholar = SemanticScholarClient()
        self.crossref = CrossRefClient()

        logger.info("All pipeline components initialized.")

    def analyze_single_paper(self, pdf_path: str) -> PaperAnalysis:
        """Run the full pipeline on a single PDF."""
        logger.info(f"=== Analyzing paper: {pdf_path} ===")

        # Stage 1: Parse PDF
        parser = PDFParser(pdf_path)
        parsed = parser.extract_text()

        # Stage 2: Clean text
        cleaner = TextCleaner(parsed["full_text"])
        clean_text = cleaner.clean()

        # Stage 3: Detect sections
        detector = SectionDetector(clean_text)
        sections = detector.detect_sections()

        # Stage 4: Chunk the full text
        chunks = self.chunker.chunk_text(clean_text)

        # Stage 4b: Embed chunks and store in vector store
        if chunks:
            embeddings = self.embedder.embed_chunks(chunks)
            title = sections.get("title", "Unknown")[:100]
            metadata = [
                {"text": c[:200], "paper_title": title, "index": i}
                for i, c in enumerate(chunks)
            ]
            self.vector_store.add(embeddings, metadata)

        # Stage 5: Summarize (limit chunks to avoid overloading CPU)
        summary = self.summarizer.hierarchical_summarize(chunks[:4])
        section_summaries = self.summarizer.summarize_sections(sections)

        # Stage 6: Extract insights
        insights_dict = self.insight_extractor.extract_all(sections)

        # Stage 7: Detect gaps
        context = sections.get("abstract", "") + " " + sections.get("methodology", "")[:500]
        gaps = self.gap_detector.detect_gaps(insights_dict, context)

        # Build evidence spans
        evidence = []
        for section_name, content in sections.items():
            if content and section_name not in ("title", "others"):
                evidence.append(EvidenceSpan(
                    section=section_name,
                    text=content[:300],
                ))

        return PaperAnalysis(
            title=sections.get("title", "Untitled")[:200],
            sections={k: v[:500] for k, v in sections.items() if v},
            summary=summary,
            section_summaries=section_summaries,
            insights=InsightResult(**insights_dict),
            gaps=gaps,
            evidence_spans=evidence,
        )

    def compare_papers(self, analyses: list[PaperAnalysis]) -> ComparisonMatrix:
        """Generate a comparison matrix from multiple paper analyses."""
        if len(analyses) < 2:
            return None

        entries = []
        all_methods = []
        for analysis in analyses:
            entries.append(ComparisonEntry(
                paper_title=analysis.title[:100],
                methods=analysis.insights.methods,
                results=analysis.insights.results,
                gaps=analysis.gaps[:3],
            ))
            all_methods.extend(analysis.insights.methods)

        method_counts = {}
        for m in all_methods:
            m_lower = m.lower()
            method_counts[m_lower] = method_counts.get(m_lower, 0) + 1

        common = [m for m, c in method_counts.items() if c > 1]
        differing = [m for m, c in method_counts.items() if c == 1]

        return ComparisonMatrix(
            entries=entries,
            common_methods=common,
            differing_methods=differing,
        )

    def find_common_themes(self, analyses: list[PaperAnalysis]) -> list[str]:
        """Identify common themes across papers."""
        themes = []
        if len(analyses) < 2:
            return themes

        all_contribs = []
        for a in analyses:
            all_contribs.extend(a.insights.contributions)

        seen = {}
        for c in all_contribs:
            key = c.lower().strip()
            seen[key] = seen.get(key, 0) + 1

        themes = [k for k, v in seen.items() if v > 1]
        if not themes:
            themes = all_contribs[:3] if all_contribs else ["No common themes detected"]

        return themes

    def fetch_related_works(self, analyses: list[PaperAnalysis]) -> list[RelatedWork]:
        """Fetch related works from Semantic Scholar and CrossRef."""
        related = []
        seen_titles = set()

        for analysis in analyses[:3]:
            title = analysis.title
            if not title or title == "Untitled":
                continue

            scholar_results = self.scholar.search_papers(title, limit=3)
            for r in scholar_results:
                if r["title"].lower() not in seen_titles:
                    seen_titles.add(r["title"].lower())
                    related.append(RelatedWork(**r))

            crossref_results = self.crossref.search_by_title(title, limit=3)
            for r in crossref_results:
                if r["title"].lower() not in seen_titles:
                    seen_titles.add(r["title"].lower())
                    related.append(RelatedWork(
                        title=r["title"],
                        authors=r["authors"],
                        year=r["year"],
                        url=r["url"],
                        source="crossref",
                    ))

        return related[:10]

    def run(
        self,
        pdf_paths: list[str],
        query: Optional[str] = None,
        fetch_related: bool = True,
    ) -> LiteratureReviewResult:
        """Run the complete literature review pipeline.

        Args:
            pdf_paths: List of paths to PDF files.
            query: Optional research query/theme.
            fetch_related: Whether to fetch related works from APIs.

        Returns:
            LiteratureReviewResult with all analyses, comparisons, and gaps.
        """
        start_time = time.time()
        logger.info(f"Starting literature review for {len(pdf_paths)} paper(s)...")

        # Analyze each paper
        analyses = []
        for path in pdf_paths:
            try:
                analysis = self.analyze_single_paper(path)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze {path}: {e}")

        if not analyses:
            logger.error("No papers were successfully analyzed.")
            return LiteratureReviewResult(
                processing_time_seconds=time.time() - start_time
            )

        # Multi-paper comparison
        comparison = None
        common_themes = []
        if len(analyses) > 1:
            comparison = self.compare_papers(analyses)
            common_themes = self.find_common_themes(analyses)

        # Aggregate research gaps
        all_gaps = []
        for a in analyses:
            all_gaps.extend(a.gaps)
        all_gaps = list(dict.fromkeys(all_gaps))

        # Related works
        related_works = []
        if fetch_related:
            try:
                related_works = self.fetch_related_works(analyses)
            except Exception as e:
                logger.warning(f"Related works fetch failed: {e}")

        elapsed = time.time() - start_time
        logger.info(f"Literature review complete in {elapsed:.1f}s.")

        return LiteratureReviewResult(
            papers=analyses,
            comparison_matrix=comparison,
            common_themes=common_themes,
            research_gaps=all_gaps,
            related_works=related_works,
            processing_time_seconds=round(elapsed, 2),
        )
