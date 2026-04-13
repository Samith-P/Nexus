"""Quick test: generate PDF from mock 3-paper result to debug export."""
import sys, json
sys.path.insert(0, ".")

from app.models.schemas import (
    LiteratureReviewResult, PaperAnalysis, ComparisonMatrix,
    ComparisonEntry, InsightResult, EvidenceSpan
)
from app.utils.export import ReportExporter
from datetime import datetime, timezone

papers = []
for title in ["Paper A: BERT", "Paper B: Attention", "Paper C: GPT-3"]:
    papers.append(PaperAnalysis(
        title=title,
        sections={"title": title, "abstract": "Abstract text.", "introduction": "Intro.",
                  "methodology": "Methods.", "results": "Results.", "conclusion": "Conclusion."},
        summary=f"Summary of {title} with important findings.",
        section_summaries={"abstract": "Abstract summary.", "introduction": "Intro summary.",
                           "methodology": "Method summary.", "results": "Results summary.",
                           "conclusion": "Conclusion summary."},
        insights=InsightResult(
            contributions=["Contribution 1", "Contribution 2"],
            methods=["Method A", "Method B"],
            results=["Result X", "Result Y"]
        ),
        gaps=["Gap 1", "Gap 2"],
        evidence_spans=[EvidenceSpan(section="abstract", text="Evidence text.")]
    ))

result = LiteratureReviewResult(
    papers=papers,
    comparison_matrix=ComparisonMatrix(
        entries=[
            ComparisonEntry(paper_title="Paper A", methods=["GLUE"], results=["BERT fine", "tuning"], gaps=[]),
            ComparisonEntry(paper_title="Paper B", methods=["BLEU"], results=["28.4 BLEU"], gaps=["Gap X"]),
            ComparisonEntry(paper_title="Paper C", methods=[], results=["Few-shot"], gaps=["Gap Y"]),
        ],
        common_methods=["Transformer"],
        differing_methods=["GLUE vs BLEU", "Fine-tune vs few-shot"]
    ),
    common_themes=["Pre-training is key", "Attention mechanisms"],
    research_gaps=["Computational cost", "Cross-lingual gaps", "Deployment constraints"],
    related_works=[],
    processing_time_seconds=300.0,
    timestamp=datetime.now(timezone.utc).isoformat()
)

exporter = ReportExporter()
exporter.export_pdf(result, "test_3paper.pdf")
print("SUCCESS! PDF saved to test_3paper.pdf")
