# app/utils/export.py
# Report export — generate PDF reports from analysis results

from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from fpdf import FPDF

    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    logger.warning("fpdf2 not installed — PDF export will be unavailable.")


class ReportExporter:
    """Export literature review results as PDF."""

    def export_pdf(self, result, output_path: str) -> str:
        """Generate a formatted PDF report.

        Args:
            result: LiteratureReviewResult object.
            output_path: Path to save the PDF file.

        Returns:
            Path to the generated PDF.
        """
        if not FPDF_AVAILABLE:
            raise ImportError("fpdf2 is required for PDF export. Install: pip install fpdf2")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 15, "Literature Review Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated: {result.timestamp}", ln=True, align="C")
        pdf.cell(
            0, 8,
            f"Papers analyzed: {len(result.papers)} | "
            f"Processing time: {result.processing_time_seconds:.1f}s",
            ln=True, align="C",
        )
        pdf.ln(10)

        # Per-paper analysis
        for i, paper in enumerate(result.papers):
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, f"Paper {i + 1}: {paper.title[:80]}", ln=True)
            pdf.ln(3)

            # Summary
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Summary", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, paper.summary or "No summary available.")
            pdf.ln(3)

            # Section summaries
            if paper.section_summaries:
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 7, "Section Summaries", ln=True)
                for section, summary in paper.section_summaries.items():
                    pdf.set_font("Helvetica", "BI", 9)
                    pdf.cell(0, 6, f"  {section.title()}", ln=True)
                    pdf.set_font("Helvetica", "", 9)
                    pdf.multi_cell(0, 5, f"  {summary}")
                pdf.ln(3)

            # Insights
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Key Insights", ln=True)
            pdf.set_font("Helvetica", "", 9)

            for category in ["contributions", "methods", "results"]:
                items = getattr(paper.insights, category, [])
                if items:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.cell(0, 6, f"  {category.title()}:", ln=True)
                    pdf.set_font("Helvetica", "", 9)
                    for item in items:
                        pdf.cell(0, 5, f"    - {item}", ln=True)
            pdf.ln(3)

            # Gaps
            if paper.gaps:
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 7, "Research Gaps", ln=True)
                pdf.set_font("Helvetica", "", 9)
                for gap in paper.gaps:
                    pdf.cell(0, 5, f"  - {gap}", ln=True)
            pdf.ln(5)

        # Comparison matrix
        if result.comparison_matrix and len(result.papers) > 1:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Cross-Paper Comparison", ln=True)
            pdf.ln(3)

            if result.comparison_matrix.common_methods:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, "Common Methods:", ln=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.multi_cell(0, 5, ", ".join(result.comparison_matrix.common_methods))
                pdf.ln(3)

            if result.common_themes:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, "Common Themes:", ln=True)
                pdf.set_font("Helvetica", "", 9)
                for theme in result.common_themes:
                    pdf.cell(0, 5, f"  - {theme}", ln=True)

        # Related works
        if result.related_works:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Related Works", ln=True)
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 9)
            for rw in result.related_works:
                authors = ", ".join(rw.authors[:3]) if rw.authors else "Unknown"
                year = f" ({rw.year})" if rw.year else ""
                pdf.cell(0, 6, f"- {rw.title}{year}", ln=True)
                pdf.cell(0, 5, f"    Authors: {authors} | Source: {rw.source}", ln=True)
                pdf.ln(2)

        # Aggregated gaps
        if result.research_gaps:
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Aggregated Research Gaps", ln=True)
            pdf.set_font("Helvetica", "", 9)
            for gap in result.research_gaps:
                pdf.cell(0, 5, f"  - {gap}", ln=True)

        pdf.output(output_path)
        logger.info(f"PDF report saved to {output_path}")
        return output_path
