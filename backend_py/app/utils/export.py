# app/utils/export.py
# Report export — generate PDF reports from analysis results.

from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    logger.warning("fpdf2 not installed — PDF export will be unavailable.")


def _safe(text) -> str:
    """Normalize text to latin-1 safe characters for FPDF."""
    if not text:
        return ""
    text = str(text)
    replacements = {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--", "\u2026": "...",
        "\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff",
        "\ufb03": "ffi", "\ufb04": "ffl",
        "\u20d7": "", "\u2303": "", "\u2208": "in",
        "\u2192": "->", "\u2264": "<=", "\u2265": ">=",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("\n", " ").replace("\r", "")
    while "  " in text:
        text = text.replace("  ", " ")
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text.strip()


class ReportExporter:
    """Export literature review results as PDF."""

    def _write(self, pdf, text, h=5):
        """Safe multi_cell wrapper — skips empty text."""
        text = _safe(text)
        if not text:
            logger.debug("Skipping empty export text block.")
            return
        logger.debug("Writing PDF text block chars=%s preview=%r", len(text), text[:120])
        pdf.multi_cell(w=0, h=h, text=text, new_x="LMARGIN", new_y="NEXT")

    def export_pdf(self, result, output_path: str) -> str:
        if not FPDF_AVAILABLE:
            raise ImportError("fpdf2 is required. pip install fpdf2")

        logger.debug(
            "Starting PDF export output_path=%s papers=%s has_comparison_matrix=%s related_works=%s",
            output_path,
            len(result.papers),
            bool(result.comparison_matrix),
            len(result.related_works),
        )

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)

        # ── Title Page ──
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 22)
        pdf.cell(0, 20, "Literature Review Report", ln=True, align="C")
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, _safe(f"Generated: {result.timestamp}"), ln=True, align="C")
        pdf.cell(0, 7, _safe(
            f"Papers analyzed: {len(result.papers)} | "
            f"Processing time: {result.processing_time_seconds:.1f}s"
        ), ln=True, align="C")
        pdf.ln(8)

        # ── Per-Paper ──
        for i, paper in enumerate(result.papers):
            if i > 0:
                pdf.add_page()

            # Title
            logger.debug("Exporting paper index=%s title_preview=%r", i, (paper.title or "")[:120])
            pdf.set_font("Helvetica", "B", 13)
            title = _safe(paper.title[:120]) or f"Paper {i+1}"
            self._write(pdf, f"Paper {i+1}: {title}", h=7)
            pdf.ln(4)

            # Summary
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Summary", ln=True)
            pdf.set_font("Helvetica", "", 9)
            self._write(pdf, paper.summary or "No summary available.")
            pdf.ln(4)

            # Section Summaries
            if paper.section_summaries:
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 7, "Section Summaries", ln=True)
                for sec, summ in paper.section_summaries.items():
                    if summ and summ.strip():
                        logger.debug("Exporting section summary section=%s chars=%s", sec, len(summ))
                        pdf.set_font("Helvetica", "BI", 10)
                        pdf.cell(0, 6, _safe(sec.title()), ln=True)
                        pdf.set_font("Helvetica", "", 9)
                        self._write(pdf, summ)
                        pdf.ln(2)
                pdf.ln(3)

            # Key Insights
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Key Insights", ln=True)
            for cat in ["contributions", "methods", "results"]:
                items = getattr(paper.insights, cat, [])
                items = [x for x in items if x and x.strip()]
                if items:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.cell(0, 6, _safe(f"{cat.title()}:"), ln=True)
                    pdf.set_font("Helvetica", "", 9)
                    for item in items:
                        logger.debug("Exporting insight category=%s chars=%s preview=%r", cat, len(item), item[:100])
                        self._write(pdf, f"- {item[:200]}")
            pdf.ln(4)

            # Gaps
            gaps = [g for g in (paper.gaps or []) if g and g.strip()]
            if gaps:
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 7, "Research Gaps", ln=True)
                pdf.set_font("Helvetica", "", 9)
                for gap in gaps:
                    logger.debug("Exporting paper gap chars=%s preview=%r", len(gap), gap[:100])
                    self._write(pdf, f"- {gap[:200]}")
                pdf.ln(3)

        # ── Comparison Matrix ──
        if result.comparison_matrix and len(result.papers) > 1:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Cross-Paper Comparison", ln=True)
            pdf.ln(3)

            dm = [m for m in (result.comparison_matrix.differing_methods or []) if m and m.strip()]
            if dm:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, "Differing Methods:", ln=True)
                pdf.set_font("Helvetica", "", 9)
                for m in dm:
                    logger.debug("Exporting differing method chars=%s preview=%r", len(m), m[:100])
                    self._write(pdf, f"- {m[:150]}")
                pdf.ln(3)

            cm = [m for m in (result.comparison_matrix.common_methods or []) if m and m.strip()]
            if cm:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, "Common Methods:", ln=True)
                pdf.set_font("Helvetica", "", 9)
                logger.debug("Exporting common methods count=%s", len(cm))
                self._write(pdf, ", ".join(cm))
                pdf.ln(3)

        # ── Common Themes ──
        themes = [t for t in (result.common_themes or []) if t and t.strip()]
        if themes:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Common Themes", ln=True)
            pdf.set_font("Helvetica", "", 9)
            for theme in themes:
                logger.debug("Exporting common theme chars=%s preview=%r", len(theme), theme[:100])
                self._write(pdf, f"- {theme[:150]}")
            pdf.ln(3)

        # ── Research Gaps ──
        rgaps = [g for g in (result.research_gaps or []) if g and g.strip()]
        if rgaps:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Aggregated Research Gaps", ln=True)
            pdf.set_font("Helvetica", "", 9)
            for gap in rgaps:
                logger.debug("Exporting aggregated research gap chars=%s preview=%r", len(gap), gap[:100])
                self._write(pdf, f"- {gap[:200]}")

        # ── Related Works ──
        if result.related_works:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Related Works", ln=True)
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 9)
            for rw in result.related_works:
                authors = ", ".join(rw.authors[:3]) if rw.authors else "Unknown"
                year = f" ({rw.year})" if rw.year else ""
                logger.debug("Exporting related work title=%r source=%s", (rw.title or "")[:120], rw.source)
                self._write(pdf, f"- {rw.title[:120]}{year}")
                self._write(pdf, f"  Authors: {authors} | Source: {rw.source}")
                pdf.ln(2)

        pdf.output(output_path)
        logger.info(f"PDF report saved to {output_path}")
        logger.debug("PDF export completed successfully output_path=%s", output_path)
        return output_path
