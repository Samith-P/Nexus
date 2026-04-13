from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path


def _clean_text(text: str) -> str:
    t = text or ""
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF.

    Tries PyMuPDF first (best quality), then falls back to pypdf.
    """

    if not pdf_bytes:
        return ""

    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_lines: list[list[str]] = []
        header_footer_counts: dict[str, int] = {}

        for page in doc:
            raw = page.get_text("text", sort=True)  # sort improves reading order
            lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
            page_lines.append(lines)

            # Track potential header/footer noise.
            candidates = []
            if lines:
                candidates.extend(lines[:2])
                candidates.extend(lines[-2:])
            for ln in candidates:
                if len(ln) <= 80:
                    header_footer_counts[ln] = header_footer_counts.get(ln, 0) + 1

        num_pages = len(page_lines)
        repeated: set[str] = set()
        if num_pages >= 3:
            # Consider a line "repeated" if it appears on most pages.
            threshold = max(3, int(0.6 * num_pages))
            repeated = {ln for ln, c in header_footer_counts.items() if c >= threshold}

        parts: list[str] = []
        for lines in page_lines:
            if repeated:
                lines = [ln for ln in lines if ln not in repeated]
            parts.append("\n".join(lines))

        return _clean_text("\n\n".join(parts))
    except Exception:
        pass

    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(BytesIO(pdf_bytes))
        parts = []
        for p in reader.pages:
            try:
                parts.append(p.extract_text() or "")
            except Exception:
                parts.append("")
        return _clean_text("\n".join(parts))
    except Exception:
        return ""


def extract_text_from_any(*, pdf_bytes: bytes | None = None, text: str | None = None) -> str:
    if text and text.strip():
        return _clean_text(text)
    if pdf_bytes:
        return extract_text_from_pdf(pdf_bytes)
    return ""


def extract_text_from_upload(file_bytes: bytes, filename: str | None) -> str:
    name = (filename or "").strip()
    ext = Path(name).suffix.lower() if name else ""

    if ext in {".pdf"} or not ext:
        return extract_text_from_pdf(file_bytes)

    if ext in {".txt"}:
        try:
            return _clean_text(file_bytes.decode("utf-8"))
        except Exception:
            return _clean_text(file_bytes.decode("latin-1", errors="ignore"))

    if ext in {".docx"}:
        try:
            from docx import Document  # type: ignore

            doc = Document(BytesIO(file_bytes))
            parts = [p.text for p in doc.paragraphs if (p.text or "").strip()]
            return _clean_text("\n".join(parts))
        except Exception:
            return ""

    # Fallback: treat as text
    try:
        return _clean_text(file_bytes.decode("utf-8"))
    except Exception:
        return _clean_text(file_bytes.decode("latin-1", errors="ignore"))
