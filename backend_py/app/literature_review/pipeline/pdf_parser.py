# app/pipeline/pdf_parser.py
# Stage 1 — PDF text extraction using PyMuPDF
# Unchanged from original, just moved into the package.

import fitz  # PyMuPDF

from literature_review.utils.logger import get_logger

logger = get_logger(__name__)


class PDFParser:

    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_text(self) -> dict:
        """Extract text from a PDF file.

        Returns:
            dict with keys: full_text, pages (list of {page_number, text}), total_pages
        """
        try:
            doc = fitz.open(self.file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF: {self.file_path} — {e}")
            raise ValueError(f"Error opening PDF: {e}")

        full_text = []
        pages = []

        for page_num, page in enumerate(doc):
            try:
                text = page.get_text("text").strip()
                pages.append({"page_number": page_num + 1, "text": text})
                full_text.append(text)
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num + 1}: {e}")

        if not full_text:
            logger.error("PDF produced no text — file may be scanned/image-only.")
            raise ValueError("PDF produced no extractable text.")

        logger.info(f"Extracted {len(pages)} pages from {self.file_path}")
        return {
            "full_text": "\n".join(full_text),
            "pages": pages,
            "total_pages": len(pages),
        }
