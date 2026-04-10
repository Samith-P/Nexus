# src/stage1_parser/pdf_parser.py

import fitz  # PyMuPDF


class PDFParser:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_text(self):
        try:
            doc = fitz.open(self.file_path)
        except Exception as e:
            raise Exception(f"Error opening PDF: {e}")

        full_text = []
        pages = []

        for page_num, page in enumerate(doc):
            try:
                text = page.get_text("text")

                # Basic cleaning (minimal for now)
                text = text.strip()

                pages.append({
                    "page_number": page_num + 1,
                    "text": text
                })

                full_text.append(text)

            except Exception as e:
                print(f"[WARNING] Failed on page {page_num + 1}: {e}")

        return {
            "full_text": "\n".join(full_text),
            "pages": pages,
            "total_pages": len(pages)
        }