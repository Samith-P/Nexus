# app/pipeline/text_cleaner.py
# Stage 2 — Text cleaning (GENERALIZED — removed BERT-paper-specific patterns)
# Improved: better academic header/footer removal, handles conference lines.

import re

from literature_review.utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:

    def __init__(self, text: str):
        if not text or not text.strip():
            logger.warning("TextCleaner received empty input.")
        self.text = text

    def remove_emails(self):
        self.text = re.sub(r"\S+@\S+", "", self.text)

    def remove_urls(self):
        self.text = re.sub(r"https?://\S+", "", self.text)

    def remove_dois(self):
        self.text = re.sub(r"doi:\s*\S+", "", self.text, flags=re.IGNORECASE)

    def remove_page_numbers(self):
        self.text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", self.text)

    def remove_headers_footers(self):
        """Generic removal of common academic headers/footers and conference lines."""
        patterns = [
            r"Proceedings of .*",
            r"©\s*\d{4}.*",
            r"c⃝\s*\d{4}.*",                        # special copyright symbol
            r"Published by .*",
            r"arXiv:\S+",
            r"^\s*Page\s+\d+\s+of\s+\d+\s*$",
            r"Association for Computational Linguistics.*",
            # Conference location/date lines (generic)
            r"^[A-Z][a-z]+,\s*[A-Z][a-z]+,?\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}.*\d{4}.*$",
            # Page numbers like "4171"
            r"^\d{4,5}\s*$",
        ]
        for pattern in patterns:
            self.text = re.sub(pattern, "", self.text, flags=re.IGNORECASE | re.MULTILINE)

    def fix_hyphenated_words(self):
        self.text = re.sub(r"-\n", "", self.text)

    def fix_line_breaks(self):
        self.text = re.sub(r"\n{3,}", "\n\n", self.text)

    def normalize_spaces(self):
        self.text = re.sub(r"[ \t]+", " ", self.text)
        self.text = self.text.strip()

    def clean(self) -> str:
        """Run all cleaning steps and return cleaned text."""
        self.remove_emails()
        self.remove_urls()
        self.remove_dois()
        self.remove_page_numbers()
        self.remove_headers_footers()
        self.fix_hyphenated_words()
        self.fix_line_breaks()
        self.normalize_spaces()

        if not self.text.strip():
            logger.warning("Text is empty after cleaning.")

        logger.info(f"Cleaned text: {len(self.text)} chars remaining.")
        return self.text
