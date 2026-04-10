# stage-2/text_cleaner.py

import re


class TextCleaner:

    def __init__(self, text: str):
        self.text = text

    def remove_emails(self):
        self.text = re.sub(r"\S+@\S+", "", self.text)

    def remove_page_numbers(self):
        self.text = re.sub(r"\n\d+\n", "\n", self.text)

    def remove_headers(self):
        patterns = [
            r"Proceedings.*",
            r"Association for Computational Linguistics.*",
            r"Minneapolis.*",
            r"June.*"
        ]
        for pattern in patterns:
            self.text = re.sub(pattern, "", self.text)

    def fix_hyphenated_words(self):
        self.text = re.sub(r"-\n", "", self.text)

    def fix_line_breaks(self):
        self.text = re.sub(r"\n{2,}", "\n\n", self.text)        # self.text = re.sub(r"(?<!\.)\n(?!\n)", " ", self.text)

    def normalize_spaces(self):
        self.text = re.sub(r"\s+", " ", self.text).strip()

    def clean(self):
        self.remove_emails()
        self.remove_page_numbers()
        self.remove_headers()
        self.fix_hyphenated_words()
        self.fix_line_breaks()
        self.normalize_spaces()

        return self.text