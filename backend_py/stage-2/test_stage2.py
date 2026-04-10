# stage-2/test_stage2.py

import sys
import os

# Add stage-1 path
sys.path.append(os.path.abspath("../stage-1"))

from pdf_parser import PDFParser
from text_cleaner import TextCleaner


def run_pipeline(pdf_path):
    print("\n🚀 Running Stage 1 + Stage 2 Pipeline...\n")

    # Stage 1
    parser = PDFParser(pdf_path)
    parsed = parser.extract_text()

    raw_text = parsed["full_text"]

    print("🔴 RAW TEXT (first 300 chars):\n")
    print(raw_text[:300])

    # Stage 2
    cleaner = TextCleaner(raw_text)
    clean_text = cleaner.clean()

    print("\n🟢 CLEANED TEXT (first 300 chars):\n")
    print(clean_text[:300])


if __name__ == "__main__":
    pdf_path = r"../data/sample_papers/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf"
    run_pipeline(pdf_path)