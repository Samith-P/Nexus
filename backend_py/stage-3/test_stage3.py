# stage-3/test_stage3.py

import sys
import os

# Add previous stages
sys.path.append(os.path.abspath("../stage-1"))
sys.path.append(os.path.abspath("../stage-2"))

from pdf_parser import PDFParser
from text_cleaner import TextCleaner
from section_detector import SectionDetector


def run_pipeline(pdf_path):
    print("\n🚀 Running Stage 1 → 2 → 3 Pipeline...\n")

    # Stage 1
    parser = PDFParser(pdf_path)
    parsed = parser.extract_text()

    # Stage 2
    cleaner = TextCleaner(parsed["full_text"])
    clean_text = cleaner.clean()

    # Stage 3
    detector = SectionDetector(clean_text)
    sections = detector.detect_sections()

    print("\n📊 DETECTED SECTIONS:\n")

    for key, value in sections.items():
        print(f"\n===== {key.upper()} =====")
        print(value[:300])  # preview


if __name__ == "__main__":
    pdf_path = r"../data/sample_papers/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf"
    run_pipeline(pdf_path)