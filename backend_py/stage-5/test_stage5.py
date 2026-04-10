# stage-5/test_stage5.py

import sys
import os

# Add previous stages
sys.path.append(os.path.abspath("../stage-1"))
sys.path.append(os.path.abspath("../stage-2"))
sys.path.append(os.path.abspath("../stage-3"))
sys.path.append(os.path.abspath("../stage-4"))

from pdf_parser import PDFParser
from text_cleaner import TextCleaner
from section_detector import SectionDetector
from chunker import TextChunker

from summarizer import Summarizer


def run_pipeline(pdf_path):
    print("\n🚀 Running Stage 1 → 5 Pipeline...\n")

    # Stage 1
    parser = PDFParser(pdf_path)
    parsed = parser.extract_text()

    # Stage 2
    cleaner = TextCleaner(parsed["full_text"])
    clean_text = cleaner.clean()

    # Stage 3
    detector = SectionDetector(clean_text)
    sections = detector.detect_sections()

    # Stage 4 (chunking only)
    chunker = TextChunker()

    # Stage 5
    summarizer = Summarizer()

    for section, text in sections.items():
        if not text.strip():
            continue

        print(f"\n📌 SECTION: {section.upper()}")

        chunks = chunker.chunk_text(text)
        chunks = chunks[:5]
        print(f"Chunks: {len(chunks)}")

        final_summary = summarizer.hierarchical_summarize(chunks)

        print("\n🟢 FINAL SUMMARY:\n")
        print(final_summary[:500])


if __name__ == "__main__":
    pdf_path = r"../data/sample_papers/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf"
    run_pipeline(pdf_path)