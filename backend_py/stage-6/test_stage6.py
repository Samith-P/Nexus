# stage-6/test_stage6.py

import sys
import os

# Add all previous stages
sys.path.append(os.path.abspath("../stage-1"))
sys.path.append(os.path.abspath("../stage-2"))
sys.path.append(os.path.abspath("../stage-3"))
sys.path.append(os.path.abspath("../stage-4"))
sys.path.append(os.path.abspath("../stage-5"))

from pdf_parser import PDFParser
from text_cleaner import TextCleaner
from section_detector import SectionDetector
from chunker import TextChunker
from summarizer import Summarizer

from insight_extractor import InsightExtractor


def run_pipeline(pdf_path):
    print("\n🚀 Running Stage 1 → 6 Pipeline...\n")

    # Stage 1
    parser = PDFParser(pdf_path)
    parsed = parser.extract_text()

    # Stage 2
    cleaner = TextCleaner(parsed["full_text"])
    clean_text = cleaner.clean()

    # Stage 3
    detector = SectionDetector(clean_text)
    sections = detector.detect_sections()

    # Stage 4
    chunker = TextChunker()

    # Stage 5
    summarizer = Summarizer()

    # Stage 6
    extractor = InsightExtractor()

    summaries = {}

    print("\n🧠 GENERATING SECTION SUMMARIES...\n")

    for key in ["abstract", "introduction", "methodology", "results"]:
        if sections.get(key):
            print(f"📌 Processing {key.upper()}")

            chunks = chunker.chunk_text(sections[key])
            chunks = chunks[:3]  # limit for speed

            summary = summarizer.hierarchical_summarize(chunks)
            summaries[key] = summary

            print(f"Summary Preview:\n{summary[:200]}\n")

    print("\n🔍 EXTRACTING INSIGHTS...\n")

    insights = {
        "contributions": extractor.extract(
            summaries.get("abstract", "") + " " + summaries.get("introduction", ""),
            "contributions"
        ),
        "methods": extractor.extract(
            sections.get("methodology", ""),
            "methods"
        ),
        "results": extractor.extract(
            summaries.get("results", ""),
            "results"
        )
    }

    print("\n📊 FINAL INSIGHTS:\n")

    for key, value in insights.items():
        print(f"\n🔹 {key.upper()}:")
        for item in value:
            print(f"- {item}")


if __name__ == "__main__":
    pdf_path = r"../data/sample_papers/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf"
    run_pipeline(pdf_path)