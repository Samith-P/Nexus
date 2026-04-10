import sys
import os

# Add previous stages
sys.path.append(os.path.abspath("../stage-1"))
sys.path.append(os.path.abspath("../stage-2"))
sys.path.append(os.path.abspath("../stage-3"))
sys.path.append(os.path.abspath("../stage-4"))
sys.path.append(os.path.abspath("../stage-6"))

from pdf_parser import PDFParser
from text_cleaner import TextCleaner
from section_detector import SectionDetector
from chunker import TextChunker
from insight_extractor import InsightExtractor

from gap_detector import GapDetector


def run_pipeline(pdf_path):
    print("\n🚀 Running Stage 1 → 7 Pipeline...\n")

    # Stage 1
    print("🔹 Stage 1: Parsing PDF...")
    parser = PDFParser(pdf_path)
    parsed = parser.extract_text()

    # Stage 2
    print("\n🔹 Stage 2: Cleaning text...")
    cleaner = TextCleaner(parsed["full_text"])
    clean_text = cleaner.clean()

    # Stage 3
    print("\n🔹 Stage 3: Detecting sections...")
    detector = SectionDetector(clean_text)
    sections = detector.detect_sections()

    print("\n📊 Sections Found:")
    for k, v in sections.items():
        print(f"{k}: {len(v)} chars")

    # Stage 4
    chunker = TextChunker()

    # Stage 6
    print("\n🔹 Stage 6: Extracting insights...")
    extractor = InsightExtractor()

    # Stage 7
    gap_detector = GapDetector()

    print("\n🧠 Preparing input...")

    # 🔥 SAFE SMALL TEXT
    abstract = " ".join(chunker.chunk_text(sections.get("abstract", ""))[:1])
    methodology = " ".join(chunker.chunk_text(sections.get("methodology", ""))[:1])
    results = " ".join(chunker.chunk_text(sections.get("results", ""))[:1])

    print("\n📌 Abstract Preview:", abstract[:150])
    print("\n📌 Methodology Preview:", methodology[:150])
    print("\n📌 Results Preview:", results[:150])

    print("\n🔍 Extracting insights...")

    insights = {
        "contributions": extractor.extract(abstract, "contributions"),
        "methods": extractor.extract(methodology, "methods"),
        "results": extractor.extract(results, "results")
    }

    print("\n📊 Stage 6 Output:")
    for key, value in insights.items():
        print(f"\n🔹 {key.upper()}: {value}")

    print("\n🧠 Detecting research gaps...")

    # 🔥 CRITICAL FIX → ADD CONTEXT
    extra_context = abstract + " " + methodology[:500]

    gaps = gap_detector.detect_gaps(insights, extra_context)

    print("\n🚨 FINAL RESEARCH GAPS:")

    if not gaps:
        print("❌ No gaps detected.")
    else:
        for gap in gaps:
            print(f"- {gap}")


if __name__ == "__main__":
    pdf_path = r"../data/sample_papers/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf"
    run_pipeline(pdf_path)