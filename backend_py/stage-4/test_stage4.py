# stage-4/test_stage4.py

import sys
import os

# Add previous stages
sys.path.append(os.path.abspath("../stage-1"))
sys.path.append(os.path.abspath("../stage-2"))
sys.path.append(os.path.abspath("../stage-3"))

from pdf_parser import PDFParser
from text_cleaner import TextCleaner
from section_detector import SectionDetector

from chunker import TextChunker
from embedder import Embedder


def run_pipeline(pdf_path):
    print("\n🚀 Running Stage 1 → 2 → 3 → 4 Pipeline...\n")

    # ---------------- STAGE 1 ----------------
    print("🔹 Stage 1: PDF Parsing...")
    parser = PDFParser(pdf_path)
    parsed = parser.extract_text()

    print(f"Total Pages: {parsed['total_pages']}")
    print(f"Raw text length: {len(parsed['full_text'])}\n")

    # ---------------- STAGE 2 ----------------
    print("🔹 Stage 2: Text Cleaning...")
    cleaner = TextCleaner(parsed["full_text"])
    clean_text = cleaner.clean()

    print(f"Clean text length: {len(clean_text)}")
    print(f"Preview:\n{clean_text[:200]}\n")

    # ---------------- STAGE 3 ----------------
    print("🔹 Stage 3: Section Detection...")
    detector = SectionDetector(clean_text)
    sections = detector.detect_sections()

    print("\nDetected Sections:")
    for key, value in sections.items():
        print(f"{key}: length = {len(value)}")

    # ---------------- STAGE 4 ----------------
    print("\n🔹 Stage 4: Chunking + Embedding...")

    chunker = TextChunker()
    embedder = Embedder()

    for section, text in sections.items():
        print(f"\n➡️ Processing Section: {section.upper()}")

        if not text.strip():
            print("⚠️ Skipped (empty)")
            continue

        print(f"Text length: {len(text)}")

        # Chunking
        chunks = chunker.chunk_text(text)
        print(f"Chunks created: {len(chunks)}")

        if not chunks:
            print("❌ No chunks created!")
            continue

        print(f"First chunk preview:\n{chunks[0][:200]}")

        # Embedding
        try:
            embeddings = embedder.embed_chunks(chunks)

            print(f"Embeddings generated: {len(embeddings)}")

            if embeddings:
                print(f"Embedding dimension: {len(embeddings[0])}")
            else:
                print("❌ Embeddings empty!")

        except Exception as e:
            print(f"❌ Embedding failed: {e}")


if __name__ == "__main__":
    pdf_path = r"../data/sample_papers/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf"
    run_pipeline(pdf_path)