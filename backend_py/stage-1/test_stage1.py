# src/stage1_parser/test_stage1.py

from pdf_parser import PDFParser


def test_pdf(file_path):
    print(f"\n📄 Testing file: {file_path}")

    parser = PDFParser(file_path)
    result = parser.extract_text()

    print("\n✅ Total Pages:", result["total_pages"])

    print("\n🔹 First 500 characters:\n")
    print(result["full_text"][:500])

    print("\n🔹 Sample Page Breakdown:")
    for page in result["pages"][:2]:
        print(f"\nPage {page['page_number']} Preview:")
        print(page["text"][:200])


if __name__ == "__main__":
    test_pdf("../data/sample_papers/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf")