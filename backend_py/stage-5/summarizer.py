# stage-5/summarizer.py

from transformers import pipeline


class Summarizer:

    def __init__(self):
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6"
        )

    def summarize_text(self, text):
        input_len = len(text.split())

        max_len = min(150, input_len // 2 + 20)
        min_len = min(40, input_len // 4)

        summary = self.summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        return summary[0]['summary_text']

    def hierarchical_summarize(self, chunks):
        print(f"🔹 Summarizing {len(chunks)} chunks...")

        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            print(f"   ➤ Chunk {i+1}/{len(chunks)}")

            summary = self.summarize_text(chunk)

            if summary:
                chunk_summaries.append(summary)

        if not chunk_summaries:
            return ""

        combined = " ".join(chunk_summaries)

        combined = " ".join(combined.split()[:800])

        print("🔹 Generating final summary...")
        final_summary = self.summarize_text(combined)

        return final_summary