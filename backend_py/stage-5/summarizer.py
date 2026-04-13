# stage-5/summarizer.py

import os
import requests

class Summarizer:

    def __init__(self):
        self.model_name = "sshleifer/distilbart-cnn-12-6"
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"
        self.token = os.environ.get("HF_TOKEN")

    def summarize_text(self, text):
        input_len = len(text.split())

        max_len = min(150, input_len // 2 + 20)
        min_len = min(40, input_len // 4)

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        payload = {
            "inputs": text,
            "parameters": {
                "max_length": max_len,
                "min_length": min_len,
                "do_sample": False
            }
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
                return result[0]["summary_text"]
            else:
                print(f"Error or unexpected format: {result}")
                return ""
        except Exception as e:
            print(f"Summarization API failed: {e}")
            return ""

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