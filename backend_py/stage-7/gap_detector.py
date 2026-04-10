from transformers import pipeline
import re


class GapDetector:

    def __init__(self):
        print("🔹 [Stage 7] Loading FLAN-T5 base model...")
        self.model = pipeline(
            "text2text-generation",
            model="google/flan-t5-base"
        )

    def build_prompt(self, text):

        prompt = f"""
You are an expert AI researcher.

Analyze the following research summary and identify:

- limitations of the approach
- missing aspects
- possible improvements

IMPORTANT:
- Do NOT repeat input
- Do NOT summarize
- Think critically

Return 3-5 short bullet points.

Text:
{text}
"""
        return prompt

    def detect_gaps(self, insights, context_text):

        print("\n🧠 [Stage 7] Detecting gaps using hybrid approach...")

        gaps = []

        # ---------------- RULE-BASED GAPS ---------------- #

        methods = " ".join(insights.get("methods", [])).lower()

        if "training" in methods:
            gaps.append("High computational cost due to training process")

        if "fine-tuning" in methods:
            gaps.append("Requires large labeled datasets for fine-tuning")

        if "transformer" in methods or "bert" in context_text.lower():
            gaps.append("Limited performance on long sequences")

        # Always useful generic research gaps
        gaps.append("Lack of domain-specific adaptation")
        gaps.append("No real-time or low-latency optimization")

        # ---------------- OPTIONAL LLM ADDITION ---------------- #

        combined = ""

        for key, values in insights.items():
            if values:
                combined += f"{key}: " + ", ".join(values) + "\n"

        combined += "\ncontext: " + context_text[:300]

        safe_text = combined[:600]

        prompt = f"""
    Find ONE additional research limitation.

    Do NOT repeat input.
    Return only one short sentence.

    Text:
    {safe_text}
    """

        try:
            output = self.model(
                prompt,
                max_length=50,
                do_sample=False
            )

            extra = output[0]["generated_text"].strip()

            if extra and len(extra) < 120:
                gaps.append(extra)

        except:
            pass

        # remove duplicates
        gaps = list(dict.fromkeys(gaps))

        return gaps[:5]