# stage-6/insight_extractor.py

from transformers import pipeline
import re


class InsightExtractor:

    def __init__(self):
        self.model = pipeline(
            "text2text-generation",
            model="google/flan-t5-base"
        )

    # ---------------- PROMPT BUILDER ---------------- #

    def build_prompt(self, text, task):

        if task == "contributions":
            instruction = """
Extract 2-4 clear contributions of the paper.
Focus on what is NEW or proposed.
"""

        elif task == "methods":
            instruction = """
Extract specific methods, techniques, or algorithms used.

IMPORTANT:
Look for exact method names such as:
- Masked Language Modeling
- Next Sentence Prediction
- Pre-training
- Fine-tuning
- Transformer architecture

Do NOT give generic answers like "architecture".
"""

        elif task == "results":
            instruction = """
Extract 2-4 key numerical results.

Focus on:
- scores
- accuracy
- benchmarks
- improvements

Return each result separately.
"""

        else:
            instruction = "Extract important information."

        prompt = f"""
You are an expert AI researcher.

{instruction}

Return ONLY bullet points (2-4 points).
Be specific and concise.

Text:
{text}
"""
        return prompt

    # ---------------- MAIN EXTRACTOR ---------------- #

    def extract(self, text, task):

        if not text.strip():
            return []

        prompt = self.build_prompt(text, task)

        try:
            output = self.model(
                prompt,
                max_length=400,
                do_sample=False
            )

            result = output[0]["generated_text"]

            # ---------------- CLEAN OUTPUT ---------------- #

            lines = re.split(r"\n|•|-|\. ", result)
            cleaned = []

            for line in lines:
                line = line.strip()

                # remove empty / weak lines
                if len(line) < 10:
                    continue

                # normalize spacing
                line = re.sub(r"\s+", " ", line)

                cleaned.append(line)

            # remove duplicates
            cleaned = list(dict.fromkeys(cleaned))

            # ---------------- BOOST METHODS ---------------- #

            if task == "methods":
                cleaned = self.boost_methods(text, cleaned)

            return cleaned

        except Exception as e:
            print(f"❌ Error in {task}: {e}")
            return []

    # ---------------- RULE-BASED BOOST ---------------- #

    def boost_methods(self, text, extracted):

        keywords = {
            "masked language modeling": "Masked Language Modeling",
            "next sentence prediction": "Next Sentence Prediction",
            "pre-training": "Pre-training",
            "fine-tuning": "Fine-tuning",
            "transformer": "Transformer Architecture"
        }

        text = text.lower()

        for key, value in keywords.items():
            if key in text and not any(value.lower() in e.lower() for e in extracted):
                extracted.append(value)

        return extracted