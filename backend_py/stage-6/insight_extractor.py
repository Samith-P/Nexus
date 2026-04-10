from transformers import pipeline
import re


class InsightExtractor:

    def __init__(self):
        print("🔹 Loading FLAN-T5 base model (one-time)...")
        self.model = pipeline(
            "text2text-generation",
            model="google/flan-t5-base"
        )

    # ---------------- PROMPT ---------------- #

    def build_prompt(self, text, task):

        if task == "contributions":
            instruction = """
Extract ONLY key contributions.
Return 2-3 short phrases.
No full sentences.
"""

        elif task == "methods":
            instruction = """
Extract ONLY method/technique names.

Examples:
- Masked Language Modeling
- Next Sentence Prediction
- Pre-training
- Fine-tuning

Only return names.
"""

        elif task == "results":
            instruction = """
Extract ONLY key results with numbers.

Examples:
- GLUE score 80.5%
- MultiNLI accuracy 86%

Return 2-3 short points.
"""

        prompt = f"""
{instruction}

Text:
{text}
"""
        return prompt

    # ---------------- EXTRACT ---------------- #

    def extract(self, text, task):

        if not text.strip():
            return []

        # 🔥 SAFE INPUT LIMIT
        safe_text = text[:800]

        prompt = self.build_prompt(safe_text, task)

        try:
            output = self.model(
                prompt,
                max_length=100,
                do_sample=False
            )

            result = output[0]["generated_text"]

            # 🔹 CLEAN OUTPUT
            lines = re.split(r"\n|•|-|\. ", result)
            cleaned = []

            for line in lines:
                line = line.strip()

                if len(line) < 5 or len(line) > 80:
                    continue

                line = re.sub(r"\s+", " ", line)

                cleaned.append(line)

            # remove duplicates
            cleaned = list(dict.fromkeys(cleaned))

            # 🔥 METHOD BOOST (FINAL FIX)
            if task == "methods":
                cleaned = self.boost_methods(text, cleaned)

            return cleaned[:4]

        except Exception as e:
            print(f"❌ Error in {task}: {e}")
            return []

    # ---------------- BOOST ---------------- #

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