import json
import time
import os
from google import genai

# Configure Gemini (new google-genai client)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)
# Prefer current fast text model; adjust here if you need higher quality (e.g., gemini-2.5-pro)
model_name = "gemini-2.5-flash"


def generate_with_retry(prompt: str, attempts: int = 3) -> str | None:
    """Call the model with simple backoff to handle 429 quota responses."""
    backoff = 20  # seconds
    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:  # noqa: BLE001
            if attempt == attempts:
                raise
            print(f"Retrying after error (attempt {attempt}/{attempts}): {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
    return None

with open("journals_clean.json", "r", encoding="utf-8") as f:
    journals = json.load(f)

for idx, journal in enumerate(journals, start=1):
    if journal.get("domain_text"):
        continue

    title = journal["title"]

    prompt = f"""
Write a concise 2-sentence academic description of the research scope of the journal titled:
"{title}".

Focus only on the research areas and topics it publishes.
Do not include impact factor, rankings, or promotional language.
"""

    try:
        text = generate_with_retry(prompt)
        journal["domain_text"] = text
    except Exception as e:  # noqa: BLE001
        journal["domain_text"] = None
        print(f"Failed for {title}: {e}")

    # Free-tier limit is 5 req/min per model; keep a generous gap
    time.sleep(12.5)

    if idx % 10 == 0:
        with open("journals_text_ready.json", "w", encoding="utf-8") as f:
            json.dump(journals, f, indent=2)

with open("journals_text_ready.json", "w", encoding="utf-8") as f:
    json.dump(journals, f, indent=2)

print("Generated domain_text for", len(journals), "journals")
