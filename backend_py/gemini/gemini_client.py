# gemini/gemini_client.py

import json
import time
import re
from typing import Dict

from google import genai

from gemini.prompts import FORMAT_ABSTRACT_PROMPT
from gemini.schema import validate_gemini_output


MODEL_PRIORITY = [
    "models/gemini-2.5-pro",
    "models/gemini-3-pro-preview",
    "models/gemini-pro-latest",
    "models/gemini-3-flash-preview",
    "models/gemini-2.5-flash",
    "models/gemini-flash-latest",
]


def _extract_json_debug(text: str) -> Dict:
    print("\n========== RAW MODEL OUTPUT ==========")
    print(f"Response text (first 500 chars): {text[:500]}")
    print("==========\n")

    text = text.strip()
    if not text:
        raise ValueError("Empty response from model")
    
    # Remove markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    # Extract JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:100]}")

    json_text = match.group(0)

    # Try to parse
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON decode error: {e}")
        # Fix trailing commas
        fixed = re.sub(r',\s*}', '}', json_text)
        fixed = re.sub(r',\s*]', ']', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e2:
            raise ValueError(f"JSON parse failed: {e2}")


def format_abstract_with_gemini(abstract: str, api_key: str) -> Dict:
    if not abstract or len(abstract.strip()) < 30:
        raise ValueError("Abstract is too short or empty")

    client = genai.Client(api_key=api_key)
    last_error = None

    for attempt, model_name in enumerate(MODEL_PRIORITY):
        try:
            print(f"\n🔁 Trying model: {model_name}")

            prompt = FORMAT_ABSTRACT_PROMPT.format(abstract=abstract)

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                },
            )

            raw_text = response.text
            
            if not raw_text or len(raw_text.strip()) < 10:
                raise ValueError(f"Model returned empty or too short response: {repr(raw_text[:50])}")

            data = _extract_json_debug(raw_text)

            print("🔍 Parsed JSON keys:", list(data.keys()))

            if not validate_gemini_output(data):
                print(f"❌ JSON schema validation failed. Keys found: {list(data.keys())}")
                raise ValueError("Invalid JSON schema: missing required fields or wrong types")

            print(f"✅ SUCCESS with model: {model_name}")
            return data

        except Exception as e:
            print(f"❌ Model failed: {model_name}")
            print(f"Error: {type(e).__name__}: {str(e)}")
            last_error = f"[{model_name}] {str(e)}"
            wait_time = 1 + (attempt * 0.5)  # Exponential backoff
            time.sleep(wait_time)
            continue

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")
