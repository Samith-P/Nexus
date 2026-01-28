# gemini/schema.py

REQUIRED_KEYS = {
    "primary_research_area": str,
    "secondary_areas": list,
    "methods": list,
    "application_domains": list,
    "key_concepts": list,
    "condensed_summary": str,
}


def validate_gemini_output(data: dict) -> bool:
    if not isinstance(data, dict):
        return False

    for key, expected_type in REQUIRED_KEYS.items():
        if key not in data:
            return False
        if not isinstance(data[key], expected_type):
            return False

    return True
