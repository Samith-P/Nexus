import json
import re

def parse_sjr(sjr_raw):
    if not sjr_raw:
        return {"sjr": 0.0, "quartile": None}

    match = re.search(r"([\d.]+)", sjr_raw)
    sjr = float(match.group(1)) if match else 0.0

    quartile = None
    if "Q1" in sjr_raw:
        quartile = "Q1"
    elif "Q2" in sjr_raw:
        quartile = "Q2"
    elif "Q3" in sjr_raw:
        quartile = "Q3"
    elif "Q4" in sjr_raw:
        quartile = "Q4"

    return {"sjr": sjr, "quartile": quartile}


with open("journals_raw.json", "r", encoding="utf-8") as f:
    journals = json.load(f)

cleaned = []

for j in journals:
    sjr_data = parse_sjr(j.get("sjr", ""))

    cleaned.append({
        "title": j["title"],
        "type": j["type"],
        "sjr": sjr_data["sjr"],
        "quartile": sjr_data["quartile"],
        "h_index": int(j["h_index"]),
        "total_docs_2024": int(j["total_docs_2024"]),
        "total_docs_3y": int(j["total_docs_3y"]),
        "total_refs_2024": int(j["total_refs_2024"]),
        "total_citations_3y": int(j["total_citations_3y"]),
        "citable_docs_3y": int(j["citable_docs_3y"]),
        "citations_per_doc_2y": float(j["citations_per_doc_2y"]),
        "refs_per_doc_2024": float(j["refs_per_doc_2024"]),
        "female_percent_2024": float(j["female_percent_2024"]) if j["female_percent_2024"] else None
    })

with open("journals_clean.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2)

print("Cleaned journals:", len(cleaned))
