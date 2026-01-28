import json
import re

# ---------- Helpers ----------

def safe_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value == "-" or value.lower() == "nan":
        return None
    return float(value.replace(",", "."))


def safe_int(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value == "-" or value.lower() == "nan":
        return None
    return int(value.replace(",", ""))


def parse_sjr(sjr_raw):
    if not sjr_raw:
        return {"sjr": None, "quartile": None}

    sjr_raw = sjr_raw.strip()

    # Extract numeric SJR
    match = re.search(r"([\d.,]+)", sjr_raw)
    sjr = safe_float(match.group(1)) if match else None

    quartile = None
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        if q in sjr_raw:
            quartile = q
            break

    return {"sjr": sjr, "quartile": quartile}


# ---------- Load raw data ----------

with open("journals_scimago_raw.json", "r", encoding="utf-8") as f:
    journals = json.load(f)

cleaned = []

# ---------- Cleaning loop ----------

for j in journals:
    sjr_data = parse_sjr(j.get("sjr_raw"))

    cleaned.append({
        "title": j.get("title"),
        "type": j.get("type"),
        "sjr": sjr_data["sjr"],
        "quartile": sjr_data["quartile"],
        "h_index": safe_int(j.get("h_index")),
        "total_docs_2024": safe_int(j.get("total_docs_2024")),
        "total_docs_3y": safe_int(j.get("total_docs_3y")),
        "total_refs_2024": safe_int(j.get("total_refs_2024")),
        "total_citations_3y": safe_int(j.get("total_citations_3y")),
        "citable_docs_3y": safe_int(j.get("citable_docs_3y")),
        "citations_per_doc_2y": safe_float(j.get("citations_per_doc_2y")),
        "refs_per_doc_2024": safe_float(j.get("refs_per_doc_2024")),
        "female_percent_2024": safe_float(j.get("female_percent_2024")),
        "publisher": j.get("publisher"),
        "open_access": j.get("open_access"),
        "country": j.get("country"),
        "categories": j.get("categories"),
        "areas": j.get("areas"),
    })

# ---------- Save cleaned data ----------

with open("journals_clean.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print("Cleaned journals:", len(cleaned))
