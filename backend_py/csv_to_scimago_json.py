import csv
import json

INPUT_CSV = "scimagojr 2024  Subject Category - Artificial Intelligence.csv"
OUTPUT_JSON = "journals_scimago_raw.json"

journals = []

with open(INPUT_CSV, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=";")
    header = next(reader)  # skip header row

    for row in reader:
        try:
            journal = {
                "title": row[2].strip(),
                "detail_url": f"https://www.scimagojr.com/journalsearch.php?q={row[1]}&tip=sid&clean=0",
                "type": row[3].strip(),
                "issn": row[4].strip(),
                "publisher": row[5].strip(),
                "open_access": row[6].strip(),
                "sjr_raw": row[9].strip(),  # keep raw for now
                "h_index": row[10].strip(),
                "total_docs_2024": row[11].strip(),
                "total_docs_3y": row[12].strip(),
                "total_refs_2024": row[13].strip(),
                "total_citations_3y": row[14].strip(),
                "citable_docs_3y": row[15].strip(),
                "citations_per_doc_2y": row[16].strip(),
                "refs_per_doc_2024": row[17].strip(),
                "female_percent_2024": row[18].strip(),
                "country": row[21].strip(),
                "region": row[22].strip(),
                "categories": row[24].strip(),
                "areas": row[25].strip()
            }

            journals.append(journal)

        except IndexError:
            # Skip malformed rows safely
            continue

print("Total journals parsed:", len(journals))

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(journals, f, indent=2, ensure_ascii=False)

print("Saved:", OUTPUT_JSON)
