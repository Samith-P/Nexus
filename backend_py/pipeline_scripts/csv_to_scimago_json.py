import csv
import json
import os
from glob import glob
from pathlib import Path

# Get the backend_py directory (parent of pipeline_scripts)
BACKEND_DIR = Path(__file__).parent.parent

DATASETS_DIR = BACKEND_DIR / "datasets"
OUTPUT_JSON = BACKEND_DIR / "pipeline_cache" / "journals_scimago_raw.json"

# Create pipeline_cache directory if it doesn't exist
(BACKEND_DIR / "pipeline_cache").mkdir(exist_ok=True)

# Find all CSV files in datasets folder
csv_files = glob(str(DATASETS_DIR / "*.csv"))

if not csv_files:
    print(f"[ERROR] No CSV files found in {DATASETS_DIR}")
    exit(1)

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")

all_journals = []
seen_titles = set()  # To track duplicates across years

for csv_file in csv_files:
    print(f"\nProcessing: {os.path.basename(csv_file)}")
    
    try:
        with open(csv_file, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader)  # skip header row
            
            count = 0
            duplicates = 0
            
            for row in reader:
                try:
                    title = row[2].strip()
                    
                    # Skip duplicates (keep first occurrence)
                    if title in seen_titles:
                        duplicates += 1
                        continue
                    
                    journal = {
                        "title": title,
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

                    all_journals.append(journal)
                    seen_titles.add(title)
                    count += 1

                except IndexError:
                    # Skip malformed rows safely
                    continue
            
            print(f"  Parsed: {count} journals")
            if duplicates > 0:
                print(f"  Skipped: {duplicates} duplicates")
    
    except Exception as e:
        print(f"  [ERROR] Failed to process {csv_file}: {e}")
        continue

print(f"\n[OK] Total unique journals parsed: {len(all_journals)}")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_journals, f, indent=2, ensure_ascii=False)

print(f"[OK] Saved: {OUTPUT_JSON.name}")
