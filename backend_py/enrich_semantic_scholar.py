import requests
import json
import time
import os

API_KEY = os.getenv("S2_API_KEY")

headers = {
    "x-api-key": API_KEY
}

with open("journals_filtered.json") as f:
    journals = json.load(f)

enriched = []

for j in journals:
    query = j["title"]

    url = f"https://api.semanticscholar.org/graph/v1/journal/search?query={query}&limit=1"

    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()

        if data.get("data"):
            info = data["data"][0]
            j["publisher"] = info.get("publisher")
            j["open_access"] = info.get("openAccess", False)
            j["fields_of_study"] = info.get("fieldsOfStudy", [])
        else:
            j["publisher"] = None
            j["open_access"] = None
            j["fields_of_study"] = []

    except Exception as e:
        j["publisher"] = None
        j["open_access"] = None
        j["fields_of_study"] = []

    enriched.append(j)
    time.sleep(0.5)  # rate limit safety

with open("journals_enriched.json", "w") as f:
    json.dump(enriched, f, indent=2)

print("Enriched journals:", len(enriched))
