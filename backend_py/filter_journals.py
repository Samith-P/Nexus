import json

with open("journals_clean.json", "r", encoding="utf-8") as f:
    journals = json.load(f)

filtered = []

for j in journals:
    sjr = j.get("sjr")

    # Skip journals without SJR
    if sjr is None:
        continue

    # Apply quality filters
    if (
        sjr >= 0.5
        and j.get("quartile") in {"Q1", "Q2"}
        and j.get("type") == "journal"
    ):
        filtered.append(j)

with open("journals_filtered.json", "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)

print("Filtered journals:", len(filtered))
