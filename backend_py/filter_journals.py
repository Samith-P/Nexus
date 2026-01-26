import json

with open("journals_clean.json", "r") as f:
    journals = json.load(f)

filtered = [
    j for j in journals
    if j["sjr"] >= 0.5
    and j["h_index"] >= 20
    and j["quartile"] in ["Q1", "Q2"]
]

filtered.sort(key=lambda x: x["sjr"], reverse=True)

with open("journals_filtered.json", "w") as f:
    json.dump(filtered, f, indent=2)

print("Filtered journals:", len(filtered))
