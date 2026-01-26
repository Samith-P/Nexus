import json
import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

user_abstract = """
This research focuses on transformer-based deep learning models
for natural language processing, including semantic understanding,
text generation, and cross-lingual representation learning.
"""

query_embedding = model.encode(user_abstract, convert_to_tensor=True)

with open("journals_embedded.json", "r", encoding="utf-8") as f:
    journals = json.load(f)

journal_embeddings = torch.tensor([j["embedding"] for j in journals])

scores = util.cos_sim(query_embedding, journal_embeddings)[0]

results = []
for j, score in zip(journals, scores):
    results.append({
        "title": j["title"],
        "sjr": j["sjr"],
        "quartile": j["quartile"],
        "score": float(score)
    })

results.sort(key=lambda x: x["score"], reverse=True)

print("\nTop Recommended Journals:\n")
for r in results[:5]:
    print(f"{r['title']} | similarity={round(r['score'], 3)} | SJR={r['sjr']}")
