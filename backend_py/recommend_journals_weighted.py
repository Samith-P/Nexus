import json
import torch
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# User input
user_abstract = """
This research focuses on transformer-based deep learning models
for natural language processing, including semantic understanding,
text generation, and cross-lingual representation learning.
"""

query_embedding = model.encode(user_abstract, convert_to_tensor=True)

# Load journals
with open("journals_embedded.json", "r", encoding="utf-8") as f:
    journals = json.load(f)

journal_embeddings = torch.tensor([j["embedding"] for j in journals])

# Semantic similarity
similarities = util.cos_sim(query_embedding, journal_embeddings)[0]

# Collect metric ranges
sjr_values = [j["sjr"] for j in journals]
cite_values = [j["citations_per_doc_2y"] for j in journals]

min_sjr, max_sjr = min(sjr_values), max(sjr_values)
min_cite, max_cite = min(cite_values), max(cite_values)

def norm(x, min_x, max_x):
    if max_x == min_x:
        return 0.0
    return (x - min_x) / (max_x - min_x)

results = []

for j, sim in zip(journals, similarities):
    sjr_norm = norm(j["sjr"], min_sjr, max_sjr)
    cite_norm = norm(j["citations_per_doc_2y"], min_cite, max_cite)

    final_score = (
        0.60 * float(sim)
        + 0.25 * sjr_norm
        + 0.15 * cite_norm
    )

    results.append({
        "title": j["title"],
        "semantic_similarity": round(float(sim), 3),
        "sjr": j["sjr"],
        "citations_per_doc_2y": j["citations_per_doc_2y"],
        "final_score": round(final_score, 3)
    })

# Sort by final score
results.sort(key=lambda x: x["final_score"], reverse=True)

print("\nTop Recommended Journals (Metric-Aware):\n")
for r in results[:5]:
    print(
        f"{r['title']} | final={r['final_score']} | "
        f"sim={r['semantic_similarity']} | SJR={r['sjr']}"
    )
