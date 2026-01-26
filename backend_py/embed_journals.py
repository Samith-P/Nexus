import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("journals_text_ready.json", "r", encoding="utf-8") as f:
    journals = json.load(f)

texts = [j["domain_text"] for j in journals]

embeddings = model.encode(texts, convert_to_tensor=False)

for j, emb in zip(journals, embeddings):
    j["embedding"] = emb.tolist()

with open("journals_embedded.json", "w", encoding="utf-8") as f:
    json.dump(journals, f, indent=2)

print("Embeddings generated for", len(journals), "journals")

