import os
import json
from sentence_transformers import SentenceTransformer

# Avoid GPU probing delays on Windows
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load filtered journals
with open("journals_filtered.json", "r", encoding="utf-8") as f:
    journals = json.load(f)

# Build domain_text deterministically
texts = []
for j in journals:
    categories = j.get("categories", "")
    areas = j.get("areas", "")
    title = j.get("title", "")

    domain_text = (
        f"{title}. This journal publishes research in {categories}. "
        f"It belongs to the broader research areas of {areas}."
    )

    j["domain_text"] = domain_text
    texts.append(domain_text)

# Generate embeddings
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_tensor=False
)

# Attach embeddings
for j, emb in zip(journals, embeddings):
    j["embedding"] = emb.tolist()

# Save output
with open("journals_embedded.json", "w", encoding="utf-8") as f:
    json.dump(journals, f, indent=2, ensure_ascii=False)

print("Embeddings generated for", len(journals), "journals")
