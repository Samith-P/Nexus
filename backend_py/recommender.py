import json
import torch
from sentence_transformers import SentenceTransformer, util

class JournalRecommender:
    def __init__(self, data_path="journals_embedded.json"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        with open(data_path, "r", encoding="utf-8") as f:
            self.journals = json.load(f)

        self.embeddings = torch.tensor([j["embedding"] for j in self.journals])

        self.sjr_vals = [j["sjr"] for j in self.journals]
        self.cite_vals = [j["citations_per_doc_2y"] for j in self.journals]
        self.min_sjr, self.max_sjr = min(self.sjr_vals), max(self.sjr_vals)
        self.min_cite, self.max_cite = min(self.cite_vals), max(self.cite_vals)

    def _norm(self, x, min_x, max_x):
        if max_x == min_x:
            return 0.0
        return (x - min_x) / (max_x - min_x)

    def recommend(self, abstract: str, top_k: int = 5):
        q = self.model.encode(abstract, convert_to_tensor=True)
        sims = util.cos_sim(q, self.embeddings)[0]

        results = []
        for j, sim in zip(self.journals, sims):
            sjr_n = self._norm(j["sjr"], self.min_sjr, self.max_sjr)
            cite_n = self._norm(j["citations_per_doc_2y"], self.min_cite, self.max_cite)

            final_score = (
                0.60 * float(sim)
                + 0.25 * sjr_n
                + 0.15 * cite_n
            )

            results.append({
                "title": j["title"],
                "final_score": round(final_score, 3),
                "semantic_similarity": round(float(sim), 3),
                "sjr": j["sjr"],
                "citations_per_doc_2y": j["citations_per_doc_2y"],
                "explanation": self.generate_explanation(j, float(sim))
            })



        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]
    
    def generate_explanation(self,journal, sim_score):
        focus = journal["domain_text"].split(".")[0].lower()

        if sim_score > 0.35:
            relevance = "strongly aligns"
        elif sim_score > 0.20:
            relevance = "aligns"
        else:
            relevance = "partially aligns"

        return (
            f"This journal is recommended because your research {relevance} with its focus on "
            f"{focus}. It also has strong academic impact, with an SJR of {journal['sjr']} "
            f"and {journal['citations_per_doc_2y']} citations per document."
        )
