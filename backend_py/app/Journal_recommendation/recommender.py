import json
import torch
from sentence_transformers import SentenceTransformer, util

from pathlib import Path
class JournalRecommender:
    def __init__(self, data_path=None):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        if data_path is None:
            data_path = Path(__file__).parent / "journal_master.json"

        with open(data_path, "r", encoding="utf-8") as f:
            self.journals = json.load(f)
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

    def recommend(self, gemini_output: dict, top_k: int = 5, search_depth: int = None):
        query = (
            f"{gemini_output.get('primary_research_area', '')} "
            f"{' '.join(gemini_output.get('methods', []))} "
            f"{' '.join(gemini_output.get('application_domains', []))} "
            f"{gemini_output.get('condensed_summary', '')}"
        )

        q = self.model.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(q, self.embeddings)[0]

        # optional speed optimization
        if search_depth:
            top_indices = torch.topk(sims, k=search_depth).indices
        else:
            top_indices = range(len(self.journals))

        results = []
        for idx in top_indices:
            j = self.journals[int(idx)]
            sim = float(sims[int(idx)])

            sjr_n = self._norm(j["sjr"], self.min_sjr, self.max_sjr)
            cite_n = self._norm(j["citations_per_doc_2y"], self.min_cite, self.max_cite)

            final_score = 0.60 * sim + 0.25 * sjr_n + 0.15 * cite_n

            results.append({
                "title": j["title"],
                "type": j.get("type"),
                "quartile": j.get("quartile"),
                "h_index": j.get("h_index"),
                "publisher": j.get("publisher"),
                "open_access": j.get("open_access"),
                "country": j.get("country"),

                "sjr": j["sjr"],
                "citations_per_doc_2y": j["citations_per_doc_2y"],
                "semantic_score": round(sim, 3),
                "final_score": round(final_score, 3),

                "explanation": self.generate_explanation(j, sim)
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