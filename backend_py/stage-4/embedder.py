# stage-4/embedder.py

from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_embedding(self, text):
        return self.model.encode(text)

    def embed_chunks(self, chunks):
        return [self.get_embedding(chunk) for chunk in chunks]