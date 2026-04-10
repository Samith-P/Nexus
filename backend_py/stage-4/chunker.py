# stage-4/chunker.py

class TextChunker:

    def __init__(self, chunk_size=400, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = words[start:end]

            chunks.append(" ".join(chunk))

            start += self.chunk_size - self.overlap

        return chunks