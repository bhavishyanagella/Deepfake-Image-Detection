
from pathlib import Path
import json
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, index_path, metadata_path, embedding_model):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.metadata = []

    def chunk_text(self, text, size=900, overlap=120):
        text = re.sub(r"\s+", " ", text).strip()
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap
        return chunks

    def build(self, documents_dir):
        documents_dir = Path(documents_dir)
        records = []

        for path in sorted(documents_dir.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            for i, chunk in enumerate(self.chunk_text(text)):
                records.append({
                    "source": str(path.relative_to(documents_dir)),
                    "chunk_id": i,
                    "text": chunk
                })

        texts = [r["text"] for r in records]
        vectors = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype("float32")

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        self.index = index
        self.metadata = records
        return len(records)

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(
            self.metadata_path.read_text(encoding="utf-8")
        )

    def search(self, query, top_k=4):
        if self.index is None:
            self.load()

        q = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        scores, ids = self.index.search(q, top_k)
        results = []

        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[int(idx)])
            item["score"] = float(score)
            results.append(item)

        return results
