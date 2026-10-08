
import os
from app.ml.config import ROOT
from app.rag.vector_store import VectorStore

embedding_model = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

store = VectorStore(
    ROOT / "models/rag_index.faiss",
    ROOT / "models/rag_metadata.json",
    embedding_model
)

count = store.build(ROOT / "data/documents")
print(f"Indexed {count} chunks.")
