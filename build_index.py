# build_index.py
import os
import pickle
import faiss
from app.document_loader import load_documents
from app.embedder import embed_documents

VECTOR_INDEX_PATH = "vector_index/index.pkl"

def build_index():
    # 1. Load and chunk documents
    texts = load_documents("data/")  # assumes PDFs or DOCX in /data

    # 2. Get embeddings
    embeddings = embed_documents(texts)

    # 3. Save vector index
    if not os.path.exists("vector_index"):
        os.makedirs("vector_index")

    with open(VECTOR_INDEX_PATH, "wb") as f:
        pickle.dump((embeddings, texts, None), f)

    print(f"✅ Vector index built and saved to {VECTOR_INDEX_PATH}")

# Optional: allow running this directly from command line
if __name__ == "__main__":
    build_index()
