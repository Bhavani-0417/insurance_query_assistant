# app/embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_documents(chunks):
    """
    Takes a list of text chunks and returns their vector embeddings.
    """
    return model.encode(chunks, show_progress_bar=True)

def embed_query_texts(query: str):
    """
    Takes a single query string and returns its embedding.
    """
    return model.encode(query)
