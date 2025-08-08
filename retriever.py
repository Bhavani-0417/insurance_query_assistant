# app/retriever.py

import os
import numpy as np
import faiss
import pickle

VECTOR_INDEX_PATH = "vector_index/index.pkl"
CHUNKS_DIR = "docs_chunks"

def load_vector_index():
    with open(VECTOR_INDEX_PATH, "rb") as f:
        index_data = pickle.load(f)
    return index_data[0], index_data[1], index_data[2]

def retrieve_relevant_chunks(query_embedding, top_k=3):
    index, chunks, _ = load_vector_index()
    distances, indices = index.search(np.array([query_embedding]), top_k)
    top_chunks = [chunks[i] for i in indices[0]]
    return top_chunks


def get_top_chunks(user_query, embed_fn, top_k=3):
    query_vector = embed_fn(user_query)
    top_chunks = retrieve_relevant_chunks(query_vector, top_k)
    return top_chunks
