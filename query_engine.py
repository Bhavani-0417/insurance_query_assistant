import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
import requests

# Load the same model used for embedding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("vector_index.index")

# Load sources
with open("vector_index_sources.json", "r", encoding="utf-8") as f:
    sources = json.load(f)

def get_query_embedding(query):
    return model.encode([query])[0]

def search_index(query_embedding, k=5):
    query_vector = np.array([query_embedding]).astype("float32")
    _, I = index.search(query_vector, k)
    return I[0]  # Top-k indices



def build_context(indices):
    return "\n---\n".join([sources[i] for i in indices if i < len(sources)])

def ask_ollama(context, query):
    prompt = f"""You are a strict JSON-generating assistant that answers questions based on document context.

Respond ONLY in valid JSON with the following keys:
- decision (string)
- amount (integer or null)
- justification (string)

Context:
{context}

Question:
{query}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi",
            "prompt": prompt,
            "stream": False
        }
    )

    print("📦 Raw Ollama Response:", response.json())

    raw_text = response.json().get("response") or ""

    try:
        # Try parsing just the JSON part
        json_start = raw_text.find("{")
        json_end = raw_text.rfind("}") + 1
        json_str = raw_text[json_start:json_end]
        parsed = json.loads(json_str)
        formatted = (
            f"📌 Decision: {parsed['decision']}\n"
            f"💰 Amount: {parsed.get('amount', 'N/A')}\n"
            f"📝 Justification: {parsed['justification']}"
        )
        return formatted
    except Exception as e:
        return f"⚠️ Failed to parse JSON:\n\n{raw_text}\n\nError: {str(e)}"



def search_chunks_and_ask_llm(query: str) -> str:
    emb = get_query_embedding(query)
    idxs = search_index(emb)
    context = build_context(idxs)
    result = ask_ollama(context, query)
    return result

