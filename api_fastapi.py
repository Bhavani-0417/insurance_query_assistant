# api_fastapi.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.document_loader import load_documents_from_url
from app.embedder import embed_documents
from app.llm_reasoner import query_llm
import tempfile
import os

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str  # URL to PDF
    questions: list[str]

@app.post("/hackrx/run")
async def run_query(request: QueryRequest):
    try:
        # 1. Download and extract text
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "policy.pdf")
            
            # ✅ This downloads the document
            load_documents_from_url(request.documents, file_path)

            # ✅ Now pass the file to the loader
            chunks = [c["content"] for c in load_documents_from_url(request.documents, file_path)]

        # 2. Get answers from LLM
        context = "\n\n".join(chunks[:3])  # keep context short
        results = []
        for q in request.questions:
            result = query_llm(q, context)
            if isinstance(result, dict) and "justification" in result:
                results.append(result["justification"])
            else:
                results.append(str(result))
        
        return {"answers": results}
    except Exception as e:
        return {"error": str(e)}
