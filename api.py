# app/api.py

from app.llm_reasoner import query_llm

def query_insurance_llm(user_query, chunks):
    """
    Accept user query and top document chunks, format context, and pass to LLM.
    """
    try:
        context = "\n\n".join(chunk["content"] if isinstance(chunk, dict) and "content" in chunk else str(chunk) for chunk in chunks)



        # Pass both query and context to the LLM
        response = query_llm(user_query, context)
        return response

    except Exception as e:
        return {"error": f"API failed: {str(e)}"}
