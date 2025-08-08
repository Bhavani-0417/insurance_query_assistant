# app/llm_reasoner.py

import ollama
import re
import json

def extract_json_from_response(response: str):
    """
    Extract the first JSON object from the response string.
    """
    try:
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if match:
            json_text = match.group(0)
            return json.loads(json_text)
        else:
            return {"error": "No JSON found in response"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {str(e)}"}

def query_llm(user_query, retrieved_docs):
    """
    Query the LLM model via Ollama using the user query and retrieved documents.
    """
    prompt = f"""
You are an intelligent insurance document assistant. Use the following policy documents to answer the user's question. Be accurate, concise, and truthful.

Context:
{retrieved_docs}

Question:
{user_query}

Provide the answer in the following JSON format only:

{{
    "decision": "Approved or Denied",
    "amount": number,
    "justification": "Short explanation based on rules"
}}
"""
    try:
        response = ollama.chat(model="phi", messages=[{"role": "user", "content": prompt}])
        raw_output = response['message']['content']
        print("📦 Raw Ollama Response:", raw_output)

        result = extract_json_from_response(raw_output)
        return result
    except Exception as e:
        return {"error": f"LLM processing failed: {str(e)}"}
