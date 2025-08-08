# app/ui.py
import os
import shutil
import json
from datetime import datetime
import streamlit as st
from build_index import build_index
from api import query_insurance_llm
from retriever import get_top_chunks
from app.embedder import embed_query_texts

st.set_page_config(page_title="🛡️ Insurance Document Query Assistant")
st.title("🛡️ Insurance Document Query Assistant")
st.markdown("Ask any question based on the uploaded insurance policy documents.")

# 📁 PDF Upload Section
st.sidebar.header("📄 Upload New PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save to data/ folder
    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Rebuild the index
    with st.spinner("🔄 Rebuilding vector index..."):
        build_index()
        st.sidebar.success("✅ Index rebuilt with new document!")
# 📜 View Query Logs
st.sidebar.header("📜 View Query Logs")

if os.path.exists("output/query_logs.jsonl"):
    with st.sidebar.expander("Show Past Queries"):
        with open("output/query_logs.jsonl", "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f.readlines()[-5:]]  # Show last 5 entries

        for i, log in enumerate(reversed(logs), 1):
            st.sidebar.markdown(f"**{i}. [{log['timestamp']}]**")
            st.sidebar.markdown(f"- ❓ **Question:** {log['question']}")
            if 'decision' in log['response']:
                st.sidebar.markdown(f"- ✅ **Decision:** {log['response']['decision']}")
            else:
                st.sidebar.markdown(f"- 🤖 **Response:** {str(log['response'])[:100]}...")
            st.sidebar.markdown("---")
else:
    st.sidebar.info("ℹ️ No logs found yet.")

# 💬 Question Input
user_input = st.text_input("📤 Enter your question:")

if st.button("Submit") and user_input.strip():
    with st.spinner("🔍 Analyzing your question..."):
        # Step 1: Retrieve relevant chunks
        retrieved_docs = get_top_chunks(user_input, embed_query_texts, top_k=3)

        # Show retrieved chunks in sidebar
        st.sidebar.header("📄 Top Retrieved Chunks")
        for i, chunk in enumerate(retrieved_docs, 1):
            st.sidebar.markdown(f"**Chunk {i}:**\n\n{str(chunk)[:500]}...")

        # Step 2: Query the LLM
        result = query_insurance_llm(user_input, retrieved_docs)

        # Step 3: Log the interaction
        log_entry = {
            "timestamp": str(datetime.now()),
            "question": user_input,
            "retrieved_chunks": retrieved_docs,
            "response": result
        }

        os.makedirs("output", exist_ok=True)
        with open("output/query_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # Step 4: Show result to user
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            if "decision" in result:
                st.success(f"✅ Decision: {result['decision']}")
                st.write(f"💰 Amount: {result.get('amount', 'N/A')}")
                st.write(f"📄 Justification: {result.get('justification', 'N/A')}")
            else:
                st.write("🤖 Response:")
                st.json(result)
