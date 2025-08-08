# app/document_loader.py

import os
import requests
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def load_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            content = extract_text_from_pdf(file_path)
            documents.append({
                "filename": filename,
                "content": content
            })
    return documents
def load_documents_from_file(file_path):
    """
    Extract text content from a single PDF file path.
    Returns extracted text as a string.
    """
    try:
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text
        else:
            raise ValueError("Unsupported file type: only PDFs supported.")
    except Exception as e:
        return f"Error reading file: {e}"
def load_documents_from_url(url: str, save_path: str):
    

    # Download PDF from URL
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download file from URL: {url}")

    # Extract text using PyPDF2
    reader = PdfReader(save_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    return [{"filename": os.path.basename(save_path), "content": text}]
