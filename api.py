import base64
import io
import json
import pickle
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np
from PIL import Image
import pytesseract
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Config ===
EMBEDDING_DATA_FILE = "data/embedding_data.pkl"
FAISS_INDEX_FILE = "data/faiss_index.index"
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
OLLAMA_MODEL = "mistral"

# === Load Data and Models ===
with open(EMBEDDING_DATA_FILE, "rb") as f:
    embedding_data = pickle.load(f)
index = faiss.read_index(FAISS_INDEX_FILE)
emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# === In-memory Session Memory ===
session_memory: Dict[str, List[Dict[str, str]]] = {}

# === Schemas ===
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None
    document_ids: Optional[List[str]] = None
    session_id: Optional[str] = "default"

class Link(BaseModel):
    url: Optional[str]
    text: str

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    timestamp: str

# === Utilities ===
def normalize(v):
    return v / np.linalg.norm(v)


def rerank(query, docs, model):
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_embs = [model.encode(doc["combined_text"], convert_to_tensor=True) for doc in docs]
    scores = [util.pytorch_cos_sim(query_emb, d_emb).item() for d_emb in doc_embs]
    for doc, score in zip(docs, scores):
        doc["rerank_score"] = score
    docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return docs

def truncate_context(texts, max_tokens=1500):
    total = 0
    truncated = []
    for text in texts:
        tokens = len(text.split())
        if total + tokens > max_tokens:
            break
        truncated.append(text)
        total += tokens
    return truncated

def retrieve(query, model, index, embedding_data, top_k=10, allowed_docs=None):
    query_emb = normalize(model.encode(query, convert_to_numpy=True)).astype("float32")
    D, I = index.search(np.array([query_emb]), top_k)
    results = []
    for i, idx in enumerate(I[0]):
        data = embedding_data[idx]
        if allowed_docs and data.get("doc_id") not in allowed_docs:
            continue
        data_with_score = data.copy()
        data_with_score["score"] = float(D[0][i])
        results.append(data_with_score)
    reranked = rerank(query, results, model)
    return reranked, reranked[0]["rerank_score"] if reranked else 0.0

def build_prompt(query: str, retrieved_texts: List[str], history: str = "") -> str:
    context = "\n\n".join(retrieved_texts)
    full_prompt = f"""You are a helpful assistant specialized in regulatory guidelines.

Context:
{context}

Conversation history:
{history}

Current Question:
{query}

Answer:"""
    return full_prompt

def query_ollama(prompt: str, model_name: str = OLLAMA_MODEL) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "No response generated.")
    except Exception as e:
        logger.error(f"Ollama LLM call failed: {e}")
        return "Error: LLM failed to generate response."

# === Main API Endpoint ===
@app.post("/api/", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    query = req.question.strip()
    session_id = req.session_id or "default"

    allowed_docs = req.document_ids or None

    retrieved_docs, top_score = retrieve(query, emb_model, index, embedding_data, top_k=10, allowed_docs=allowed_docs)
    retrieved_texts = truncate_context([d["combined_text"] for d in retrieved_docs])
   
    # === Memory Handling ===
    if session_id not in session_memory:
        session_memory[session_id] = []

    # Build chat history string from last few exchanges
    history_snippet = "\n".join([
        f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}"
        for m in session_memory[session_id][-4:]
    ])

    # Update prompt with memory
    prompt = build_prompt(query, retrieved_texts, history=history_snippet)
    answer = query_ollama(prompt)

    # Store in memory
    session_memory[session_id].append({"role": "user", "content": query})
    session_memory[session_id].append({"role": "assistant", "content": answer})

    return QueryResponse(
        answer=answer.strip(),
        confidence=round(top_score, 4),
        timestamp=datetime.utcnow().isoformat()
    )

from fastapi import UploadFile, File
import shutil
from main import update_index_with_new_file  

@app.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    try:
        save_path = os.path.join("data/documents", file.filename)
        os.makedirs("data/documents", exist_ok=True)

        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Update the index
        update_index_with_new_file(save_path)

        return {"message": f"Uploaded and indexed: {file.filename}"}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed.")
