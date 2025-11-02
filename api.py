import logging
import os
from datetime import datetime
from typing import Optional, List, Dict
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import fitz  # pymupdf

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
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
OLLAMA_MODEL = "mistral"

# === Load Models Only (No Persistent Storage) ===
# We only load the embedding model, not persistent data
# All documents are managed in-memory per session
emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_data = []  # Empty - no persistent storage
index = None  # Empty - no persistent storage
print(f"[+] Loaded embedding model: {EMBEDDING_MODEL_NAME}")
print(f"[!] Running in fresh mode - no persistent documents loaded")

# === In-memory Session Memory ===
session_memory: Dict[str, List[Dict[str, str]]] = {}

# === Session-based Document Management ===
# Stores embeddings and indexes per session
session_embeddings: Dict[str, Dict] = {}  # session_id -> {"embeddings": [...], "index": faiss_index, "data": [...]}

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

class ClearSessionRequest(BaseModel):
    session_id: str = "default"

# Utilities
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

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

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

def initialize_session(session_id: str):
    """Initialize a new session with empty embeddings"""
    if session_id not in session_embeddings:
        session_embeddings[session_id] = {
            "embeddings": np.array([]),
            "index": None,
            "data": [],
            "emb_model": None
        }
        logger.info(f"Initialized new session: {session_id}")

def clear_session_documents(session_id: str):
    """Clear all documents for a session"""
    if session_id in session_embeddings:
        session_embeddings[session_id] = {
            "embeddings": np.array([]),
            "index": None,
            "data": [],
            "emb_model": None
        }
        logger.info(f"Cleared documents for session: {session_id}")
    if session_id in session_memory:
        session_memory[session_id] = []
        logger.info(f"Cleared conversation history for session: {session_id}")

def add_document_to_session(session_id: str, file_path: str, content: str):
    """Add a document to a specific session and rebuild its index"""
    initialize_session(session_id)
    
    # Load embedding model if not loaded
    if session_embeddings[session_id]["emb_model"] is None:
        session_embeddings[session_id]["emb_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    model = session_embeddings[session_id]["emb_model"]
    
    # CHUNK the document into smaller pieces
    chunks = chunk_text(content, chunk_size=500, overlap=100)
    
    # Create embeddings for each chunk
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk, convert_to_numpy=True)
        embedding = normalize(embedding).astype("float32")
        chunk_embeddings.append(embedding)
        
        # Add metadata for each chunk
        doc_meta = {
            "topic_title": f"{os.path.basename(file_path)} (chunk {i+1}/{len(chunks)})",
            "combined_text": chunk,
            "doc_id": os.path.splitext(os.path.basename(file_path))[0],
            "chunk_num": i
        }
        session_embeddings[session_id]["data"].append(doc_meta)
    
    # Stack all chunk embeddings
    chunk_embeddings_array = np.vstack(chunk_embeddings)
    
    # Add to embeddings array
    if session_embeddings[session_id]["embeddings"].shape[0] == 0:
        session_embeddings[session_id]["embeddings"] = chunk_embeddings_array
    else:
        session_embeddings[session_id]["embeddings"] = np.vstack([
            session_embeddings[session_id]["embeddings"],
            chunk_embeddings_array
        ])
    
    # Rebuild index
    session_embeddings[session_id]["index"] = faiss.IndexFlatIP(
        session_embeddings[session_id]["embeddings"].shape[1]
    )
    session_embeddings[session_id]["index"].add(
        session_embeddings[session_id]["embeddings"]
    )
    
    logger.info(f"Added document to session {session_id}: {os.path.basename(file_path)} ({len(chunks)} chunks)")

def get_session_index(session_id: str):
    """Get the appropriate index for a session"""
    # Use session-specific index if available
    if session_id in session_embeddings and session_embeddings[session_id]["index"] is not None:
        return session_embeddings[session_id]["index"], session_embeddings[session_id]["data"]
    # Return None if no session data (user must upload documents first)
    return None, []

def retrieve(query, model, index, embedding_data, top_k=10, allowed_docs=None):
    # Safety check
    if index is None or len(embedding_data) == 0:
        return [], 0.0
    
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
    context = "\n\n".join(retrieved_texts[:3])  # Limit to top 3 chunks to reduce noise
    full_prompt = f"""You are a strict document-based Q&A assistant. Answer ONLY using the provided document excerpts below.

ABSOLUTE RULES:
1. ONLY use information EXACTLY as stated in the provided excerpts
2. NEVER use any external knowledge, general facts, or information from outside these excerpts
3. NEVER reference document names or codes NOT mentioned in the excerpts
4. If the answer is not in the excerpts, respond EXACTLY: "This information is not available in the provided documents."
5. Do NOT make assumptions or add information not explicitly stated

PROVIDED DOCUMENT EXCERPTS (ONLY USE THESE):
{context}

QUESTION: {query}

IMPORTANT: Extract your answer ONLY from the excerpts above. If the answer isn't there, say "This information is not available in the provided documents."

ANSWER:"""
    return full_prompt

def query_ollama(prompt: str, model_name: str = OLLAMA_MODEL) -> str:
    try:
        # Strong system message to enforce document-only responses
        system_message = "You are a document-only Q&A assistant. CRITICAL: Answer ONLY from the provided document excerpts. Use ZERO external knowledge. If information is missing, say 'This information is not available in the provided documents.'"
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name, 
                "prompt": prompt, 
                "stream": False,
                "system": system_message
            }
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

    # Get session-specific index
    session_index, session_data = get_session_index(session_id)
    
    # Check if session has documents uploaded
    if session_index is None:
        raise HTTPException(
            status_code=400,
            detail="No documents found in this session. Please upload documents first before asking questions."
        )
    
    session_model = emb_model
    
    # Use session model if available
    if session_id in session_embeddings and session_embeddings[session_id]["emb_model"] is not None:
        session_model = session_embeddings[session_id]["emb_model"]

    retrieved_docs, top_score = retrieve(query, session_model, session_index, session_data, top_k=10, allowed_docs=allowed_docs)
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

from fastapi import UploadFile, File, Form
import shutil

@app.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...), session_id: str = Form("default")):
    try:
        save_path = os.path.join("data/documents", file.filename)
        os.makedirs("data/documents", exist_ok=True)

        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Process the file content
        if save_path.endswith('.pdf'):
            doc = fitz.open(save_path)
            content = ""
            for page in doc:
                content += page.get_text()
            doc.close()
        else:
            with open(save_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        
        # Clean text
        content = " ".join(content.strip().split())
        
        # Add to session-specific index (not global)
        add_document_to_session(session_id, save_path, content)

        return {
            "message": f"Uploaded and indexed: {file.filename}",
            "session_id": session_id,
            "documents_in_session": len(session_embeddings.get(session_id, {}).get("data", []))
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed.")

@app.post("/clear-session")
async def clear_session_endpoint(data: dict):
    """Clear all documents and history for a session"""
    session_id = data.get("session_id", "default")
    clear_session_documents(session_id)
    return {
        "message": f"Cleared session: {session_id}",
        "session_id": session_id
    }

@app.get("/debug/documents")
async def debug_documents(session_id: Optional[str] = None):
    """Debug endpoint to show currently loaded documents"""
    if session_id and session_id in session_embeddings:
        # Show session-specific documents
        data = session_embeddings[session_id]["data"]
        return {
            "session_id": session_id,
            "total_documents": len(data),
            "documents": [
                {
                    "doc_id": doc.get("doc_id", "N/A"),
                    "title": doc.get("topic_title", "N/A"),
                    "content_length": len(doc.get("combined_text", ""))
                }
                for doc in data
            ]
        }
    else:
        # No global documents in fresh mode
        return {
            "session_id": "none",
            "total_documents": 0,
            "documents": [],
            "message": "Running in fresh mode - no persistent documents. Upload documents to a session first."
        }

# === Server Startup ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Starting AI RAG Chatbot API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print(f"🌐 Server running on port {port}")
    print("Press Ctrl+C to stop the server")
    uvicorn.run(app, host="0.0.0.0", port=port)
