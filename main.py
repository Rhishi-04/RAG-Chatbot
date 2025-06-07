import os
import json
import requests
from typing import List, Dict, Tuple
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# === Configuration ===
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
TXT_DIR = "data/documents"  # Folder where Text_v1.txt, Text_v2.txt are stored
TOP_K = 5
BATCH_SIZE = 8
MIN_WORDS_PER_CONTENT = 20

# === Utility Functions ===
def clean_text(text: str) -> str:
    return " ".join(text.strip().split())

def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

# === Load TXT Documents ===
def load_txt_files(txt_dir: str) -> List[Dict]:
    documents = []
    for filename in os.listdir(txt_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(txt_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            documents.append({
                "title": filename,
                "content": clean_text(content),
                "doc_id": filename.split(".")[0]
            })
    return documents

# === Embedding Preparation ===
def prepare_txt_embeddings(model: SentenceTransformer, docs: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
    texts, meta = [], []
    for doc in docs:
        content = doc["content"]
        if not content or len(content.split()) < MIN_WORDS_PER_CONTENT:
            continue
        texts.append(content)
        meta.append({
            "topic_title": doc["title"],
            "combined_text": content,
            "doc_id": doc["doc_id"]
        })
    if not texts:
        return [], np.array([])
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=BATCH_SIZE, show_progress_bar=True)
    embeddings = np.array([normalize(e) for e in embeddings], dtype=np.float32)
    return meta, embeddings

# === FAISS Index ===
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query: str, model: SentenceTransformer, index: faiss.IndexFlatIP,
             embedding_data: List[Dict], top_k: int = TOP_K) -> List[Dict]:
    query_emb = normalize(model.encode(query, convert_to_numpy=True)).astype("float32")
    D, I = index.search(np.array([query_emb]), top_k)
    return [embedding_data[idx] | {"score": float(D[0][i])} for i, idx in enumerate(I[0])]

# === LLM Answer Generation via Ollama ===
def generate_answer_ollama(query: str, retrieved_texts: List[str], model_name: str = "mistral") -> str:
    context = "\n\n".join(retrieved_texts)
    prompt = f"""You are an expert regulatory assistant.

Use the following context to answer the question:

Context:
{context}

Question: {query}

Answer:"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "No response generated.")
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

# === Main Execution ===
if __name__ == "__main__":
    print("Loading text documents...")
    documents = load_txt_files(TXT_DIR)

    print(" Loading embedding model...")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)

    print(" Preparing embeddings...")
    embedding_data, embeddings = prepare_txt_embeddings(emb_model, documents)

    print(" Building FAISS index...")
    index = build_faiss_index(embeddings)

    os.makedirs("data", exist_ok=True)
    import pickle
    with open("data/embedding_data.pkl", "wb") as f:
        pickle.dump(embedding_data, f)
    faiss.write_index(index, "data/faiss_index.index")
    print(" Embeddings and index saved.")

    # Sample Query
    query = "What are GMP documentation requirements?"

    print("\n Retrieving answers...")
    results = retrieve(query, emb_model, index, embedding_data)
    for i, res in enumerate(results, 1):
        print(f"[{i}] Score: {res['score']:.4f} | Document: {res['topic_title']}")

    retrieved_texts = [r["combined_text"] for r in results]
    answer = generate_answer_ollama(query, retrieved_texts, model_name="mistral")
    print("\n Generated Answer:\n", answer)

def update_index_with_new_file(file_path: str):
    from sentence_transformers import SentenceTransformer
    import pickle

    print(f"\n[+] Updating index with {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if len(content.split()) < MIN_WORDS_PER_CONTENT:
        print("[-] Skipping - not enough content.")
        return

    # Load model and existing data/index
    model = SentenceTransformer(EMBEDDING_MODEL)
    with open("data/embedding_data.pkl", "rb") as f:
        embedding_data = pickle.load(f)
    index = faiss.read_index("data/faiss_index.index")

    # Create new embedding
    embedding = model.encode(content, convert_to_numpy=True)
    embedding = normalize(embedding).astype("float32")

    # Update data structures
    new_meta = {
        "topic_title": os.path.basename(file_path),
        "combined_text": content,
        "doc_id": os.path.splitext(os.path.basename(file_path))[0]
    }
    embedding_data.append(new_meta)
    index.add(np.array([embedding]))

    # Save updated index and data
    with open("data/embedding_data.pkl", "wb") as f:
        pickle.dump(embedding_data, f)
    faiss.write_index(index, "data/faiss_index.index")

    print("[+] Index updated.")
