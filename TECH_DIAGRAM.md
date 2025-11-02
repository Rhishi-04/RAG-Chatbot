# Technical Architecture Diagram

## 🎯 RAG System Architecture

### System Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │      DOCUMENT UPLOAD PHASE          │
                    └─────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
            ┌───────▼────────┐           ┌────────▼────────┐
            │ PDF Extraction │           │ Text Processing │
            │  (PyMuPDF)     │           │    Cleaning     │
            └───────┬────────┘           └────────┬────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  TEXT CHUNKING              │
                    │  (500 words, 100 overlap)   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  EMBEDDING GENERATION       │
                    │  Sentence Transformers      │
                    │  (384-dim vectors)          │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  FAISS INDEXING             │
                    │  Session-based storage      │
                    └─────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │        QUERY PHASE                  │
                    └─────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Query Embedding            │
                    │  Sentence Transformers      │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  FAISS SEARCH               │
                    │  Inner Product Similarity   │
                    │  Top-10 candidates          │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  RE-RANKING                 │
                    │  Cross-Encoder              │
                    │  Top-3 final chunks         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  LLM GENERATION (Ollama)    │
                    │  Mistral 7B                 │
                    │  Context + Question         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  RESPONSE                   │
                    │  Answer + Confidence + Cite │
                    └─────────────────────────────┘
```

## 📊 Technical Data Flow

### Complete Process Flow

1. **Ingestion** → PyMuPDF extracts text from PDFs
2. **Preprocessing** → Text cleaning and normalization  
3. **Chunking** → Smart splitting (500w × 100w overlap)
4. **Embedding** → Sentence Transformers → 384-dim vectors
5. **Indexing** → FAISS vector database (per session)
6. **Querying** → Semantic similarity search
7. **Reranking** → Cross-encoder precision boost
8. **Generation** → LLM contextual answer
9. **Response** → User receives cited answer

## 🔧 Component Specifications

### Embedding Model
- **Model**: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`
- **Dimensions**: 384
- **Task**: Question-answering optimization
- **Inference**: Fast CPU/GPU

### Vector Database
- **Technology**: FAISS IndexFlatIP
- **Similarity**: Inner Product
- **Latency**: Sub-millisecond queries
- **Scalability**: Millions of documents

### LLM
- **Model**: Mistral 7B via Ollama
- **Infrastructure**: Local deployment
- **Privacy**: On-premise, no data leaves
- **Cost**: Open-source, free

### Retrieval Strategy
- **Stage 1**: FAISS semantic search (Top-10)
- **Stage 2**: Cross-encoder reranking (Top-3)
- **Chunk Size**: 500 words with 100 overlap
- **Precision**: Two-stage filtering

## 💡 Key Innovations

**Two-Stage Retrieval**: Combines fast semantic search with precise reranking
- FAISS provides speed at scale
- Cross-encoder ensures maximum relevance

**Smart Chunking**: Context-aware document splitting
- Overlap preserves information boundaries
- Optimal chunk size for retrieval precision

**Session Isolation**: Multi-tenant architecture
- Independent indexes per session
- No cross-contamination

**Hallucination Prevention**: Strict document grounding
- Responses limited to retrieved context
- Explicit "not available" responses

## 🔄 End-to-End Example

### Upload Phase
```
Document.pdf → PyMuPDF → Raw Text (20KB)
→ Chunking → 16 chunks (500w each)
→ Embedding → 16 vectors (384-dim)
→ FAISS Index → Session storage
```

### Query Phase
```
Question → Embedding → Query Vector (384-dim)
→ FAISS Search → 10 similar chunks
→ Reranking → Top 3 most relevant
→ LLM Context → Answer generation
→ Response with citation
```

---

*This technical diagram represents the complete RAG pipeline implemented in the project.*

