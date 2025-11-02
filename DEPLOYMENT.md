# Deployment Guide

## Hugging Face Spaces (Not Recommended)

**⚠️ Limitations:**
- Cannot run Ollama locally
- No LLM support without external API
- Limited resources

**Alternative:** Use Ollama Cloud API or OpenAI instead of local Ollama.

## GitHub Deployment (Recommended)

This project is best deployed on:
- **Local machine** with Ollama installed
- **Cloud VM** (AWS EC2, Google Cloud, etc.) with Ollama
- **Docker** container with Ollama included

## Docker Deployment

```bash
# Build image
docker build -t rag-chatbot .

# Run container
docker run -p 8000:8000 rag-chatbot
```

## Requirements

- Python 3.8+
- Ollama installed and running
- Internet connection for model downloads

