# Docker Setup Guide

## Prerequisites
- Docker installed on your system
- Docker Compose (optional but recommended)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up --build

# Access the application
open http://localhost:8000/index.html
```

### Option 2: Using Docker Only

```bash
# Build the image
docker build -t rag-chatbot .

# Run the container
docker run -d \
  -p 8000:8000 \
  -p 11434:11434 \
  -v $(pwd)/data/documents:/app/data/documents \
  --name rag-chatbot \
  rag-chatbot

# Access the application
open http://localhost:8000/index.html
```

## Stopping the Container

```bash
# If using docker-compose
docker-compose down

# If using docker directly
docker stop rag-chatbot
docker rm rag-chatbot
```

## Checking Logs

```bash
# Docker compose
docker-compose logs -f

# Docker only
docker logs -f rag-chatbot
```

## Features

- ✅ Complete RAG chatbot
- ✅ Ollama integrated
- ✅ Mistral model auto-downloaded
- ✅ Persistent document storage
- ✅ Session management
- ✅ Production-ready

## Troubleshooting

### Port Already in Use

```bash
# Stop any conflicting containers
docker ps | grep rag-chatbot
docker kill <container-id>

# Or change ports in docker-compose.yml
```

### Ollama Not Starting

```bash
# Check Ollama logs
docker exec rag-chatbot ollama list

# Restart Ollama
docker exec rag-chatbot ollama serve
```

### Clear All Data

```bash
docker-compose down -v
docker-compose up --build
```

## Production Deployment

For production, consider:
- Using `docker-compose.prod.yml` with volume mounts
- Setting up reverse proxy (nginx)
- Using environment variables for secrets
- Adding health checks

