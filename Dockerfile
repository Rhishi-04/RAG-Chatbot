FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY index.html .
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p data/documents

# Expose ports
EXPOSE 8000 11434

# Start Ollama in background and run API
CMD ollama serve & sleep 5 && ollama pull mistral && python api.py

