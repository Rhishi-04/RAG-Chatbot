#!/bin/bash

# Start Ollama server in background
echo "🚀 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
sleep 10

# Pull a smaller model that works on CPU
# Using mistral:7b-instruct-q4_K_M - 4-bit quantized (smaller)
echo "📥 Pulling Mistral 7B quantized model..."
ollama pull mistral:7b-instruct-q4_K_M

# Wait a bit more for model to be ready
sleep 5

echo "✅ Ollama ready!"

# Start the FastAPI app on port 7860
echo "🚀 Starting FastAPI application..."
cd /app
python3 api.py &
API_PID=$!

# Keep container running
wait $OLLAMA_PID $API_PID

