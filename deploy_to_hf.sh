#!/bin/bash

# Script to deploy RAG-Chatbot to Hugging Face Spaces
# Usage: ./deploy_to_hf.sh YOUR_USERNAME YOUR_SPACE_NAME

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -ne 2 ]; then
    echo -e "${RED}Usage: ./deploy_to_hf.sh YOUR_USERNAME YOUR_SPACE_NAME${NC}"
    echo -e "${YELLOW}Example: ./deploy_to_hf.sh rhishi-04 rag-chatbot${NC}"
    exit 1
fi

HF_USERNAME=$1
SPACE_NAME=$2
HF_REPO="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Hugging Face Spaces Deployment${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Step 1: Create Space first (manual step)
echo -e "${YELLOW}Step 1: Create Space on Hugging Face${NC}"
echo -e "1. Go to: https://huggingface.co/spaces"
echo -e "2. Click 'Create new Space'"
echo -e "3. Space name: ${SPACE_NAME}"
echo -e "4. SDK: ${GREEN}Docker${NC}"
echo -e "5. Hardware: CPU basic"
echo ""
read -p "Press Enter after creating the space..."

# Step 2: Clone the space
echo -e "${GREEN}Step 2: Cloning HF Space repository...${NC}"
if [ -d "${SPACE_NAME}" ]; then
    echo -e "${YELLOW}Directory ${SPACE_NAME} exists. Removing...${NC}"
    rm -rf "${SPACE_NAME}"
fi

git clone "${HF_REPO}" "${SPACE_NAME}"
cd "${SPACE_NAME}"

# Step 3: Copy files
echo -e "${GREEN}Step 3: Copying project files...${NC}"
cp ../Dockerfile.spaces Dockerfile
cp ../entrypoint.sh ./
cp ../api.py ./
cp ../index.html ./
cp ../requirements.txt ./
cp -r ../data/ ./

# Step 4: Copy README for HF
echo -e "${GREEN}Step 4: Creating README.md for HF...${NC}"
cat > README.md << 'EOF'
---
title: AI RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: latest
app_port: 7860
---

# AI-Powered RAG Chatbot

An intelligent Retrieval-Augmented Generation chatbot that answers questions using uploaded documents.

## Features

- 📄 Upload PDF/TXT documents
- 🔍 Semantic search with FAISS
- 🧠 AI-powered answers with Mistral
- 💬 Conversation memory
- 📊 Confidence scores

## Usage

1. Upload documents using the left panel
2. Ask questions in the chat
3. Get AI-powered answers from your documents

Built with FastAPI, Sentence Transformers, and Ollama.
EOF

# Step 5: Commit and push
echo -e "${GREEN}Step 5: Committing and pushing to Hugging Face...${NC}"
git add .
git commit -m "Initial deployment: RAG Chatbot"
git push

cd ..
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Your app will be live at:"
echo -e "${GREEN}https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}${NC}"
echo ""
echo -e "⏳ First build takes ~15 minutes"
echo -e "📊 Monitor progress at the URL above"
echo ""

