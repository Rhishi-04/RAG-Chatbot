# Deploying to Hugging Face Spaces

## ✅ Yes, You CAN Deploy Ollama on HF Spaces!

**UPDATE:** With Docker Spaces and a quantized model, it **IS** possible!

## 🚀 Step-by-Step Deployment

### 1. Create a Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Space name**: `your-rag-chatbot` (or any unique name)
   - **License**: MIT
   - **SDK**: **Docker** ← Important!
   - **Hardware**: **CPU basic** (free tier)

### 2. Clone Your Space Repository

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
```

### 3. Copy Project Files

Copy these files from your project:
- `Dockerfile.spaces` (rename to `Dockerfile`)
- `entrypoint.sh`
- `api.py`
- `index.html`
- `requirements.txt`
- `data/` folder

```bash
cp Dockerfile.spaces Dockerfile
cp entrypoint.sh ./
cp api.py ./
cp index.html ./
cp requirements.txt ./
cp -r data/ ./
```

### 4. Commit and Push

```bash
git add .
git commit -m "Deploy RAG chatbot to Hugging Face Spaces"
git push
```

### 5. Wait for Build

- Hugging Face will automatically build your Docker image
- This takes 10-15 minutes (downloading Ollama + model)
- Watch the logs on your Space page

### 6. Access Your App

Once built, your app is live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

## ⚙️ How It Works

### The Architecture

```
┌─────────────────────────────────────────────┐
│  Hugging Face Docker Container              │
├─────────────────────────────────────────────┤
│  1. ollama/ollama:latest base image        │
│  2. Ollama server runs on 0.0.0.0:11434    │
│  3. Pulls quantized Mistral model          │
│  4. FastAPI runs on port 7860               │
│  5. index.html served statically           │
└─────────────────────────────────────────────┘
```

### Why This Works

✅ **Docker Spaces**: Allows custom Docker environments  
✅ **Quantized Model**: 4-bit GGUF fits in 16GB RAM  
✅ **Entrypoint**: Orchestrates Ollama + FastAPI startup  
✅ **Port 7860**: Standard HF Spaces port  

## 📊 Resource Usage

### Free Tier Limits:
- **RAM**: 16GB
- **CPU**: Basic
- **Storage**: Limited

### Your App Uses:
- **Ollama**: ~2GB
- **Mistral 7B Q4**: ~4-5GB
- **Embeddings**: ~400MB
- **Python/AI**: ~2GB
- **Buffer**: ~6GB
- **Total**: ~14-15GB ✅ Fits!

## ⚠️ Important Notes

### Model Selection
Using `mistral:7b-instruct-q4_K_M` because:
- ✅ 4-bit quantized (smaller)
- ✅ Works on CPU
- ✅ Better performance for limited resources
- ✅ Fits in 16GB RAM

### Performance Expectation
On free CPU tier:
- **First response**: 30-60 seconds (loading model)
- **Subsequent**: 5-15 seconds per query
- **Slower** than GPU, but **works!**

### Persistence
- Models cached within container
- Rebuilds may redownload
- Consider HF's paid tier for faster rebuilds

## 🔍 Troubleshooting

### Build Fails
```bash
# Check logs on HF Space page
# Common issues:
- Model too large (use smaller/quantized)
- Timeout (increase HEALTHCHECK delay)
- Port conflicts
```

### Model Not Pulling
```bash
# In entrypoint.sh, verify model name:
ollama pull mistral:7b-instruct-q4_K_M
```

### Ollama Not Starting
```bash
# Check OLLAMA_HOST variable
# Should be: 0.0.0.0:11434
```

### Slow Responses
- Normal on free CPU tier
- Upgrade to paid GPU for speed
- Or accept it as demo version

## 🎯 Alternative: Paid GPU Tier

For **faster performance**, consider HF's paid GPU:
- Cost: ~$0.60/hour
- GPU access
- Faster inference (2-5 seconds)
- Can use full Mistral 7B

Update in Space settings → Hardware → GPU

## 🚀 Other Deployment Options

Your project also works great on:
- **Railway.app**: Auto-detects docker-compose.yml
- **Render.com**: Simple Docker deployment
- **Fly.io**: Container hosting
- **DigitalOcean**: App Platform
- **AWS/GCP/Azure**: Full control

## 📝 Files Needed for HF Spaces

Required files in your Space:
```
YOUR_SPACE_NAME/
├── Dockerfile              # From Dockerfile.spaces
├── entrypoint.sh           # Startup script
├── api.py                  # FastAPI backend
├── index.html              # Frontend
├── requirements.txt        # Dependencies
└── data/
    └── documents/.gitkeep  # Keep folder structure
```

## 🎉 Success!

Once deployed, share your Space URL:
```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

Your AI RAG Chatbot is now live on the web! 🌐
