#!/usr/bin/env python3
"""
Startup script for IS Code Regulatory Assistant
"""

import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import sentence_transformers
        import faiss
        print("✓ All Python dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
            return True
        else:
            print("✗ Ollama is not responding properly")
            return False
    except Exception:
        print("✗ Ollama is not running")
        print("Please start Ollama: ollama serve")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("Starting backend server...")
    try:
        # Start the backend server
        backend_process = subprocess.Popen([
            sys.executable, "api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for the server to start
        time.sleep(3)
        
        # Check if server is running
        import requests
        try:
            response = requests.get("http://localhost:8000/docs", timeout=5)
            if response.status_code == 200:
                print("✓ Backend server is running on http://localhost:8000")
                return backend_process
            else:
                print("✗ Backend server failed to start")
                return None
        except Exception:
            print("✗ Backend server failed to start")
            return None
            
    except Exception as e:
        print(f"✗ Error starting backend: {e}")
        return None

def start_frontend():
    """Start a simple HTTP server for the frontend"""
    print("Starting frontend server...")
    try:
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "http.server", "8001"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(2)
        print("✓ Frontend server is running on http://localhost:8001")
        return frontend_process
    except Exception as e:
        print(f"✗ Error starting frontend: {e}")
        return None

def main():
    """Main startup function"""
    print("🚀 Starting IS Code Regulatory Assistant...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check Ollama
    if not check_ollama():
        print("\nNote: You can still start the application, but LLM functionality won't work.")
        print("To enable LLM: ollama serve")
    
    print("\n" + "=" * 50)
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("Failed to start backend server")
        sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("Failed to start frontend server")
        backend_process.terminate()
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 IS Code Regulatory Assistant is ready!")
    print("\nAccess points:")
    print("• Frontend: http://localhost:8001")
    print("• Backend API: http://localhost:8000")
    print("• API Docs: http://localhost:8000/docs")
    
    # Open browser
    try:
        webbrowser.open("http://localhost:8001")
        print("\n✓ Browser opened automatically")
    except:
        print("\nPlease open your browser and go to: http://localhost:8001")
    
    print("\nPress Ctrl+C to stop all servers")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("✓ All servers stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
