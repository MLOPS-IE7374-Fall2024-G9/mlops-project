#!/bin/bash

# Start Ollama serve in the background
ollama serve &

# Wait for Ollama to initialize
sleep 5

# Start the Uvicorn server in the background
nohup uvicorn backend.app:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
