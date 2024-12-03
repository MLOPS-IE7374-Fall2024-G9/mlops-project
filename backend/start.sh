#!/bin/bash

# Start Ollama serve in the background
ollama serve &

# Wait for Ollama to initialize
sleep 5

# Start the Uvicorn server
exec uvicorn backend.app:app --host 0.0.0.0 --port 8000
