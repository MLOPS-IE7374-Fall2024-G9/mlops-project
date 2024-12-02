#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
BACKEND_DOCKERFILE="./backend/Dockerfile"
FRONTEND_DOCKERFILE="./frontend/Dockerfile"

# Build and run the backend container
build_and_run_backend() {
  echo "Building backend Docker image..."
  docker build -t backend -f $BACKEND_DOCKERFILE .

  echo "Running backend Docker container..."
  docker run -d -p 8000:8000 backend
}

# Build and run the frontend container
build_and_run_frontend() {
  echo "Building frontend Docker image..."
  docker build -t frontend -f $FRONTEND_DOCKERFILE .

  echo "Running frontend Docker container..."
  docker run -d -p 8501:8501 frontend
}

# Main script execution
echo "Starting Docker setup..."
build_and_run_backend
build_and_run_frontend
echo "All containers are up and running!"

# Display running containers
docker ps
