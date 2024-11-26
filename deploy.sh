#!/bin/bash

# Enable error handling - exit on any error
set -e

# Set variables
ZONE="us-central1-c"
INSTANCE_NAME="deployment-machine"
PROJECT="mlops-437516"
REMOTE_DIR="/home/user/deployment"
LOCAL_CODE_DIR="./"  # Local directory containing your code
MODEL_SCRIPT="./model/mflow_model_registry.py"  # Path to the script to fetch the latest model
BACKEND_DOCKERFILE="./backend/Dockerfile"  # Backend Dockerfile location
FRONTEND_DOCKERFILE="./frontend/Dockerfile"  # Frontend Dockerfile location

# Step 1: SSH into the remote system and create the deployment directory
echo "Creating deployment directory on the remote machine..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    mkdir -p $REMOTE_DIR
"

# Step 2: Copy the code base to the remote system (VM instance)
echo "Copying code base to the remote machine..."
gcloud compute scp --zone "$ZONE" --project "$PROJECT" --recurse "$LOCAL_CODE_DIR" "$INSTANCE_NAME:$REMOTE_DIR"

# Step 3: SSH into the remote system and run the model fetching script
echo "Running the model fetching script on the remote machine..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    cd $REMOTE_DIR && \
    python3 $MODEL_SCRIPT --operation fetch_latest
"

# Step 4: Build and run the backend Docker container
echo "Building and starting the backend Docker container..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    cd $REMOTE_DIR && \
    docker build -t backend -f $BACKEND_DOCKERFILE . && \
    docker run -d --name backend backend
"

# Step 5: Build and run the frontend Docker container
echo "Building and starting the frontend Docker container..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    cd $REMOTE_DIR && \
    docker build -t frontend -f $FRONTEND_DOCKERFILE . && \
    docker run -d --name frontend frontend
"

echo "Deployment complete."
