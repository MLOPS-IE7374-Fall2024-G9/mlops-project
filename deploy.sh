#!/bin/bash

# Enable error handling - exit on any error
set -e

# Set variables
ZONE="us-central1-c"
INSTANCE_NAME="deployment-machine"
PROJECT="mlops-437516"
REMOTE_DIR="/home/user/deployment"
REPO_URL="https://github.com/MLOPS-IE7374-Fall2024-G9/mlops-project.git"  # GitHub repository URL
MODEL_SCRIPT="./model/scripts/mlflow_model_registry.py"  # Path to the script to fetch the latest model
BACKEND_DOCKERFILE="./backend/Dockerfile"  # Backend Dockerfile location
FRONTEND_DOCKERFILE="./frontend/Dockerfile"  # Frontend Dockerfile location
REQUIREMENTS_FILE="./airflow-config/requirements.txt"  # Path to requirements.txt for Airflow dependencies

# Step 1: Ensure the remote directory exists and remove it if it already exists
echo "Ensuring remote directory exists and removing if necessary..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    if [ -d $REMOTE_DIR ]; then
        echo 'Directory $REMOTE_DIR exists, removing it...'
        rm -rf $REMOTE_DIR
    fi
    mkdir -p $REMOTE_DIR
"

# Step 2: Git clone the repository to the remote system
echo "Cloning the repository to the remote machine..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    git clone $REPO_URL $REMOTE_DIR
"

# Step 3: Ensure Python and pip are installed on the remote machine
echo "Ensuring Python and pip are installed on the remote machine..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    sudo apt-get update
    if ! command -v python3 &> /dev/null; then
        echo 'Python3 not found, installing Python3...'
        sudo apt update && sudo apt install -y python3 python3-pip
    fi
    if ! command -v pip3 &> /dev/null; then
        echo 'pip not found, installing pip...'
        sudo apt install -y python3-pip
    fi
"

# Step 4: Change directory to the deployment folder and install Airflow dependencies
echo "Changing directory to $REMOTE_DIR and installing requirements..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    cd $REMOTE_DIR && \
    pip3 uninstall -r $REQUIREMENTS_FILE
    pip3 install -r $REQUIREMENTS_FILE
"

# Step 5: Run the model fetching script
echo "Running the model fetching script on the remote machine..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    cd $REMOTE_DIR && \
    python3 $MODEL_SCRIPT --operation fetch_latest
"

# Step 6: Build and run the backend Docker container
echo "Building and starting the backend Docker container..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    cd $REMOTE_DIR && \
    docker build -t backend -f $BACKEND_DOCKERFILE . && \
    docker run -d --name backend backend
"

# Step 7: Build and run the frontend Docker container
echo "Building and starting the frontend Docker container..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    cd $REMOTE_DIR && \
    docker build -t frontend -f $FRONTEND_DOCKERFILE . && \
    docker run -d --name frontend frontend
"

echo "Deployment complete."
