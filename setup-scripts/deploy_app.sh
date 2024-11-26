#!/bin/bash

# Enable error handling - exit on any error
set -e

# Set variables
USER="rkeshri98"                  # SSH username for the VM
VM_IP="35.232.147.234"           # External IP address of the VM
REMOTE_DIR="/home/$USER/deployment"  # Remote directory for deployment
REPO_URL="https://github.com/MLOPS-IE7374-Fall2024-G9/mlops-project.git"  # GitHub repository URL
MODEL_SCRIPT="./model/scripts/mlflow_model_registry.py"  # Path to the script to fetch the latest model
BACKEND_DOCKERFILE="./backend/Dockerfile"  # Backend Dockerfile location
FRONTEND_DOCKERFILE="./frontend/Dockerfile"  # Frontend Dockerfile location
REQUIREMENTS_FILE="./airflow-config/requirements.txt"  # Path to requirements.txt
PASSWORD="mlops"

# Helper function to execute a command over SSH
ssh_exec() {
    sshpass -p $PASSWORD ssh $USER@$VM_IP "$1"
}

# Step 1: Check if directory exists, perform git pull or clone
echo "Ensuring repository is up-to-date on the remote machine..."
ssh_exec "
    if [ -d $REMOTE_DIR ]; then
        echo 'Directory $REMOTE_DIR exists, pulling latest changes...'
        cd $REMOTE_DIR && git pull
    else
        echo 'Directory $REMOTE_DIR does not exist, cloning repository...'
        git clone $REPO_URL $REMOTE_DIR
    fi
"

# Step 2: Check and build/run backend Docker container
echo "Checking and running the backend Docker container..."
ssh_exec "
    cd $REMOTE_DIR && \
    if [ \$(docker ps -q -f name=backend) ]; then
        echo 'Backend container is already running, stopping and removing it...'
        docker stop backend && docker rm backend
    fi
    echo 'Starting the backend container...'
    docker build -t backend -f $BACKEND_DOCKERFILE . && \
    docker run -d --name backend backend
"

# Step 3: Check and build/run frontend Docker container
echo "Checking and running the frontend Docker container..."
ssh_exec "
    cd $REMOTE_DIR && \
    if [ \$(docker ps -q -f name=frontend) ]; then
        echo 'Frontend container is already running, stopping and removing it...'
        docker stop frontend && docker rm frontend
    fi
    echo 'Starting the frontend container...'
    docker build -t frontend -f $FRONTEND_DOCKERFILE . && \
    docker run -d --name frontend frontend
"

echo "Deployment complete."
