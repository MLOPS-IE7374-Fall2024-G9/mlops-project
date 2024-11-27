#!/bin/bash

# Enable error handling - exit on any error
set -e

# Load configuration from config.json
CONFIG_FILE="./config.json"

# Helper function to fetch values from the config file
get_config_value() {
    jq -r ".$1" "$CONFIG_FILE"
}

# Load variables from the configuration file
USER=$(get_config_value "USER")
VM_IP=$(get_config_value "VM_IP")
REMOTE_DIR=$(get_config_value "REMOTE_DIR")
REPO_URL=$(get_config_value "REPO_URL")
REQUIREMENTS_FILE=$(get_config_value "REQUIREMENTS_FILE")
PASSWORD=$(get_config_value "PASSWORD")
MODEL_SCRIPT=$(get_config_value "MODEL_SCRIPT")
BACKEND_SCRIPT=$(get_config_value "BACKEND_SCRIPT") 
BACKEND_DOCKERFILE=$(get_config_value "BACKEND_DOCKERFILE") 
FRONTEND_DOCKERFILE=$(get_config_value "FRONTEND_DOCKERFILE") 

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

# Step 2: Build and run the backend using the Dockerfile
echo "Building and running the backend Docker container..."
ssh_exec "
    cd $REMOTE_DIR && \
    if [ \$(docker ps -q -f name=backend) ]; then
        echo 'Backend container is already running, stopping and removing it...'
        docker stop backend && docker rm backend
    fi
    echo 'Building the backend Docker image...'
    docker build -t backend -f $BACKEND_DOCKERFILE . && \
    echo 'Running the backend container...' && \
    docker run -d -p 8000:8000 backend
"

# # Step 2: Check and run the backend application in the background
# echo "Checking and running the backend application..."
# ssh_exec "
#     cd $REMOTE_DIR && \
#     echo 'Starting the backend application in background...' && \
#     nohup bash -c 'source venv/bin/activate && uvicorn backend.app:app --host 0.0.0.0 --port 8000'
# "

# # Step 5: Check and build/run frontend Docker container
# echo "Checking and running the frontend Docker container..."
# ssh_exec "
#     cd $REMOTE_DIR && \
#     if [ \$(docker ps -q -f name=frontend) ]; then
#         echo 'Frontend container is already running, stopping and removing it...'
#         docker stop frontend && docker rm frontend
#     fi
#     echo 'Starting the frontend container...'
#     docker build -t frontend -f $FRONTEND_DOCKERFILE . && \
#     docker run -d --name frontend frontend
# "

echo "Deployment complete."
