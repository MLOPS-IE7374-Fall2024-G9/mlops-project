#!/bin/bash

# Enable error handling - exit on any error
set -e

# Get the directory of the current script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Construct the path to config.json
CONFIG_FILE="$SCRIPT_DIR/config.json"

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
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$VM_IP" "$1"
}

# Step 1: Ensure the repository is up-to-date
echo "Ensuring repository is up-to-date on the remote machine..."
ssh_exec "
    if [ -d $REMOTE_DIR ]; then
        echo 'Directory $REMOTE_DIR exists, checking out production branch and pulling latest changes...'
        cd $REMOTE_DIR && \
        git fetch origin && \
        git checkout production && \
        git pull origin production
    else
        echo 'Directory $REMOTE_DIR does not exist, cloning repository...'
        git clone $REPO_URL $REMOTE_DIR && \
        cd $REMOTE_DIR && \
        git checkout production
    fi
"
# step 1a: Run setup.sh
echo "Running setup"
ssh_exec "
    cd $REMOTE_DIR && \
    echo 'Running setup' && \
    chmod +x setup.sh && ./setup.sh
"

# Step 2: Stop and remove all running Docker containers
echo "Stopping and removing all running Docker containers..."
ssh_exec "
    echo 'Stopping all containers...' && \
    docker stop \$(docker ps -q) || true && \
    echo 'Removing all containers...' && \
    docker rm \$(docker ps -aq) || true
"
# docker rmi -f \$(docker images -aq) || true

# Step 4: Build the frontend image
echo "Building the frontend Docker image..."
ssh_exec "
    cd $REMOTE_DIR && \
    echo 'Building the frontend Docker image...' && \
    docker build -t frontend -f $FRONTEND_DOCKERFILE .
"

# Step 5: Run the frontend container
echo "Running the frontend Docker container..."
ssh_exec "
    cd $REMOTE_DIR && \
    if [ \$(docker ps -q -f name=frontend) ]; then
        echo 'Stopping and removing existing frontend container...'
        docker stop frontend && docker rm frontend
    fi
    echo 'Starting the frontend container...' && \
    docker run -d -p 8501:8501 frontend
"

# Step 2: Build the backend image
echo "Building the backend Docker image..."
ssh_exec "
    cd $REMOTE_DIR && \
    echo 'Building the backend Docker image...' && \
    docker build -t backend -f $BACKEND_DOCKERFILE .
"

# Step 3: Run the backend container
echo "Running the backend Docker container..."
ssh_exec "
    cd $REMOTE_DIR && \
    if [ \$(docker ps -q -f name=backend) ]; then
        echo 'Stopping and removing existing backend container...'
        docker stop backend && docker rm backend
    fi
    echo 'Starting the backend container...' && \
    docker run -d -p 8000:8000 backend
"

echo "Deployment complete."
