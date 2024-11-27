#!/bin/bash

# Enable error handling - exit on any error
set -e

# Set variables
ZONE="us-central1-c"
INSTANCE_NAME="deployment-machine"
PROJECT="mlops-437516"
REMOTE_DIR="/home/user/deployment"  # Directory to remove

# Step 1: SSH into the remote system and remove the deployment directory
echo "Removing deployment directory on the remote machine..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    rm -rf $REMOTE_DIR
"

# Step 2: SSH into the remote system and stop and remove all Docker containers
echo "Stopping and removing all Docker containers..."
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" "$INSTANCE_NAME" --command "
    docker stop \$(docker ps -aq) 2>/dev/null || true  # Stop all containers (ignore errors if no containers)
    docker rm \$(docker ps -aq) 2>/dev/null || true    # Remove all containers (ignore errors if no containers)
"

echo "Deployment code removed and all Docker containers removed successfully."
