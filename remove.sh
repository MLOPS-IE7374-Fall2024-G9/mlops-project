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

echo "Deployment code removed successfully."
