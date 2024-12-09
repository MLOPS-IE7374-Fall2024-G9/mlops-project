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
PASSWORD=$(get_config_value "PASSWORD")
MODEL_SCRIPT=$(get_config_value "MODEL_SCRIPT") # Assuming the model script location in config

# Helper function to execute a command over SSH
ssh_exec() {
    sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USER@$VM_IP "$1"
}

# Step 1: Ensure we are connected to the remote machine
echo "Connecting to remote machine..."
ssh_exec "echo 'Connected to remote machine $VM_IP'"

# Step 2: Get the backend container ID (using the image name "backend")
echo "Fetching the backend Docker container ID..."
CONTAINER_ID=$(ssh_exec "
    docker ps -q -f ancestor=backend
")

if [ -z "$CONTAINER_ID" ]; then
    echo "Error: No running container found for the 'backend' image!"
    exit 1
fi

echo "Backend container ID: $CONTAINER_ID"

# Step 3: Execute the command inside the backend container
echo "Running command inside the backend Docker container..."
ssh_exec "
    cd $REMOTE_DIR && docker exec $CONTAINER_ID bash -c 'python model/scripts/mlflow_model_registry.py --operation fetch_latest'
"

echo "Model deployment complete."
