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

# Helper function to execute a command over SSH
ssh_exec() {
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$VM_IP" "$1"
}

# Step 1: Remove all Docker resources
echo "Cleaning up all Docker containers, images, volumes, and networks..."
ssh_exec "
    if command -v docker &> /dev/null; then
        docker stop $(docker ps -q)
        docker rm $(docker ps -a -q)
        docker rmi $(docker images -q)
        echo 'Running docker system prune...'
        docker system prune -a --volumes -f
        sudo systemctl restart docker
        echo 'Docker cleanup complete.'
    else
        echo 'Docker is not installed, skipping cleanup.'
    fi
"

# Step 2: Delete the repository directory
echo "Deleting the repository directory: $REMOTE_DIR..."
ssh_exec "
    if [ -d $REMOTE_DIR ]; then
        rm -rf $REMOTE_DIR
        echo 'Repository directory removed.'
    else
        echo 'Repository directory does not exist, skipping.'
    fi
"

# Step 4: Remove Python virtual environment (if exists)
echo "Removing Python virtual environment (if exists)..."
ssh_exec "
    if [ -d $REMOTE_DIR/venv ]; then
        rm -rf $REMOTE_DIR/venv
        echo 'Virtual environment removed.'
    else
        echo 'Virtual environment does not exist, skipping removal.'
    fi
"

echo "Cleanup complete."
