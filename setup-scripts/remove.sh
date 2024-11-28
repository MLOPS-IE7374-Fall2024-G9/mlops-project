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
PASSWORD=$(get_config_value "PASSWORD")

# Helper function to execute a command over SSH
ssh_exec() {
    sshpass -p "$PASSWORD" ssh "$USER@$VM_IP" "$1"
}

# Step 1: Remove Docker containers and images
echo "Stopping and removing Docker containers and images..."
ssh_exec "
    if command -v docker &> /dev/null; then
        echo 'Stopping all Docker containers...'
        docker stop \$(docker ps -q)
        echo 'Removing all Docker containers...'
        docker rm \$(docker ps -a -q)
        echo 'Removing all Docker images...'
        docker rmi \$(docker images -q)
    else
        echo 'Docker is not installed, skipping container and image removal.'
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

# Step 3: Uninstall Docker and clean up dependencies
echo "Uninstalling Docker and cleaning up packages..."
ssh_exec "
    if command -v docker &> /dev/null; then
        echo 'Uninstalling Docker...'
        sudo apt-get remove -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        sudo apt-get autoremove -y
        sudo apt-get clean
        echo 'Docker uninstalled successfully.'
    else
        echo 'Docker is not installed, skipping uninstallation.'
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

# Step 5: Optionally, remove Python packages if installed globally
echo "Optionally, you can remove globally installed Python packages..."
ssh_exec "
    if command -v pip3 &> /dev/null; then
        echo 'Uninstalling Python packages...'
        sudo pip3 freeze | xargs sudo pip3 uninstall -y
        echo 'Python packages uninstalled.'
    else
        echo 'pip is not installed, skipping.'
    fi
"

echo "Cleanup complete."
