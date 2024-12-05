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

# Helper function to execute a command over SSH
ssh_exec() {
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$VM_IP" "$1"
}

# Step 1: Ensure Docker is installed (if not already installed)
echo "Checking if Docker is installed on the remote machine..."
ssh_exec "
    if ! command -v docker &> /dev/null; then
        echo 'Docker not found, installing Docker...'
        
        # Remove old Docker packages
        for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do 
            sudo apt-get remove -y \$pkg
        done

        # Add Docker's official GPG key
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl
        sudo install -m 0755 -d /etc/apt/keyrings
        sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
        sudo chmod a+r /etc/apt/keyrings/docker.asc

        # Add Docker repository to Apt sources
        echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \$(. /etc/os-release && echo \"\$VERSION_CODENAME\") stable\" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update

        # Install Docker packages
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        sudo systemctl restart docker
        newgrp docker
    else
        echo 'Docker is already installed.'
    fi
"

# Step 2: Ensure Python, pip, Git, and virtualenv are installed on the remote machine
echo "Ensuring Python, pip, Git are installed on the remote machine..."
ssh_exec "
    if ! command -v python3 &> /dev/null; then
        echo 'Python3 not found, installing Python3...'
        sudo apt-get install -y python3
    fi
    if ! command -v pip3 &> /dev/null; then
        echo 'pip not found, installing pip...'
        sudo apt-get install -y python3-pip
    fi
    if ! command -v git &> /dev/null; then
        echo 'Git not found, installing Git...'
        sudo apt-get install -y git
    fi
"

# Step 3: Check if directory exists, perform git stash and then git pull
echo "Ensuring repository is up-to-date on the remote machine..."
ssh_exec "
    if [ -d $REMOTE_DIR ]; then
        echo 'Directory $REMOTE_DIR exists, stashing any local changes and pulling latest changes...'
        cd $REMOTE_DIR && \
        git stash && \
        git pull
    else
        echo 'Directory $REMOTE_DIR does not exist, cloning repository...'
        git clone $REPO_URL $REMOTE_DIR
    fi
"

# Step 4: Run setup.sh after repository update (if available)
echo "Running setup.sh script after repository update..."
ssh_exec "
    cd $REMOTE_DIR && \
    if [ -f setup.sh ]; then
        echo 'Running setup.sh...'
        chmod +x setup.sh && ./setup.sh
    else
        echo 'setup.sh not found in the repository.'
    fi
"

echo "Setup complete."
