#!/bin/bash

# Define the source directory
SOURCE_DIR="airflow-config"

# Check if the source directory exists
if [ -d "$SOURCE_DIR" ]; then
    # Copy all files from the source directory to the current directory
    cp "$SOURCE_DIR"/* .
    echo "Files copied from $SOURCE_DIR to the current directory."
else
    echo "Source directory $SOURCE_DIR does not exist."
fi

# Create a .env file in the current directory if it doesn't exist
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    touch "$ENV_FILE"
    echo "Created .env file in the current directory."
else
    echo ".env file already exists in the current directory."
fi

# Set the environment variables
DEMAND_API_KEY="f8tGzRmnyw6dJyy1PyS49REmg1qrT2isvVi8i9mt"
WEATHER_API_KEY="820479673a8444f69ac162421242809"

# Add or overwrite DEMAND_API_KEY and WEATHER_API_KEY in .env
echo "Configuring environment variables in .env file."
if grep -q "^DEMAND_API_KEY=" "$ENV_FILE"; then
    sed -i "s/^DEMAND_API_KEY=.*/DEMAND_API_KEY=\"$DEMAND_API_KEY\"/" "$ENV_FILE"
else
    echo "DEMAND_API_KEY=\"$DEMAND_API_KEY\"" >> "$ENV_FILE"
fi

if grep -q "^WEATHER_API_KEY=" "$ENV_FILE"; then
    sed -i "s/^WEATHER_API_KEY=.*/WEATHER_API_KEY=\"$WEATHER_API_KEY\"/" "$ENV_FILE"
else
    echo "WEATHER_API_KEY=\"$WEATHER_API_KEY\"" >> "$ENV_FILE"
fi

echo ".env file setup complete with all configurations."

# Decrypt secrets.json if the encrypted file exists
ENCRYPTED_FILE="mlops-437516-b9a69694c897.json.enc"
DECRYPTED_FILE="mlops-437516-b9a69694c897.json"
DECRYPTION_PASSWORD="mlops-group-9"  # Replace with the actual password or prompt user input

if [ -f "$ENCRYPTED_FILE" ]; then
    if [ ! -f "$DECRYPTED_FILE" ]; then
        echo "Decrypting $ENCRYPTED_FILE to $DECRYPTED_FILE..."
        openssl enc -d -aes-256-cbc -in "$ENCRYPTED_FILE" -out "$DECRYPTED_FILE" -k "$DECRYPTION_PASSWORD"
        echo "Decryption complete."
    else
        echo "$DECRYPTED_FILE already exists. Skipping decryption."
    fi
else
    echo "Encrypted secrets file $ENCRYPTED_FILE not found."
fi

ENCRYPTED_FILE="mlops-7374-3e7424e80d76.json.enc"
DECRYPTED_FILE="mlops-7374-3e7424e80d76.json"
DECRYPTION_PASSWORD="mlops-group-9"  # Replace with the actual password or prompt user input

if [ -f "$ENCRYPTED_FILE" ]; then
    if [ ! -f "$DECRYPTED_FILE" ]; then
        echo "Decrypting $ENCRYPTED_FILE to $DECRYPTED_FILE..."
        openssl enc -d -aes-256-cbc -in "$ENCRYPTED_FILE" -out "$DECRYPTED_FILE" -k "$DECRYPTION_PASSWORD"
        echo "Decryption complete."
    else
        echo "$DECRYPTED_FILE already exists. Skipping decryption."
    fi
else
    echo "Encrypted secrets file $ENCRYPTED_FILE not found."
fi