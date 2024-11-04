#!/bin/bash

# Define the file to encrypt and the output encrypted file
FILE_TO_ENCRYPT="mlops_cloud_config.json"
ENCRYPTED_FILE="mlops-437516-b9a69694c897.json.enc"
ENCRYPTION_PASSWORD="mlops-group-9"  # Replace with the actual password or prompt user input

# Check if the file to encrypt exists
if [ -f "$FILE_TO_ENCRYPT" ]; then
    echo "Encrypting $FILE_TO_ENCRYPT to $ENCRYPTED_FILE..."
    
    # Encrypt the file
    openssl enc -aes-256-cbc -salt -in "$FILE_TO_ENCRYPT" -out "$ENCRYPTED_FILE" -k "$ENCRYPTION_PASSWORD"
    
    # Check if encryption was successful
    if [ $? -eq 0 ]; then
        echo "Encryption complete. Encrypted file: $ENCRYPTED_FILE"
    else
        echo "Encryption failed."
    fi
else
    echo "File $FILE_TO_ENCRYPT does not exist."
fi
