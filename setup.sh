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

# Add environment variables to the .env file
echo "Adding environment variables to .env file."
echo "DEMAND_API_KEY=\"f8tGzRmnyw6dJyy1PyS49REmg1qrT2isvVi8i9mt\"" >> "$ENV_FILE"
echo "WEATHER_API_KEY=\"820479673a8444f69ac162421242809\"" >> "$ENV_FILE"

echo ".env file setup complete with all configurations."


