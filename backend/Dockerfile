# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install git inside container
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*  # Clean up the apt cache to reduce image size

# Install dependencies for Ollama and curl
RUN apt-get update && apt-get install -y curl

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy the requirements file into the container at /app
COPY airflow-config/requirements.txt /app/

# Install the Python dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# # Run setup.sh
# RUN chmod +x ./setup.sh
# RUN ./setup.sh

# Init git and dvc
RUN git init

# # Configure DVC to use credentials from the JSON file
# RUN dvc remote modify --local storage credentialpath /mlops-437516-b9a69694c897.json
# RUN dvc pull --force

# run ollama server
RUN ollama serve & sleep 5 && ollama pull llama3-groq-tool-use

# Expose the port the app will run on
EXPOSE 8000
EXPOSE 11434 

# Start Ollama serve and Uvicorn in parallel using a script
COPY backend/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Command to run the app using Uvicorn
CMD ["/app/start.sh"]